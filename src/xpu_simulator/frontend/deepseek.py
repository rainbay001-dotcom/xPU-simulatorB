"""Build an abstract graph from DeepSeek-style config."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..graph_passes import apply_default_passes
from ..ir.graph import Graph, Node, TensorDesc
from ..ir.types import DType, OpKind
from .source_analysis import DeepSeekSourceAnalyzer


@dataclass
class DeepSeekConfig:
    vocab_size: int
    dim: int
    inter_dim: int
    moe_inter_dim: int
    n_layers: int
    n_dense_layers: int
    n_heads: int
    n_routed_experts: int
    n_shared_experts: int
    n_activated_experts: int
    qk_rope_head_dim: int
    v_head_dim: int
    dtype: DType

    @classmethod
    def from_json(cls, path: str | Path) -> "DeepSeekConfig":
        raw = json.loads(Path(path).read_text())
        return cls(
            vocab_size=raw["vocab_size"],
            dim=raw["dim"],
            inter_dim=raw["inter_dim"],
            moe_inter_dim=raw["moe_inter_dim"],
            n_layers=raw["n_layers"],
            n_dense_layers=raw["n_dense_layers"],
            n_heads=raw["n_heads"],
            n_routed_experts=raw["n_routed_experts"],
            n_shared_experts=raw["n_shared_experts"],
            n_activated_experts=raw["n_activated_experts"],
            qk_rope_head_dim=raw["qk_rope_head_dim"],
            v_head_dim=raw["v_head_dim"],
            dtype=DType(raw.get("dtype", "bf16")),
        )


class DeepSeekGraphBuilder:
    """Create a coarse graph suitable for Phase 1 simulation."""

    def __init__(self, config: DeepSeekConfig, source_path: str | Path | None = None):
        self.config = config
        self.source_path = Path(source_path) if source_path is not None else None
        self.source_features = None
        if self.source_path is not None:
            self.source_features = DeepSeekSourceAnalyzer().analyze(self.source_path)

    def build_graph(
        self,
        batch_size: int = 1,
        seq_len: int = 128,
        layers: int | None = None,
        layer_start: int = 0,
    ) -> Graph:
        layer_count = layers if layers is not None else self.config.n_layers
        layer_stop = min(layer_start + layer_count, self.config.n_layers)
        graph = Graph(name=f"deepseek_b{batch_size}_s{seq_len}_l{layer_start}-{layer_stop}")
        graph.metadata["model_family"] = "deepseek"
        graph.metadata["batch_size"] = batch_size
        graph.metadata["seq_len"] = seq_len
        graph.metadata["dtype"] = self.config.dtype.value
        graph.metadata["layer_start"] = layer_start
        graph.metadata["layer_stop"] = layer_stop
        if self.source_path is not None:
            graph.metadata["source_path"] = str(self.source_path)
        if self.source_features is not None:
            graph.metadata["source_features"] = self.source_features.to_metadata()
        prev: Node | None = None

        embed = self._make_embedding_node(batch_size, seq_len)
        graph.add_node(embed)
        prev = embed

        for idx in range(layer_start, layer_stop):
            layer_nodes, layer_entry, layer_exit = self._build_transformer_layer(batch_size, seq_len, idx)
            for node in layer_nodes:
                graph.add_node(node)
            if prev is not None:
                graph.add_edge(prev, layer_entry)
            for src_name, dst_name in self._layer_edges(layer_nodes, prefix=f"layer_{idx}"):
                graph.add_edge(self._find_node(layer_nodes, src_name), self._find_node(layer_nodes, dst_name))
            prev = layer_exit

        output = self._make_output_node(batch_size, seq_len)
        graph.add_node(output)
        if prev is not None:
            graph.add_edge(prev, output)
        return apply_default_passes(graph)

    def _make_embedding_node(self, batch_size: int, seq_len: int) -> Node:
        tokens = TensorDesc((batch_size, seq_len), DType.INT8)
        hidden = TensorDesc((batch_size, seq_len, self.config.dim), self.config.dtype)
        bytes_moved = tokens.size_bytes + hidden.size_bytes
        return Node(
            name="embedding",
            op_kind=OpKind.EMBEDDING,
            inputs=[tokens],
            outputs=[hidden],
            attrs={
                "kernel_family": "lookup",
                "parallel_vocab": bool(self.source_features and self.source_features.uses_distributed),
            },
            bytes_moved=bytes_moved,
        )

    def _build_transformer_layer(self, batch_size: int, seq_len: int, layer_idx: int) -> tuple[list[Node], Node, Node]:
        hidden = TensorDesc((batch_size, seq_len, self.config.dim), self.config.dtype)
        attn_proj = TensorDesc((batch_size, seq_len, self.config.n_heads * self.config.v_head_dim), self.config.dtype)
        routed = TensorDesc((batch_size, seq_len, self.config.n_activated_experts), DType.INT8)
        prefix = f"layer_{layer_idx}"
        dense_layer = layer_idx < self.config.n_dense_layers

        nodes = [
            Node(f"{prefix}_attn_norm", OpKind.NORM, [hidden], [hidden], bytes_moved=hidden.size_bytes * 2),
            Node(
                f"{prefix}_qkv_proj",
                OpKind.MATMUL,
                [hidden],
                [attn_proj],
                attrs={
                    "uses_fp8": bool(self.source_features and self.source_features.uses_fp8),
                    "parallelism": "column"
                    if self.source_features and self.source_features.parallel_linears
                    else "none",
                },
                flops=6 * batch_size * seq_len * self.config.dim * self.config.dim,
                bytes_moved=hidden.size_bytes + attn_proj.size_bytes,
            ),
            Node(
                f"{prefix}_rope",
                OpKind.ROPE,
                [attn_proj],
                [attn_proj],
                attrs={"rope_style": "deepseek_yarn" if self.source_features and self.source_features.uses_rotary else "none"},
                bytes_moved=attn_proj.size_bytes * 2,
            ),
            Node(
                f"{prefix}_attn_scores",
                OpKind.BATCHED_MATMUL,
                [attn_proj],
                [TensorDesc((batch_size, self.config.n_heads, seq_len, seq_len), self.config.dtype)],
                flops=4 * batch_size * self.config.n_heads * seq_len * seq_len * self.config.v_head_dim,
                bytes_moved=attn_proj.size_bytes * 2,
            ),
            Node(
                f"{prefix}_softmax",
                OpKind.SOFTMAX,
                [TensorDesc((batch_size, self.config.n_heads, seq_len, seq_len), self.config.dtype)],
                [TensorDesc((batch_size, self.config.n_heads, seq_len, seq_len), self.config.dtype)],
                bytes_moved=4 * batch_size * self.config.n_heads * seq_len * seq_len * hidden.bytes_per_element,
            ),
            Node(
                f"{prefix}_attn_out",
                OpKind.BATCHED_MATMUL,
                [attn_proj],
                [hidden],
                flops=4 * batch_size * self.config.n_heads * seq_len * seq_len * self.config.v_head_dim,
                bytes_moved=attn_proj.size_bytes + hidden.size_bytes,
            ),
            Node(f"{prefix}_ffn_norm", OpKind.NORM, [hidden], [hidden], bytes_moved=hidden.size_bytes * 2),
        ]

        if dense_layer:
            nodes.extend(
                [
                    Node(
                        f"{prefix}_ffn_w1",
                        OpKind.MATMUL,
                        [hidden],
                        [TensorDesc((batch_size, seq_len, self.config.inter_dim), self.config.dtype)],
                        attrs={"parallelism": "column"},
                        flops=2 * batch_size * seq_len * self.config.dim * self.config.inter_dim,
                        bytes_moved=hidden.size_bytes,
                    ),
                    Node(
                        f"{prefix}_ffn_w3",
                        OpKind.MATMUL,
                        [hidden],
                        [TensorDesc((batch_size, seq_len, self.config.inter_dim), self.config.dtype)],
                        attrs={"parallelism": "column"},
                        flops=2 * batch_size * seq_len * self.config.dim * self.config.inter_dim,
                        bytes_moved=hidden.size_bytes,
                    ),
                    Node(
                        f"{prefix}_ffn_gate",
                        OpKind.ELEMENTWISE,
                        [
                            TensorDesc((batch_size, seq_len, self.config.inter_dim), self.config.dtype),
                            TensorDesc((batch_size, seq_len, self.config.inter_dim), self.config.dtype),
                        ],
                        [TensorDesc((batch_size, seq_len, self.config.inter_dim), self.config.dtype)],
                        bytes_moved=4 * batch_size * seq_len * self.config.inter_dim * hidden.bytes_per_element,
                    ),
                    Node(
                        f"{prefix}_ffn_down",
                        OpKind.MATMUL,
                        [TensorDesc((batch_size, seq_len, self.config.inter_dim), self.config.dtype)],
                        [hidden],
                        attrs={"parallelism": "row"},
                        flops=2 * batch_size * seq_len * self.config.inter_dim * self.config.dim,
                        bytes_moved=hidden.size_bytes,
                    ),
                ]
            )
        else:
            nodes.extend(
                [
                    Node(
                        f"{prefix}_router",
                        OpKind.TOPK,
                        [hidden],
                        [routed],
                        attrs={"moe": True, "routing": "topk"},
                        bytes_moved=hidden.size_bytes + routed.size_bytes,
                    ),
                    Node(
                        f"{prefix}_dispatch",
                        OpKind.GATHER,
                        [hidden, routed],
                        [TensorDesc((batch_size, seq_len, self.config.moe_inter_dim), self.config.dtype)],
                        attrs={"moe": True, "dispatch": "expert_gather"},
                        bytes_moved=hidden.size_bytes,
                    ),
                    Node(
                        f"{prefix}_expert_ffn",
                        OpKind.MATMUL,
                        [TensorDesc((batch_size, seq_len, self.config.moe_inter_dim), self.config.dtype)],
                        [TensorDesc((batch_size, seq_len, self.config.moe_inter_dim), self.config.dtype)],
                        attrs={
                            "moe": True,
                            "expert_count": self.config.n_activated_experts,
                            "uses_fp8": bool(self.source_features and self.source_features.uses_fp8),
                        },
                        flops=2
                        * batch_size
                        * seq_len
                        * self.config.moe_inter_dim
                        * self.config.dim
                        * self.config.n_activated_experts,
                        bytes_moved=batch_size * seq_len * self.config.moe_inter_dim * hidden.bytes_per_element,
                    ),
                    Node(
                        f"{prefix}_shared_ffn_w1",
                        OpKind.MATMUL,
                        [hidden],
                        [TensorDesc((batch_size, seq_len, self.config.n_shared_experts * self.config.moe_inter_dim), self.config.dtype)],
                        attrs={"moe": True, "shared_experts": True, "parallelism": "column"},
                        flops=2
                        * batch_size
                        * seq_len
                        * self.config.dim
                        * (self.config.n_shared_experts * self.config.moe_inter_dim),
                        bytes_moved=hidden.size_bytes,
                    ),
                    Node(
                        f"{prefix}_shared_ffn_w3",
                        OpKind.MATMUL,
                        [hidden],
                        [TensorDesc((batch_size, seq_len, self.config.n_shared_experts * self.config.moe_inter_dim), self.config.dtype)],
                        attrs={"moe": True, "shared_experts": True, "parallelism": "column"},
                        flops=2
                        * batch_size
                        * seq_len
                        * self.config.dim
                        * (self.config.n_shared_experts * self.config.moe_inter_dim),
                        bytes_moved=hidden.size_bytes,
                    ),
                    Node(
                        f"{prefix}_shared_ffn_gate",
                        OpKind.ELEMENTWISE,
                        [
                            TensorDesc((batch_size, seq_len, self.config.n_shared_experts * self.config.moe_inter_dim), self.config.dtype),
                            TensorDesc((batch_size, seq_len, self.config.n_shared_experts * self.config.moe_inter_dim), self.config.dtype),
                        ],
                        [TensorDesc((batch_size, seq_len, self.config.n_shared_experts * self.config.moe_inter_dim), self.config.dtype)],
                        attrs={"moe": True, "shared_experts": True},
                        bytes_moved=4
                        * batch_size
                        * seq_len
                        * (self.config.n_shared_experts * self.config.moe_inter_dim)
                        * hidden.bytes_per_element,
                    ),
                    Node(
                        f"{prefix}_shared_ffn_down",
                        OpKind.MATMUL,
                        [TensorDesc((batch_size, seq_len, self.config.n_shared_experts * self.config.moe_inter_dim), self.config.dtype)],
                        [hidden],
                        attrs={"moe": True, "shared_experts": True, "parallelism": "row"},
                        flops=2
                        * batch_size
                        * seq_len
                        * (self.config.n_shared_experts * self.config.moe_inter_dim)
                        * self.config.dim,
                        bytes_moved=hidden.size_bytes,
                    ),
                    Node(
                        f"{prefix}_combine",
                        OpKind.SCATTER,
                        [
                            TensorDesc((batch_size, seq_len, self.config.moe_inter_dim), self.config.dtype),
                            hidden,
                        ],
                        [hidden],
                        attrs={"moe": True, "dispatch": "expert_scatter"},
                        bytes_moved=hidden.size_bytes,
                    ),
                ]
            )

        if self.source_features and self.source_features.uses_distributed:
            nodes.append(
                Node(
                    f"{prefix}_tensor_parallel_sync",
                    OpKind.ALL_REDUCE,
                    [hidden],
                    [hidden],
                    attrs={"distributed": True},
                    bytes_moved=hidden.size_bytes,
                )
            )

        exit_name = f"{prefix}_tensor_parallel_sync" if self.source_features and self.source_features.uses_distributed else (
            f"{prefix}_ffn_down" if dense_layer else f"{prefix}_combine"
        )
        entry = self._find_node(nodes, f"{prefix}_attn_norm")
        exit_node = self._find_node(nodes, exit_name)
        return nodes, entry, exit_node

    def _layer_edges(self, nodes: list[Node], prefix: str) -> list[tuple[str, str]]:
        names = {node.name for node in nodes}
        edges = [
            (f"{prefix}_attn_norm", f"{prefix}_qkv_proj"),
            (f"{prefix}_qkv_proj", f"{prefix}_rope"),
            (f"{prefix}_rope", f"{prefix}_attn_scores"),
            (f"{prefix}_attn_scores", f"{prefix}_softmax"),
            (f"{prefix}_softmax", f"{prefix}_attn_out"),
            (f"{prefix}_attn_out", f"{prefix}_ffn_norm"),
        ]
        if f"{prefix}_ffn_w1" in names:
            edges.extend(
                [
                    (f"{prefix}_ffn_norm", f"{prefix}_ffn_w1"),
                    (f"{prefix}_ffn_norm", f"{prefix}_ffn_w3"),
                    (f"{prefix}_ffn_w1", f"{prefix}_ffn_gate"),
                    (f"{prefix}_ffn_w3", f"{prefix}_ffn_gate"),
                    (f"{prefix}_ffn_gate", f"{prefix}_ffn_down"),
                ]
            )
        else:
            edges.extend(
                [
                    (f"{prefix}_ffn_norm", f"{prefix}_router"),
                    (f"{prefix}_ffn_norm", f"{prefix}_shared_ffn_w1"),
                    (f"{prefix}_ffn_norm", f"{prefix}_shared_ffn_w3"),
                    (f"{prefix}_router", f"{prefix}_dispatch"),
                    (f"{prefix}_dispatch", f"{prefix}_expert_ffn"),
                    (f"{prefix}_expert_ffn", f"{prefix}_combine"),
                    (f"{prefix}_shared_ffn_w1", f"{prefix}_shared_ffn_gate"),
                    (f"{prefix}_shared_ffn_w3", f"{prefix}_shared_ffn_gate"),
                    (f"{prefix}_shared_ffn_gate", f"{prefix}_shared_ffn_down"),
                    (f"{prefix}_shared_ffn_down", f"{prefix}_combine"),
                ]
            )
        if f"{prefix}_tensor_parallel_sync" in names:
            if f"{prefix}_ffn_down" in names:
                edges.append((f"{prefix}_ffn_down", f"{prefix}_tensor_parallel_sync"))
            else:
                edges.append((f"{prefix}_combine", f"{prefix}_tensor_parallel_sync"))
        return [(src, dst) for src, dst in edges if src in names and dst in names]

    def _find_node(self, nodes: list[Node], name: str) -> Node:
        for node in nodes:
            if node.name == name:
                return node
        raise KeyError(name)

    def _make_output_node(self, batch_size: int, seq_len: int) -> Node:
        hidden = TensorDesc((batch_size, seq_len, self.config.dim), self.config.dtype)
        logits = TensorDesc((batch_size, seq_len, self.config.vocab_size), self.config.dtype)
        return Node(
            name="lm_head",
            op_kind=OpKind.OUTPUT,
            inputs=[hidden],
            outputs=[logits],
            flops=2 * batch_size * seq_len * self.config.dim * self.config.vocab_size,
            bytes_moved=hidden.size_bytes + logits.size_bytes,
        )
