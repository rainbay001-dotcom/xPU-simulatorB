"""Build an abstract graph from config plus extracted source architecture."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..graph_passes import apply_default_passes
from ..ir.graph import Graph, Node, TensorDesc
from ..ir.types import DType, OpKind
from .source_analysis import ArchitectureDescriptor, DeepSeekSourceAnalyzer


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
    """Build a simulator graph from config plus source-extracted architecture."""

    def __init__(self, config: DeepSeekConfig, source_path: str | Path | None = None):
        self.config = config
        self.source_path = Path(source_path) if source_path is not None else None
        self.source_analyzer = DeepSeekSourceAnalyzer()
        self.source_features = None
        self.architecture = ArchitectureDescriptor(
            model_class=None,
            block_class=None,
            attention_class=None,
            dense_ffn_class=None,
            moe_ffn_class=None,
            embedding_attr=None,
            layers_attr=None,
            norm_attr=None,
            head_attr=None,
            block_norm_attrs=[],
            attention_traits={},
            dense_ffn_traits={},
            moe_ffn_traits={},
        )
        if self.source_path is not None:
            self.source_features = self.source_analyzer.analyze(self.source_path)
            self.architecture = self.source_analyzer.extract_architecture(self.source_path)

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
        graph.metadata["architecture"] = self.architecture.to_metadata()
        if self.source_path is not None:
            graph.metadata["source_path"] = str(self.source_path)
        if self.source_features is not None:
            graph.metadata["source_features"] = self.source_features.to_metadata()

        prev: Node | None = None
        embed = self._make_embedding_node(batch_size, seq_len)
        graph.add_node(embed)
        prev = embed

        for idx in range(layer_start, layer_stop):
            layer_nodes, layer_edges, layer_entry, layer_exit = self._build_transformer_layer(batch_size, seq_len, idx)
            for node in layer_nodes:
                graph.add_node(node)
            if prev is not None:
                graph.add_edge(prev, layer_entry)
            for src, dst in layer_edges:
                graph.add_edge(src, dst)
            prev = layer_exit

        output = self._make_output_node(batch_size, seq_len)
        graph.add_node(output)
        if prev is not None:
            graph.add_edge(prev, output)
        return apply_default_passes(graph)

    def _make_embedding_node(self, batch_size: int, seq_len: int) -> Node:
        tokens = TensorDesc((batch_size, seq_len), DType.INT8)
        hidden = self._hidden_tensor(batch_size, seq_len)
        return Node(
            name="embedding",
            op_kind=OpKind.EMBEDDING,
            inputs=[tokens],
            outputs=[hidden],
            attrs={
                "source_attr": self.architecture.embedding_attr,
                "parallel_vocab": bool(self.source_features and self.source_features.uses_distributed),
            },
            bytes_moved=tokens.size_bytes + hidden.size_bytes,
        )

    def _build_transformer_layer(
        self,
        batch_size: int,
        seq_len: int,
        layer_idx: int,
    ) -> tuple[list[Node], list[tuple[Node, Node]], Node, Node]:
        prefix = f"layer_{layer_idx}"
        dense_layer = self._is_dense_layer(layer_idx)
        hidden = self._hidden_tensor(batch_size, seq_len)

        nodes: list[Node] = []
        edges: list[tuple[Node, Node]] = []

        attn_norm_name = self._block_norm_name("attn", "attn_norm")
        ffn_norm_name = self._block_norm_name("ffn", "ffn_norm")
        attn_norm = self._add_node(nodes, f"{prefix}_{attn_norm_name}", OpKind.NORM, [hidden], [hidden], bytes_moved=hidden.size_bytes * 2)

        attention_nodes = self._build_attention_subgraph(prefix, batch_size, seq_len, hidden)
        nodes.extend(attention_nodes)
        self._chain(edges, [attn_norm] + attention_nodes)

        ffn_norm = self._add_node(nodes, f"{prefix}_{ffn_norm_name}", OpKind.NORM, [hidden], [hidden], bytes_moved=hidden.size_bytes * 2)
        edges.append((attention_nodes[-1], ffn_norm))

        if dense_layer:
            dense_nodes, dense_edges, dense_exit = self._build_dense_ffn_subgraph(prefix, batch_size, seq_len, hidden, ffn_norm)
            nodes.extend(dense_nodes)
            edges.extend(dense_edges)
            exit_node = dense_exit
        else:
            moe_nodes, moe_edges, moe_exit = self._build_moe_subgraph(prefix, batch_size, seq_len, hidden, ffn_norm)
            nodes.extend(moe_nodes)
            edges.extend(moe_edges)
            exit_node = moe_exit

        if self.source_features and self.source_features.uses_distributed:
            sync = self._add_node(
                nodes,
                f"{prefix}_tensor_parallel_sync",
                OpKind.ALL_REDUCE,
                [hidden],
                [hidden],
                attrs={"distributed": True},
                bytes_moved=hidden.size_bytes,
            )
            edges.append((exit_node, sync))
            exit_node = sync

        return nodes, edges, attn_norm, exit_node

    def _build_attention_subgraph(self, prefix: str, batch_size: int, seq_len: int, hidden: TensorDesc) -> list[Node]:
        attn_proj = TensorDesc((batch_size, seq_len, self.config.n_heads * self.config.v_head_dim), self.config.dtype)
        score_tensor = TensorDesc((batch_size, self.config.n_heads, seq_len, seq_len), self.config.dtype)
        traits = self.architecture.attention_traits
        nodes: list[Node] = []

        qkv_proj = self._add_node(
            nodes,
            f"{prefix}_qkv_proj",
            OpKind.MATMUL,
            [hidden],
            [attn_proj],
            attrs={
                "uses_fp8": bool(self.source_features and self.source_features.uses_fp8),
                "source_class": self.architecture.attention_class,
                "projection_count": traits.get("projection_count", 0),
            },
            flops=6 * batch_size * seq_len * self.config.dim * self.config.dim,
            bytes_moved=hidden.size_bytes + attn_proj.size_bytes,
        )
        prev = qkv_proj

        if traits.get("has_q_norm", False):
            q_norm = self._add_node(nodes, f"{prefix}_q_norm", OpKind.NORM, [attn_proj], [attn_proj], bytes_moved=attn_proj.size_bytes * 2)
            prev = q_norm
        if traits.get("has_kv_norm", False):
            kv_norm = self._add_node(nodes, f"{prefix}_kv_norm", OpKind.NORM, [attn_proj], [attn_proj], bytes_moved=attn_proj.size_bytes * 2)
            prev = kv_norm

        if self.source_features and self.source_features.uses_rotary:
            rope = self._add_node(
                nodes,
                f"{prefix}_rope",
                OpKind.ROPE,
                [attn_proj],
                [attn_proj],
                attrs={"rope_style": "source_extracted"},
                bytes_moved=attn_proj.size_bytes * 2,
            )
            prev = rope

        attn_scores = self._add_node(
            nodes,
            f"{prefix}_attn_scores",
            OpKind.BATCHED_MATMUL,
            [attn_proj],
            [score_tensor],
            flops=4 * batch_size * self.config.n_heads * seq_len * seq_len * self.config.v_head_dim,
            bytes_moved=attn_proj.size_bytes * 2,
        )
        nodes.append(attn_scores) if attn_scores not in nodes else None

        if self.source_features and self.source_features.uses_topk and traits.get("has_indexer", False):
            attn_topk = self._add_node(
                nodes,
                f"{prefix}_attn_topk",
                OpKind.TOPK,
                [score_tensor],
                [TensorDesc((batch_size, seq_len, min(seq_len, 2048)), DType.INT8)],
                attrs={"attention_masking": True},
                bytes_moved=score_tensor.size_bytes,
            )
        else:
            attn_topk = None

        softmax = self._add_node(
            nodes,
            f"{prefix}_softmax",
            OpKind.SOFTMAX,
            [score_tensor],
            [score_tensor],
            flops=5 * batch_size * self.config.n_heads * seq_len * seq_len,
            bytes_moved=4 * batch_size * self.config.n_heads * seq_len * seq_len * hidden.bytes_per_element,
        )
        attn_out = self._add_node(
            nodes,
            f"{prefix}_attn_out",
            OpKind.BATCHED_MATMUL,
            [attn_proj],
            [hidden],
            flops=4 * batch_size * self.config.n_heads * seq_len * seq_len * self.config.v_head_dim,
            bytes_moved=attn_proj.size_bytes + hidden.size_bytes,
        )

        chain = [qkv_proj]
        if traits.get("has_q_norm", False):
            chain.append(self._find_node(nodes, f"{prefix}_q_norm"))
        if traits.get("has_kv_norm", False):
            chain.append(self._find_node(nodes, f"{prefix}_kv_norm"))
        if self.source_features and self.source_features.uses_rotary:
            chain.append(self._find_node(nodes, f"{prefix}_rope"))
        chain.append(attn_scores)
        if attn_topk is not None:
            chain.append(attn_topk)
        chain.extend([softmax, attn_out])
        # Reorder list to preserve chain order without duplicates.
        ordered_nodes: list[Node] = []
        seen: set[str] = set()
        for node in chain:
            if node.name not in seen:
                ordered_nodes.append(node)
                seen.add(node.name)
        return ordered_nodes

    def _build_dense_ffn_subgraph(
        self,
        prefix: str,
        batch_size: int,
        seq_len: int,
        hidden: TensorDesc,
        entry: Node,
    ) -> tuple[list[Node], list[tuple[Node, Node]], Node]:
        inter = TensorDesc((batch_size, seq_len, self.config.inter_dim), self.config.dtype)
        nodes: list[Node] = []
        edges: list[tuple[Node, Node]] = []
        traits = self.architecture.dense_ffn_traits

        w1 = self._add_node(
            nodes,
            f"{prefix}_ffn_w1",
            OpKind.MATMUL,
            [hidden],
            [inter],
            attrs={"parallelism": "column", "source_class": self.architecture.dense_ffn_class},
            flops=2 * batch_size * seq_len * self.config.dim * self.config.inter_dim,
            bytes_moved=hidden.size_bytes,
        )
        edges.append((entry, w1))

        if traits.get("has_gate_branch", True):
            w3 = self._add_node(
                nodes,
                f"{prefix}_ffn_w3",
                OpKind.MATMUL,
                [hidden],
                [inter],
                attrs={"parallelism": "column", "source_class": self.architecture.dense_ffn_class},
                flops=2 * batch_size * seq_len * self.config.dim * self.config.inter_dim,
                bytes_moved=hidden.size_bytes,
            )
            gate = self._add_node(
                nodes,
                f"{prefix}_ffn_gate",
                OpKind.ELEMENTWISE,
                [inter, inter],
                [inter],
                bytes_moved=4 * batch_size * seq_len * self.config.inter_dim * hidden.bytes_per_element,
            )
            down_input = gate
            edges.extend([(entry, w3), (w1, gate), (w3, gate)])
        else:
            down_input = w1

        down = self._add_node(
            nodes,
            f"{prefix}_ffn_down",
            OpKind.MATMUL,
            [inter],
            [hidden],
            attrs={"parallelism": "row", "source_class": self.architecture.dense_ffn_class},
            flops=2 * batch_size * seq_len * self.config.inter_dim * self.config.dim,
            bytes_moved=hidden.size_bytes,
        )
        edges.append((down_input, down))
        return nodes, edges, down

    def _build_moe_subgraph(
        self,
        prefix: str,
        batch_size: int,
        seq_len: int,
        hidden: TensorDesc,
        entry: Node,
    ) -> tuple[list[Node], list[tuple[Node, Node]], Node]:
        routed = TensorDesc((batch_size, seq_len, self.config.n_activated_experts), DType.INT8)
        expert_hidden = TensorDesc((batch_size, seq_len, self.config.moe_inter_dim), self.config.dtype)
        shared_hidden = TensorDesc(
            (batch_size, seq_len, self.config.n_shared_experts * self.config.moe_inter_dim),
            self.config.dtype,
        )
        nodes: list[Node] = []
        edges: list[tuple[Node, Node]] = []

        router = self._add_node(
            nodes,
            f"{prefix}_router",
            OpKind.TOPK,
            [hidden],
            [routed],
            attrs={"moe": True, "routing": "topk", "source_class": self.architecture.moe_ffn_class},
            bytes_moved=hidden.size_bytes + routed.size_bytes,
        )
        dispatch = self._add_node(
            nodes,
            f"{prefix}_dispatch",
            OpKind.GATHER,
            [hidden, routed],
            [expert_hidden],
            attrs={"moe": True, "dispatch": "expert_gather", "source_class": self.architecture.moe_ffn_class},
            bytes_moved=hidden.size_bytes,
        )
        expert_ffn = self._add_node(
            nodes,
            f"{prefix}_expert_ffn",
            OpKind.MATMUL,
            [expert_hidden],
            [expert_hidden],
            attrs={
                "moe": True,
                "expert_count": self.config.n_activated_experts,
                "uses_fp8": bool(self.source_features and self.source_features.uses_fp8),
                "source_class": self.architecture.moe_ffn_class,
            },
            flops=2
            * batch_size
            * seq_len
            * self.config.moe_inter_dim
            * self.config.dim
            * self.config.n_activated_experts,
            bytes_moved=batch_size * seq_len * self.config.moe_inter_dim * hidden.bytes_per_element,
        )
        shared_w1 = self._add_node(
            nodes,
            f"{prefix}_shared_ffn_w1",
            OpKind.MATMUL,
            [hidden],
            [shared_hidden],
            attrs={"moe": True, "shared_experts": True, "parallelism": "column", "source_class": self.architecture.moe_ffn_class},
            flops=2 * batch_size * seq_len * self.config.dim * (self.config.n_shared_experts * self.config.moe_inter_dim),
            bytes_moved=hidden.size_bytes,
        )
        shared_w3 = self._add_node(
            nodes,
            f"{prefix}_shared_ffn_w3",
            OpKind.MATMUL,
            [hidden],
            [shared_hidden],
            attrs={"moe": True, "shared_experts": True, "parallelism": "column", "source_class": self.architecture.moe_ffn_class},
            flops=2 * batch_size * seq_len * self.config.dim * (self.config.n_shared_experts * self.config.moe_inter_dim),
            bytes_moved=hidden.size_bytes,
        )
        shared_gate = self._add_node(
            nodes,
            f"{prefix}_shared_ffn_gate",
            OpKind.ELEMENTWISE,
            [shared_hidden, shared_hidden],
            [shared_hidden],
            attrs={"moe": True, "shared_experts": True, "source_class": self.architecture.moe_ffn_class},
            bytes_moved=4
            * batch_size
            * seq_len
            * (self.config.n_shared_experts * self.config.moe_inter_dim)
            * hidden.bytes_per_element,
        )
        shared_down = self._add_node(
            nodes,
            f"{prefix}_shared_ffn_down",
            OpKind.MATMUL,
            [shared_hidden],
            [hidden],
            attrs={"moe": True, "shared_experts": True, "parallelism": "row", "source_class": self.architecture.moe_ffn_class},
            flops=2
            * batch_size
            * seq_len
            * (self.config.n_shared_experts * self.config.moe_inter_dim)
            * self.config.dim,
            bytes_moved=hidden.size_bytes,
        )
        combine = self._add_node(
            nodes,
            f"{prefix}_combine",
            OpKind.SCATTER,
            [expert_hidden, hidden],
            [hidden],
            attrs={"moe": True, "dispatch": "expert_scatter", "source_class": self.architecture.moe_ffn_class},
            bytes_moved=hidden.size_bytes,
        )

        edges.extend(
            [
                (entry, router),
                (entry, shared_w1),
                (entry, shared_w3),
                (router, dispatch),
                (dispatch, expert_ffn),
                (expert_ffn, combine),
                (shared_w1, shared_gate),
                (shared_w3, shared_gate),
                (shared_gate, shared_down),
                (shared_down, combine),
            ]
        )
        return nodes, edges, combine

    def _is_dense_layer(self, layer_idx: int) -> bool:
        if self.architecture.moe_ffn_class and self.architecture.dense_ffn_class:
            return layer_idx < self.config.n_dense_layers
        if self.architecture.moe_ffn_class and not self.architecture.dense_ffn_class:
            return False
        return layer_idx < self.config.n_dense_layers

    def _block_norm_name(self, fragment: str, fallback: str) -> str:
        for attr in self.architecture.block_norm_attrs:
            if fragment in attr:
                return attr
        return fallback

    def _hidden_tensor(self, batch_size: int, seq_len: int) -> TensorDesc:
        return TensorDesc((batch_size, seq_len, self.config.dim), self.config.dtype)

    def _make_output_node(self, batch_size: int, seq_len: int) -> Node:
        hidden = self._hidden_tensor(batch_size, seq_len)
        logits = TensorDesc((batch_size, seq_len, self.config.vocab_size), self.config.dtype)
        return Node(
            name="lm_head",
            op_kind=OpKind.OUTPUT,
            inputs=[hidden],
            outputs=[logits],
            attrs={"source_attr": self.architecture.head_attr},
            flops=2 * batch_size * seq_len * self.config.dim * self.config.vocab_size,
            bytes_moved=hidden.size_bytes + logits.size_bytes,
        )

    def _add_node(
        self,
        nodes: list[Node],
        name: str,
        op_kind: OpKind,
        inputs: list[TensorDesc],
        outputs: list[TensorDesc],
        attrs: dict[str, object] | None = None,
        flops: float = 0.0,
        bytes_moved: float = 0.0,
    ) -> Node:
        node = Node(
            name=name,
            op_kind=op_kind,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs or {},
            flops=flops,
            bytes_moved=bytes_moved,
        )
        nodes.append(node)
        return node

    def _chain(self, edges: list[tuple[Node, Node]], nodes: list[Node]) -> None:
        for src, dst in zip(nodes, nodes[1:]):
            edges.append((src, dst))

    def _find_node(self, nodes: list[Node], name: str) -> Node:
        for node in nodes:
            if node.name == name:
                return node
        raise KeyError(name)
