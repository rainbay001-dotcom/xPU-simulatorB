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
    n_kv_heads: int | None
    dtype: DType

    @classmethod
    def from_json(cls, path: str | Path) -> "DeepSeekConfig":
        raw = json.loads(Path(path).read_text())
        dim = raw.get("dim", raw.get("hidden_size", raw.get("d_model")))
        inter_dim = raw.get("inter_dim", raw.get("intermediate_size", raw.get("ffn_dim", dim * 4 if dim else None)))
        n_layers = raw.get("n_layers", raw.get("num_hidden_layers", raw.get("num_layers")))
        n_heads = raw.get("n_heads", raw.get("num_attention_heads"))
        dtype = cls._normalize_dtype(raw.get("dtype", raw.get("torch_dtype", "bf16")))
        return cls(
            vocab_size=raw["vocab_size"],
            dim=dim,
            inter_dim=inter_dim,
            moe_inter_dim=raw.get("moe_inter_dim", raw.get("expert_intermediate_size", 0)),
            n_layers=n_layers,
            n_dense_layers=raw.get("n_dense_layers", n_layers),
            n_heads=n_heads,
            n_routed_experts=raw.get("n_routed_experts", raw.get("num_local_experts", 0)),
            n_shared_experts=raw.get("n_shared_experts", 0),
            n_activated_experts=raw.get("n_activated_experts", raw.get("num_experts_per_tok", 0)),
            qk_rope_head_dim=raw.get("qk_rope_head_dim", raw.get("head_dim", dim // n_heads if dim and n_heads else 0)),
            v_head_dim=raw.get("v_head_dim", raw.get("head_dim", dim // n_heads if dim and n_heads else 0)),
            n_kv_heads=raw.get("n_kv_heads", raw.get("num_key_value_heads")),
            dtype=DType(dtype),
        )

    @staticmethod
    def _normalize_dtype(raw_dtype: str) -> str:
        cleaned = raw_dtype.replace("torch.", "").lower()
        aliases = {
            "float32": "fp32",
            "float16": "fp16",
            "half": "fp16",
            "bfloat16": "bf16",
            "float8_e4m3fn": "fp8",
        }
        return aliases.get(cleaned, cleaned)


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
            block_attention_attr=None,
            block_cross_attention_attr=None,
            block_ffn_attr=None,
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

        attention_nodes, attention_edges, attention_entry, attention_exit = self._build_attention_subgraph(
            prefix,
            batch_size,
            seq_len,
            hidden,
        )
        nodes.extend(attention_nodes)
        edges.append((attn_norm, attention_entry))
        edges.extend(attention_edges)
        current_exit = attention_exit

        if self.architecture.block_cross_attention_attr:
            cross_nodes, cross_edges, cross_entry, cross_exit = self._build_cross_attention_subgraph(
                prefix,
                batch_size,
                seq_len,
                hidden,
            )
            nodes.extend(cross_nodes)
            edges.append((current_exit, cross_entry))
            edges.extend(cross_edges)
            current_exit = cross_exit

        ffn_norm = self._add_node(nodes, f"{prefix}_{ffn_norm_name}", OpKind.NORM, [hidden], [hidden], bytes_moved=hidden.size_bytes * 2)
        edges.append((current_exit, ffn_norm))

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

    def _build_attention_subgraph(
        self,
        prefix: str,
        batch_size: int,
        seq_len: int,
        hidden: TensorDesc,
    ) -> tuple[list[Node], list[tuple[Node, Node]], Node, Node]:
        attn_proj = TensorDesc((batch_size, seq_len, self.config.n_heads * self.config.v_head_dim), self.config.dtype)
        score_tensor = TensorDesc((batch_size, self.config.n_heads, seq_len, seq_len), self.config.dtype)
        traits = self.architecture.attention_traits
        nodes: list[Node] = []
        edges: list[tuple[Node, Node]] = []
        projection_attrs = self._attention_projection_attrs()
        projection_exit: Node

        if traits.get("has_qkv_fused", False) or projection_attrs == ["qkv_proj"]:
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
                    "projection_style": "fused_qkv",
                },
                flops=6 * batch_size * seq_len * self.config.dim * self.config.dim,
                bytes_moved=hidden.size_bytes + attn_proj.size_bytes,
            )
            attention_entry = qkv_proj
            projection_exit = qkv_proj
        else:
            projection_nodes = [
                self._add_node(
                    nodes,
                    f"{prefix}_{projection_attr}",
                    OpKind.MATMUL,
                    [hidden],
                    [attn_proj],
                    attrs={
                        "uses_fp8": bool(self.source_features and self.source_features.uses_fp8),
                        "source_class": self.architecture.attention_class,
                        "projection_count": traits.get("projection_count", 0),
                        "projection_style": "split_qkv",
                        "projection_attr": projection_attr,
                    },
                    flops=2 * batch_size * seq_len * self.config.dim * self.config.dim,
                    bytes_moved=hidden.size_bytes + attn_proj.size_bytes,
                )
                for projection_attr in projection_attrs
            ]
            merge_inputs = [node.outputs[0] for node in projection_nodes]
            attention_entry = projection_nodes[0]
            projection_exit = self._add_node(
                nodes,
                f"{prefix}_attn_proj_merge",
                OpKind.CONCAT,
                merge_inputs,
                [attn_proj],
                attrs={
                    "source_class": self.architecture.attention_class,
                    "projection_style": "split_qkv",
                    "projection_attrs": projection_attrs,
                },
                bytes_moved=sum(tensor.size_bytes for tensor in merge_inputs) + attn_proj.size_bytes,
            )
            for projection_node in projection_nodes:
                edges.append((projection_node, projection_exit))

        prev = projection_exit

        if traits.get("has_q_norm", False):
            q_norm = self._add_node(nodes, f"{prefix}_q_norm", OpKind.NORM, [attn_proj], [attn_proj], bytes_moved=attn_proj.size_bytes * 2)
            edges.append((prev, q_norm))
            prev = q_norm
        if traits.get("has_kv_norm", False):
            kv_norm = self._add_node(nodes, f"{prefix}_kv_norm", OpKind.NORM, [attn_proj], [attn_proj], bytes_moved=attn_proj.size_bytes * 2)
            edges.append((prev, kv_norm))
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
            edges.append((prev, rope))
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
        edges.append((prev, attn_scores))

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
            edges.append((attn_scores, attn_topk))
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
        edges.append(((attn_topk or attn_scores), softmax))
        attn_out = self._add_node(
            nodes,
            f"{prefix}_attn_out",
            OpKind.BATCHED_MATMUL,
            [attn_proj],
            [hidden],
            flops=4 * batch_size * self.config.n_heads * seq_len * seq_len * self.config.v_head_dim,
            bytes_moved=attn_proj.size_bytes + hidden.size_bytes,
        )
        edges.append((softmax, attn_out))
        attention_exit = attn_out
        if traits.get("has_output_proj", False):
            output_proj_name = self._attention_output_proj_name()
            output_proj = self._add_node(
                nodes,
                f"{prefix}_{output_proj_name}",
                OpKind.MATMUL,
                [hidden],
                [hidden],
                attrs={
                    "uses_fp8": bool(self.source_features and self.source_features.uses_fp8),
                    "source_class": self.architecture.attention_class,
                    "projection_style": "output",
                },
                flops=2 * batch_size * seq_len * self.config.dim * self.config.dim,
                bytes_moved=hidden.size_bytes * 2,
            )
            edges.append((attn_out, output_proj))
            attention_exit = output_proj
        return nodes, edges, attention_entry, attention_exit

    def _build_cross_attention_subgraph(
        self,
        prefix: str,
        batch_size: int,
        seq_len: int,
        hidden: TensorDesc,
    ) -> tuple[list[Node], list[tuple[Node, Node]], Node, Node]:
        score_tensor = TensorDesc((batch_size, self.config.n_heads, seq_len, seq_len), self.config.dtype)
        nodes: list[Node] = []
        edges: list[tuple[Node, Node]] = []

        cross_scores = self._add_node(
            nodes,
            f"{prefix}_cross_attn_scores",
            OpKind.BATCHED_MATMUL,
            [hidden],
            [score_tensor],
            attrs={"cross_attention": True, "source_attr": self.architecture.block_cross_attention_attr},
            flops=4 * batch_size * self.config.n_heads * seq_len * seq_len * self._kv_head_dim(),
            bytes_moved=hidden.size_bytes * 2,
        )
        cross_softmax = self._add_node(
            nodes,
            f"{prefix}_cross_attn_softmax",
            OpKind.SOFTMAX,
            [score_tensor],
            [score_tensor],
            attrs={"cross_attention": True, "source_attr": self.architecture.block_cross_attention_attr},
            flops=5 * batch_size * self.config.n_heads * seq_len * seq_len,
            bytes_moved=4 * batch_size * self.config.n_heads * seq_len * seq_len * hidden.bytes_per_element,
        )
        cross_out = self._add_node(
            nodes,
            f"{prefix}_cross_attn_out",
            OpKind.BATCHED_MATMUL,
            [hidden],
            [hidden],
            attrs={"cross_attention": True, "source_attr": self.architecture.block_cross_attention_attr},
            flops=4 * batch_size * self.config.n_heads * seq_len * seq_len * self._kv_head_dim(),
            bytes_moved=hidden.size_bytes * 2,
        )
        edges.extend([(cross_scores, cross_softmax), (cross_softmax, cross_out)])
        return nodes, edges, cross_scores, cross_out

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
        up_name, gate_name, down_name = self._dense_ffn_node_names()

        up = self._add_node(
            nodes,
            f"{prefix}_{up_name}",
            OpKind.MATMUL,
            [hidden],
            [inter],
            attrs={"parallelism": "column", "source_class": self.architecture.dense_ffn_class},
            flops=2 * batch_size * seq_len * self.config.dim * self.config.inter_dim,
            bytes_moved=hidden.size_bytes,
        )
        edges.append((entry, up))

        if traits.get("has_gate_branch", True):
            gate_up = self._add_node(
                nodes,
                f"{prefix}_{gate_name}",
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
            edges.extend([(entry, gate_up), (up, gate), (gate_up, gate)])
        else:
            down_input = up

        down = self._add_node(
            nodes,
            f"{prefix}_{down_name}",
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
        shared_up_name, shared_gate_name, shared_down_name = self._shared_moe_node_names()
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
            f"{prefix}_{shared_up_name}",
            OpKind.MATMUL,
            [hidden],
            [shared_hidden],
            attrs={"moe": True, "shared_experts": True, "parallelism": "column", "source_class": self.architecture.moe_ffn_class},
            flops=2 * batch_size * seq_len * self.config.dim * (self.config.n_shared_experts * self.config.moe_inter_dim),
            bytes_moved=hidden.size_bytes,
        )
        shared_w3 = self._add_node(
            nodes,
            f"{prefix}_{shared_gate_name}",
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
            f"{prefix}_{shared_down_name}",
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

    def _attention_projection_attrs(self) -> list[str]:
        traits = self.architecture.attention_traits
        if traits.get("has_qkv_fused", False):
            return ["qkv_proj"]

        ordered_pairs = [
            ("has_q_proj", "q_proj"),
            ("has_k_proj", "k_proj"),
            ("has_v_proj", "v_proj"),
        ]
        extracted = [name for flag, name in ordered_pairs if traits.get(flag, False)]
        return extracted or ["qkv_proj"]

    def _attention_output_proj_name(self) -> str:
        projection_attrs = self.architecture.attention_traits.get("projection_attrs", [])
        for name in ("o_proj", "out_proj", "wo", "proj_out"):
            if name in projection_attrs:
                return name
        return "o_proj"

    def _kv_head_dim(self) -> int:
        kv_heads = self.config.n_kv_heads or self.config.n_heads
        if kv_heads <= 0:
            return self.config.v_head_dim
        return self.config.dim // kv_heads

    def _dense_ffn_node_names(self) -> tuple[str, str, str]:
        gate_attrs = set(self.architecture.dense_ffn_traits.get("gate_attrs", []))
        if {"gate_proj", "up_proj", "down_proj"} <= gate_attrs:
            return "up_proj", "gate_proj", "down_proj"
        if {"gate", "up", "down"} <= gate_attrs:
            return "up", "gate", "down"
        return "ffn_w1", "ffn_w3", "ffn_down"

    def _shared_moe_node_names(self) -> tuple[str, str, str]:
        gate_attrs = set(self.architecture.moe_ffn_traits.get("gate_attrs", []))
        if {"gate_proj", "up_proj", "down_proj"} <= gate_attrs:
            return "shared_up_proj", "shared_gate_proj", "shared_down_proj"
        if {"gate", "up", "down"} <= gate_attrs:
            return "shared_up", "shared_gate", "shared_down"
        return "shared_ffn_w1", "shared_ffn_w3", "shared_ffn_down"

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
