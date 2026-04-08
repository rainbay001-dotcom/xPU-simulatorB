"""Simple graph pass pipeline for early simulator stages."""

from __future__ import annotations

from ..ir.graph import Graph, Node
from ..ir.types import OpKind


def apply_default_passes(graph: Graph) -> Graph:
    _annotate_kernel_families(graph)
    _annotate_graph_summary(graph)
    return graph


def _annotate_kernel_families(graph: Graph) -> None:
    family_map = {
        OpKind.EMBEDDING: "lookup",
        OpKind.MATMUL: "gemm",
        OpKind.BATCHED_MATMUL: "attention_gemm",
        OpKind.NORM: "normalization",
        OpKind.SOFTMAX: "reduction",
        OpKind.ELEMENTWISE: "pointwise",
        OpKind.ROPE: "position_encoding",
        OpKind.TOPK: "selection",
        OpKind.GATHER: "data_movement",
        OpKind.SCATTER: "data_movement",
        OpKind.ALL_REDUCE: "communication",
        OpKind.CONCAT: "layout",
        OpKind.RESHAPE: "layout",
        OpKind.TRANSPOSE: "layout",
        OpKind.OUTPUT: "projection",
    }
    for node in graph.nodes:
        node.attrs.setdefault("kernel_family", family_map.get(node.op_kind, "generic"))
        node.attrs.setdefault("memory_intensive", node.bytes_moved > node.flops / 16 if node.flops else True)


def _annotate_graph_summary(graph: Graph) -> None:
    op_histogram: dict[str, int] = {}
    moe_nodes = 0
    branch_edges = 0
    for node in graph.nodes:
        op_histogram[node.op_kind.value] = op_histogram.get(node.op_kind.value, 0) + 1
        if node.attrs.get("moe", False):
            moe_nodes += 1
    for targets in graph.edges.values():
        if len(targets) > 1:
            branch_edges += len(targets) - 1
    graph.metadata["op_histogram"] = op_histogram
    graph.metadata["moe_node_count"] = moe_nodes
    graph.metadata["total_layers"] = sum(1 for node in graph.nodes if node.name.endswith("_attn_norm"))
    graph.metadata["branch_edge_count"] = branch_edges
