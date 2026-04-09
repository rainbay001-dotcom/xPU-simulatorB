"""Graph-level kernel fusion passes."""

from __future__ import annotations

from ..ir.graph import Graph, Node
from ..ir.types import OpKind


def fuse_supported_patterns(graph: Graph) -> Graph:
    """Return a new graph with supported fusion patterns collapsed."""

    name_to_node = {node.name: node for node in graph.nodes}
    replacement: dict[str, str] = {}
    emitted: dict[str, Node] = {}
    fused_patterns: list[dict[str, object]] = []

    for node in graph.topological_order():
        if node.name in replacement:
            continue

        match = _match_attention_fusion(graph, name_to_node, node)
        if match is None:
            match = _match_router_dispatch_fusion(graph, name_to_node, node)

        if match is None:
            replacement[node.name] = node.name
            emitted[node.name] = node
            continue

        fused_node, member_names, pattern = match
        for member_name in member_names:
            replacement[member_name] = fused_node.name
        emitted[fused_node.name] = fused_node
        fused_patterns.append(
            {
                "pattern": pattern,
                "fused_node": fused_node.name,
                "members": member_names,
            }
        )

    fused_graph = Graph(name=graph.name)
    fused_graph.metadata = dict(graph.metadata)
    for node in graph.topological_order():
        new_name = replacement[node.name]
        if new_name not in fused_graph.edges:
            fused_graph.add_node(emitted[new_name])

    seen_edges: set[tuple[str, str]] = set()
    for src_name, targets in graph.edges.items():
        mapped_src = replacement[src_name]
        for dst_name in targets:
            mapped_dst = replacement[dst_name]
            if mapped_src == mapped_dst:
                continue
            edge = (mapped_src, mapped_dst)
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            fused_graph.edges.setdefault(mapped_src, []).append(mapped_dst)
            fused_graph.edges.setdefault(mapped_dst, [])

    fused_graph.metadata["fused_node_count"] = len(fused_patterns)
    fused_graph.metadata["fusion_patterns"] = fused_patterns
    fused_graph.metadata["fusion_enabled"] = bool(fused_patterns)
    return fused_graph


def _match_attention_fusion(graph: Graph, name_to_node: dict[str, Node], node: Node):
    if not (
        node.name.endswith("_attn_scores")
        or node.name.endswith("_cross_attn_scores")
    ):
        return None
    maybe_next_name = _only_successor(graph, node.name)
    if maybe_next_name is None:
        return None
    maybe_next = name_to_node[maybe_next_name]
    members = [node]
    softmax: Node

    if maybe_next.op_kind == OpKind.TOPK:
        if _only_predecessor(graph, maybe_next.name) != node.name:
            return None
        softmax_name = _only_successor(graph, maybe_next.name)
        if softmax_name is None:
            return None
        softmax = name_to_node[softmax_name]
        if softmax.op_kind != OpKind.SOFTMAX or _only_predecessor(graph, softmax.name) != maybe_next.name:
            return None
        members.append(maybe_next)
    else:
        softmax = maybe_next
        if softmax.op_kind != OpKind.SOFTMAX or _only_predecessor(graph, softmax.name) != node.name:
            return None

    out_name = _only_successor(graph, softmax.name)
    if out_name is None:
        return None
    attn_out = name_to_node[out_name]
    if attn_out.op_kind != OpKind.BATCHED_MATMUL or _only_predecessor(graph, attn_out.name) != softmax.name:
        return None

    pattern = "attention_softmax_out"
    prefix = node.name.removesuffix("_attn_scores").removesuffix("_cross_attn_scores")
    fused_name = f"{prefix}_fused_attention"
    members.extend([softmax, attn_out])
    fused_node = Node(
        name=fused_name,
        op_kind=OpKind.BATCHED_MATMUL,
        inputs=list(node.inputs),
        outputs=list(attn_out.outputs),
        attrs=_fused_attrs(pattern, members, kernel_family="fused_attention"),
        flops=sum(item.flops for item in members),
        bytes_moved=_fused_bytes(members),
    )
    return fused_node, [item.name for item in members], pattern


def _match_router_dispatch_fusion(graph: Graph, name_to_node: dict[str, Node], node: Node):
    if not node.name.endswith("_router") or node.op_kind != OpKind.TOPK:
        return None
    dispatch_name = _only_successor(graph, node.name)
    if dispatch_name is None:
        return None
    dispatch = name_to_node[dispatch_name]
    if dispatch.op_kind != OpKind.GATHER or _only_predecessor(graph, dispatch.name) != node.name:
        return None

    pattern = "router_dispatch"
    prefix = node.name.removesuffix("_router")
    members = [node, dispatch]
    fused_node = Node(
        name=f"{prefix}_fused_router_dispatch",
        op_kind=OpKind.GATHER,
        inputs=list(node.inputs),
        outputs=list(dispatch.outputs),
        attrs=_fused_attrs(pattern, members, kernel_family="fused_router_dispatch"),
        flops=sum(item.flops for item in members),
        bytes_moved=_fused_bytes(members),
    )
    return fused_node, [item.name for item in members], pattern


def _fused_attrs(pattern: str, members: list[Node], kernel_family: str) -> dict[str, object]:
    attrs = dict(members[0].attrs)
    attrs["fused"] = True
    attrs["fusion_pattern"] = pattern
    attrs["fused_ops"] = [member.op_kind.value for member in members]
    attrs["fused_members"] = [member.name for member in members]
    attrs["fusion_count"] = len(members)
    attrs["kernel_family"] = kernel_family
    attrs["memory_intensive"] = any(member.attrs.get("memory_intensive", False) for member in members)
    return attrs


def _fused_bytes(members: list[Node]) -> float:
    raw_total = sum(member.bytes_moved for member in members)
    internal_tensor_bytes = sum(sum(tensor.size_bytes for tensor in member.outputs) for member in members[:-1])
    return max(raw_total - (2 * internal_tensor_bytes), 0.0)


def _only_successor(graph: Graph, node_name: str) -> str | None:
    successors = graph.successors(node_name)
    if len(successors) != 1:
        return None
    return successors[0]


def _only_predecessor(graph: Graph, node_name: str) -> str | None:
    predecessors = graph.predecessors(node_name)
    if len(predecessors) != 1:
        return None
    return predecessors[0]
