"""Graph comparison helpers across frontends."""

from __future__ import annotations

from ..ir.graph import Graph


def diff_graphs(reference: Graph, candidate: Graph) -> dict[str, object]:
    ref_hist = _op_histogram(reference)
    cand_hist = _op_histogram(candidate)
    all_ops = sorted(set(ref_hist) | set(cand_hist))
    op_delta = {
        op_name: {
            "reference": ref_hist.get(op_name, 0),
            "candidate": cand_hist.get(op_name, 0),
            "delta": cand_hist.get(op_name, 0) - ref_hist.get(op_name, 0),
        }
        for op_name in all_ops
    }
    return {
        "reference": {
            "name": reference.name,
            "nodes": reference.node_count(),
            "edges": reference.edge_count(),
        },
        "candidate": {
            "name": candidate.name,
            "nodes": candidate.node_count(),
            "edges": candidate.edge_count(),
        },
        "node_delta": candidate.node_count() - reference.node_count(),
        "edge_delta": candidate.edge_count() - reference.edge_count(),
        "op_histogram_delta": op_delta,
    }


def format_graph_diff(reference: Graph, candidate: Graph, label_a: str = "reference", label_b: str = "candidate") -> str:
    payload = diff_graphs(reference, candidate)
    lines = [
        f"Graph diff: {label_a} vs {label_b}",
        f"{label_a}: nodes={payload['reference']['nodes']} edges={payload['reference']['edges']}",
        f"{label_b}: nodes={payload['candidate']['nodes']} edges={payload['candidate']['edges']}",
        f"Delta: nodes={payload['node_delta']} edges={payload['edge_delta']}",
        "Op histogram delta:",
    ]
    for op_name, values in payload["op_histogram_delta"].items():
        lines.append(
            f"  - {op_name}: {values['reference']} -> {values['candidate']} (delta {values['delta']})"
        )
    return "\n".join(lines)


def _op_histogram(graph: Graph) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for node in graph.nodes:
        histogram[node.op_kind.value] = histogram.get(node.op_kind.value, 0) + 1
    return histogram
