"""Result formatting helpers."""

from __future__ import annotations

from ..backends.base.types import SimulationResult
from ..ir.graph import Graph


def result_to_dict(graph: Graph, result: SimulationResult) -> dict[str, object]:
    return {
        "graph": {
            "name": graph.name,
            "nodes": graph.node_count(),
            "edges": graph.edge_count(),
            "total_flops": graph.total_flops(),
            "total_bytes": graph.total_bytes(),
            "metadata": graph.metadata,
        },
        "backend": result.backend_name,
        "device": result.device_name,
        "total_latency_us": result.total_latency_us,
        "kernels": [
            {
                "name": item.task_name,
                "total_time_us": item.total_time_us,
                "bottleneck": item.bottleneck,
                "resource": item.resource,
                "start_time_us": item.start_time_us,
                "end_time_us": item.end_time_us,
            }
            for item in result.kernel_estimates
        ],
        "critical_path": result.critical_path,
        "memory": {
            "peak_live_bytes": result.memory_summary.peak_live_bytes if result.memory_summary else 0,
            "final_live_bytes": result.memory_summary.final_live_bytes if result.memory_summary else 0,
            "events": [
                {
                    "node_name": event.node_name,
                    "live_bytes": event.live_bytes,
                    "peak_bytes": event.peak_bytes,
                }
                for event in (result.memory_summary.events if result.memory_summary else [])
            ],
        },
    }


def format_summary(graph: Graph, result: SimulationResult) -> str:
    source_features = graph.metadata.get("source_features", {})
    lines = [
        f"Graph: {graph.name}",
        f"Backend: {result.backend_name} ({result.device_name})",
        f"Nodes: {graph.node_count()}  Edges: {graph.edge_count()}",
        f"Total FLOPs: {graph.total_flops():.3e}",
        f"Total Bytes: {graph.total_bytes():.3e}",
        f"Estimated Latency: {result.total_latency_us:.3f} us",
        f"MoE Nodes: {graph.metadata.get('moe_node_count', 0)}",
        f"Branch Edges: {graph.metadata.get('branch_edge_count', 0)}",
        f"Critical Path Kernels: {len(result.critical_path)}",
        f"Peak Live Memory: {_format_bytes(result.memory_summary.peak_live_bytes) if result.memory_summary else '0 B'}",
        "Top kernels:",
    ]
    if source_features:
        lines.insert(
            2,
            "Source Features: "
            f"fp8={source_features.get('uses_fp8', False)} "
            f"moe={source_features.get('uses_moe', False)} "
            f"rotary={source_features.get('uses_rotary', False)} "
            f"distributed={source_features.get('uses_distributed', False)}",
        )
    top = sorted(result.kernel_estimates, key=lambda item: item.total_time_us, reverse=True)[:5]
    for item in top:
        lines.append(
            f"  - {item.task_name}: {item.total_time_us:.3f} us ({item.bottleneck})"
        )
    if result.critical_path:
        lines.append("Critical path:")
        lines.append("  " + " -> ".join(result.critical_path[:8]) + (" -> ..." if len(result.critical_path) > 8 else ""))
    return "\n".join(lines)


def _format_bytes(value: int) -> str:
    if value >= 1024 * 1024 * 1024:
        return f"{value / (1024 * 1024 * 1024):.2f} GiB"
    if value >= 1024 * 1024:
        return f"{value / (1024 * 1024):.2f} MiB"
    if value >= 1024:
        return f"{value / 1024:.2f} KiB"
    return f"{value} B"
