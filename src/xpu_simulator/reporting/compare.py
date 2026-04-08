"""Comparison formatting helpers."""

from __future__ import annotations

from ..backends.base.types import SimulationResult
from ..ir.graph import Graph


def compare_results(graph: Graph, results: dict[str, SimulationResult]) -> dict[str, object]:
    latencies = {name: result.total_latency_us for name, result in results.items()}
    peak_memory = {
        name: (result.memory_summary.peak_live_bytes if result.memory_summary else 0)
        for name, result in results.items()
    }
    fastest_backend = min(latencies, key=latencies.get)
    slowest_backend = max(latencies, key=latencies.get)
    delta_us = latencies[slowest_backend] - latencies[fastest_backend]
    ratio = latencies[slowest_backend] / latencies[fastest_backend] if latencies[fastest_backend] else 0.0
    return {
        "graph_name": graph.name,
        "latencies_us": latencies,
        "peak_memory_bytes": peak_memory,
        "fastest_backend": fastest_backend,
        "slowest_backend": slowest_backend,
        "latency_delta_us": delta_us,
        "slowdown_ratio": ratio,
    }


def format_comparison(graph: Graph, results: dict[str, SimulationResult]) -> str:
    payload = compare_results(graph, results)
    lines = [
        f"Graph: {payload['graph_name']}",
        "Backend comparison:",
    ]
    for backend_name, latency in payload["latencies_us"].items():
        peak = payload["peak_memory_bytes"][backend_name]
        lines.append(
            f"  - {backend_name}: {latency:.3f} us, peak memory {peak / (1024 * 1024):.2f} MiB"
        )
    lines.append(
        f"Fastest: {payload['fastest_backend']}  Slowest: {payload['slowest_backend']}  "
        f"Delta: {payload['latency_delta_us']:.3f} us  Ratio: {payload['slowdown_ratio']:.3f}x"
    )
    return "\n".join(lines)
