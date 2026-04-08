"""Memory liveness analysis for the coarse graph IR."""

from __future__ import annotations

from ..backends.base.types import MemoryEvent, MemorySummary
from ..ir.graph import Graph


def analyze_memory(graph: Graph) -> MemorySummary:
    order = graph.topological_order()
    last_use = _last_use_positions(graph)
    live_sizes: dict[str, int] = {}
    live_bytes = 0
    peak_bytes = 0
    events: list[MemoryEvent] = []

    for index, node in enumerate(order):
        produced_bytes = sum(tensor.size_bytes for tensor in node.outputs)
        live_sizes[node.name] = produced_bytes
        live_bytes += produced_bytes

        to_free = [
            name for name, release_index in last_use.items()
            if release_index == index and name in live_sizes
        ]
        for name in to_free:
            live_bytes -= live_sizes.pop(name)

        peak_bytes = max(peak_bytes, live_bytes)
        events.append(MemoryEvent(node_name=node.name, live_bytes=live_bytes, peak_bytes=peak_bytes))

    return MemorySummary(
        peak_live_bytes=peak_bytes,
        final_live_bytes=live_bytes,
        events=events,
    )


def _last_use_positions(graph: Graph) -> dict[str, int]:
    order = graph.topological_order()
    positions = {node.name: index for index, node in enumerate(order)}
    last_use: dict[str, int] = {}
    for node in order:
        successors = graph.successors(node.name)
        if not successors:
            last_use[node.name] = positions[node.name]
            continue
        last_use[node.name] = max(positions[succ] for succ in successors)
    return last_use
