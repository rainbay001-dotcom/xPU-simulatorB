"""Memory liveness and cache analysis for the coarse graph IR."""

from __future__ import annotations

from ..backends.base.types import CacheSummary, MemoryEvent, MemorySummary
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


def analyze_cache(graph: Graph) -> CacheSummary | None:
    mode = str(graph.metadata.get("mode", "prefill"))
    context_len = int(graph.metadata.get("context_len", graph.metadata.get("seq_len", 0)))
    step_tokens = int(graph.metadata.get("step_tokens", graph.metadata.get("seq_len", 0)))
    kv_cache_bytes_per_layer = int(graph.metadata.get("kv_cache_bytes_per_layer", 0))
    kv_cache_total_bytes = int(graph.metadata.get("kv_cache_total_bytes", 0))
    if kv_cache_bytes_per_layer <= 0 and kv_cache_total_bytes <= 0:
        return None
    return CacheSummary(
        mode=mode,
        context_len=context_len,
        step_tokens=step_tokens,
        kv_cache_bytes_per_layer=kv_cache_bytes_per_layer,
        kv_cache_total_bytes=kv_cache_total_bytes,
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
