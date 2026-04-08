"""Simple simulator engine."""

from __future__ import annotations

from dataclasses import replace

from ..backends.base.backend import Backend
from ..backends.base.types import KernelEstimate, SimulationResult
from ..ir.graph import Graph
from ..memory import analyze_memory


class Simulator:
    def simulate(self, graph: Graph, backend: Backend) -> SimulationResult:
        tasks = {task.name: task for task in backend.lower_graph(graph)}
        estimates = {
            name: backend.estimate_kernel(task)
            for name, task in tasks.items()
        }
        scheduled = self._schedule(graph, estimates)
        return SimulationResult(
            backend_name=backend.name,
            device_name=backend.hardware.name,
            total_latency_us=max((item.end_time_us for item in scheduled.values()), default=0.0),
            kernel_estimates=[scheduled[node.name] for node in graph.topological_order()],
            critical_path=self._critical_path(scheduled),
            memory_summary=analyze_memory(graph),
        )

    def _schedule(self, graph: Graph, estimates: dict[str, KernelEstimate]) -> dict[str, KernelEstimate]:
        scheduled: dict[str, KernelEstimate] = {}
        resource_ready: dict[str, float] = {
            "compute": 0.0,
            "memory": 0.0,
            "communication": 0.0,
        }
        for node in graph.topological_order():
            preds = graph.predecessors(node.name)
            estimate = estimates[node.name]
            dep_ready = max((scheduled[pred].end_time_us for pred in preds), default=0.0)
            resource = estimate.resource
            start_time = max(dep_ready, resource_ready.get(resource, 0.0))
            end_time = start_time + estimate.total_time_us
            resource_ready[resource] = end_time
            scheduled[node.name] = replace(
                estimate,
                start_time_us=start_time,
                end_time_us=end_time,
                predecessors=preds,
            )
        return scheduled

    def _critical_path(self, scheduled: dict[str, KernelEstimate]) -> list[str]:
        if not scheduled:
            return []
        end_node = max(scheduled.values(), key=lambda item: item.end_time_us)
        path = [end_node.task_name]
        current = end_node
        while current.predecessors:
            current = max(
                (scheduled[pred] for pred in current.predecessors),
                key=lambda item: item.end_time_us,
            )
            path.append(current.task_name)
        path.reverse()
        return path
