"""Base backend abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ...ir.graph import Graph, Node
from .types import HardwareConfig, KernelEstimate, KernelTask


class Backend(ABC):
    def __init__(self, hardware: HardwareConfig):
        self.hardware = hardware

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def lower_graph(self, graph: Graph) -> list[KernelTask]:
        return [self.lower_node(node) for node in graph.nodes]

    def lower_node(self, node: Node) -> KernelTask:
        return KernelTask(
            name=node.name,
            op_kind=node.op_kind,
            flops=node.flops,
            bytes_moved=node.bytes_moved,
            attrs=node.attrs,
            resource=self.resource_for_node(node),
        )

    def resource_for_node(self, node: Node) -> str:
        return self.resource_for_op(node.op_kind, node.attrs)

    def resource_for_op(self, op_kind, attrs: dict[str, object]) -> str:
        if op_kind.name == "ALL_REDUCE":
            return "communication"
        if attrs.get("kernel_family") in {"data_movement", "selection", "layout"}:
            return "memory"
        return "compute"

    @abstractmethod
    def estimate_kernel(self, task: KernelTask) -> KernelEstimate:
        raise NotImplementedError
