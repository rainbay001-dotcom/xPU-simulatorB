"""Core graph types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .types import DType, OpKind


@dataclass
class TensorDesc:
    shape: tuple[int, ...]
    dtype: DType
    layout: str = "default"

    @property
    def numel(self) -> int:
        total = 1
        for dim in self.shape:
            total *= dim
        return total

    @property
    def bytes_per_element(self) -> int:
        return {
            DType.FP32: 4,
            DType.FP16: 2,
            DType.BF16: 2,
            DType.FP8: 1,
            DType.INT8: 1,
        }[self.dtype]

    @property
    def size_bytes(self) -> int:
        return self.numel * self.bytes_per_element


@dataclass
class Node:
    name: str
    op_kind: OpKind
    inputs: list[TensorDesc]
    outputs: list[TensorDesc]
    attrs: dict[str, Any] = field(default_factory=dict)
    flops: float = 0.0
    bytes_moved: float = 0.0


class Graph:
    """Simple DAG representation with stable insertion order."""

    def __init__(self, name: str):
        self.name = name
        self.nodes: list[Node] = []
        self.edges: dict[str, list[str]] = {}
        self.metadata: dict[str, Any] = {}

    def add_node(self, node: Node) -> Node:
        self.nodes.append(node)
        self.edges.setdefault(node.name, [])
        return node

    def add_edge(self, src: Node, dst: Node) -> None:
        self.edges.setdefault(src.name, []).append(dst.name)
        self.edges.setdefault(dst.name, [])

    def node_count(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        return sum(len(targets) for targets in self.edges.values())

    def total_flops(self) -> float:
        return sum(node.flops for node in self.nodes)

    def total_bytes(self) -> float:
        return sum(node.bytes_moved for node in self.nodes)

    def successors(self, node_name: str) -> list[str]:
        return list(self.edges.get(node_name, []))

    def predecessors(self, node_name: str) -> list[str]:
        preds: list[str] = []
        for src, targets in self.edges.items():
            if node_name in targets:
                preds.append(src)
        return preds

    def topological_order(self) -> list[Node]:
        indegree = {node.name: 0 for node in self.nodes}
        for targets in self.edges.values():
            for dst in targets:
                indegree[dst] = indegree.get(dst, 0) + 1

        ready = [node.name for node in self.nodes if indegree[node.name] == 0]
        ordered: list[Node] = []
        name_to_node = {node.name: node for node in self.nodes}

        while ready:
            current = ready.pop(0)
            ordered.append(name_to_node[current])
            for nxt in self.successors(current):
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    ready.append(nxt)

        if len(ordered) != len(self.nodes):
            raise ValueError("graph contains a cycle")
        return ordered
