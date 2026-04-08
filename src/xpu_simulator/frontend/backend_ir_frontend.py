"""Backend-IR ingestion frontend."""

from __future__ import annotations

import json
from pathlib import Path

from ..graph_passes import apply_default_passes
from ..ir.graph import Graph, Node, TensorDesc
from ..ir.types import DType, OpKind


class BackendIrGraphBuilder:
    """Build a graph from a simple lowered backend-IR JSON description."""

    def __init__(self, ir_path: str | Path):
        self.ir_path = Path(ir_path)

    def build_graph(
        self,
        batch_size: int = 1,
        seq_len: int = 128,
        layers: int | None = None,
        layer_start: int = 0,
        mode: str = "prefill",
        context_len: int | None = None,
    ) -> Graph:
        raw = json.loads(self.ir_path.read_text())
        graph = Graph(name=raw.get("name", self.ir_path.stem))
        graph.metadata["frontend"] = "backend_ir"
        graph.metadata["model_family"] = raw.get("model_family", "backend_ir")
        graph.metadata["source_path"] = str(self.ir_path)
        graph.metadata["batch_size"] = batch_size
        graph.metadata["seq_len"] = seq_len
        graph.metadata["mode"] = mode
        graph.metadata["context_len"] = context_len if context_len is not None else seq_len
        graph.metadata["step_tokens"] = seq_len if mode == "prefill" else 1
        graph.metadata["layer_start"] = layer_start
        graph.metadata["layer_stop"] = layer_start + (layers or 0)

        node_map: dict[str, Node] = {}
        for raw_node in raw.get("nodes", []):
            node = Node(
                name=raw_node["name"],
                op_kind=OpKind(raw_node["op_kind"]),
                inputs=[self._tensor_desc(item) for item in raw_node.get("inputs", [])],
                outputs=[self._tensor_desc(item) for item in raw_node.get("outputs", [])],
                attrs=raw_node.get("attrs", {}),
                flops=float(raw_node.get("flops", 0.0)),
                bytes_moved=float(raw_node.get("bytes_moved", 0.0)),
            )
            graph.add_node(node)
            node_map[node.name] = node

        for raw_edge in raw.get("edges", []):
            src = node_map[raw_edge["src"]]
            dst = node_map[raw_edge["dst"]]
            graph.add_edge(src, dst)

        return apply_default_passes(graph)

    def _tensor_desc(self, raw: dict[str, object]) -> TensorDesc:
        return TensorDesc(
            shape=tuple(int(dim) for dim in raw.get("shape", [])),
            dtype=DType(raw.get("dtype", "bf16")),
            layout=str(raw.get("layout", "default")),
        )
