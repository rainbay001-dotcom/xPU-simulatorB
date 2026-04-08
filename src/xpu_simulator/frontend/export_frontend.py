"""Optional torch.export-based frontend for executable PyTorch models."""

from __future__ import annotations

from ..graph_passes import apply_default_passes
from ..ir.graph import Graph, Node
from .fx_frontend import TorchFxGraphBuilder


class TorchExportGraphBuilder(TorchFxGraphBuilder):
    """Build simulator graphs from a torch.export-exported program."""

    def build_graph(
        self,
        batch_size: int = 1,
        seq_len: int = 128,
        layers: int | None = None,
        layer_start: int = 0,
    ) -> Graph:
        torch, _, _, _ = self._load_torch_components()
        module = self._load_module()
        model_type = getattr(module, self.model_class)
        model = model_type(self._config_namespace())
        model.eval()

        sample_inputs = self._sample_inputs(torch, model, batch_size, seq_len)
        exported = torch.export.export(model, sample_inputs)
        graph_module = exported.graph_module

        graph = Graph(name=f"export_{self.model_class}_b{batch_size}_s{seq_len}")
        graph.metadata["model_family"] = "transformer_export"
        graph.metadata["frontend"] = "torch_export"
        graph.metadata["model_class"] = self.model_class
        graph.metadata["source_path"] = str(self.source_path)
        graph.metadata["batch_size"] = batch_size
        graph.metadata["seq_len"] = seq_len
        graph.metadata["layer_start"] = layer_start
        graph.metadata["layer_stop"] = layer_start + (layers if layers is not None else self.config.n_layers)

        export_nodes = list(graph_module.graph.nodes)
        export_to_sim: dict[str, Node] = {}
        module_lookup = dict(graph_module.named_modules())

        for export_node in export_nodes:
            sim_node = self._convert_fx_node(export_node, module_lookup)
            if sim_node is None:
                continue
            sim_node.attrs["exported"] = True
            graph.add_node(sim_node)
            export_to_sim[export_node.name] = sim_node

        for export_node in export_nodes:
            dst = export_to_sim.get(export_node.name)
            if dst is None:
                continue
            for src_name in self._input_fx_nodes(export_node):
                src = export_to_sim.get(src_name)
                if src is not None:
                    graph.add_edge(src, dst)

        return apply_default_passes(graph)
