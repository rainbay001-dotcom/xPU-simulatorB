"""Optional torch.fx-based frontend for executable PyTorch models."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ..graph_passes import apply_default_passes
from ..ir.graph import Graph, Node, TensorDesc
from ..ir.types import DType, OpKind
from .deepseek import DeepSeekConfig


class TorchFxGraphBuilder:
    """Build simulator graphs from an executable PyTorch module via torch.fx."""

    def __init__(self, config: DeepSeekConfig, source_path: str | Path, model_class: str):
        self.config = config
        self.source_path = Path(source_path)
        self.model_class = model_class

    def build_graph(
        self,
        batch_size: int = 1,
        seq_len: int = 128,
        layers: int | None = None,
        layer_start: int = 0,
    ) -> Graph:
        torch, nn, symbolic_trace, ShapeProp = self._load_torch_components()
        module = self._load_module()
        model_type = getattr(module, self.model_class)
        model = model_type(self._config_namespace())
        model.eval()

        traced = symbolic_trace(model)
        sample_inputs = self._sample_inputs(torch, model, batch_size, seq_len)
        ShapeProp(traced).propagate(*sample_inputs)

        graph = Graph(name=f"fx_{self.model_class}_b{batch_size}_s{seq_len}")
        graph.metadata["model_family"] = "transformer_fx"
        graph.metadata["frontend"] = "torch_fx"
        graph.metadata["model_class"] = self.model_class
        graph.metadata["source_path"] = str(self.source_path)
        graph.metadata["batch_size"] = batch_size
        graph.metadata["seq_len"] = seq_len
        graph.metadata["layer_start"] = layer_start
        graph.metadata["layer_stop"] = layer_start + (layers if layers is not None else self.config.n_layers)

        fx_nodes = list(traced.graph.nodes)
        fx_to_sim: dict[str, Node] = {}
        module_lookup = dict(traced.named_modules())

        for fx_node in fx_nodes:
            sim_node = self._convert_fx_node(fx_node, module_lookup)
            if sim_node is None:
                continue
            graph.add_node(sim_node)
            fx_to_sim[fx_node.name] = sim_node

        for fx_node in fx_nodes:
            dst = fx_to_sim.get(fx_node.name)
            if dst is None:
                continue
            for src_name in self._input_fx_nodes(fx_node):
                src = fx_to_sim.get(src_name)
                if src is not None:
                    graph.add_edge(src, dst)

        return apply_default_passes(graph)

    def _load_torch_components(self):
        import torch
        import torch.nn as nn
        from torch.fx import symbolic_trace
        from torch.fx.passes.shape_prop import ShapeProp

        return torch, nn, symbolic_trace, ShapeProp

    def _load_module(self):
        spec = importlib.util.spec_from_file_location(self.source_path.stem, self.source_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"unable to load module from {self.source_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _config_namespace(self) -> SimpleNamespace:
        return SimpleNamespace(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.dim,
            dim=self.config.dim,
            intermediate_size=self.config.inter_dim,
            inter_dim=self.config.inter_dim,
            moe_inter_dim=self.config.moe_inter_dim,
            num_hidden_layers=self.config.n_layers,
            n_layers=self.config.n_layers,
            num_attention_heads=self.config.n_heads,
            n_heads=self.config.n_heads,
            num_key_value_heads=self.config.n_kv_heads or self.config.n_heads,
            n_kv_heads=self.config.n_kv_heads or self.config.n_heads,
            dtype=self.config.dtype.value,
        )

    def _sample_inputs(self, torch, model, batch_size: int, seq_len: int) -> tuple[Any, ...]:
        signature = inspect.signature(model.forward)
        args: list[Any] = []
        hidden = self.config.dim
        vocab_size = max(self.config.vocab_size, 16)
        for parameter in signature.parameters.values():
            if parameter.name == "self":
                continue
            if parameter.default is not inspect._empty:
                continue
            name = parameter.name.lower()
            if "input" in name or "token" in name or "ids" in name:
                args.append(torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long))
            elif "mask" in name:
                args.append(torch.ones((batch_size, seq_len), dtype=torch.bool))
            elif "encoder" in name or "hidden" in name or "states" in name:
                args.append(torch.zeros((batch_size, seq_len, hidden), dtype=torch.float32))
            else:
                args.append(torch.zeros((batch_size, seq_len, hidden), dtype=torch.float32))
        return tuple(args)

    def _convert_fx_node(self, fx_node, module_lookup: dict[str, Any]) -> Node | None:
        if fx_node.op in {"placeholder", "output"}:
            return None

        op_kind = self._infer_op_kind(fx_node, module_lookup)
        inputs = self._collect_input_tensors(fx_node)
        outputs = self._collect_output_tensors(fx_node)
        attrs = {
            "fx_op": fx_node.op,
            "fx_target": str(fx_node.target),
        }
        flops = self._estimate_flops(op_kind, inputs, outputs)
        bytes_moved = sum(t.size_bytes for t in inputs) + sum(t.size_bytes for t in outputs)
        return Node(
            name=fx_node.name,
            op_kind=op_kind,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
            flops=flops,
            bytes_moved=bytes_moved,
        )

    def _infer_op_kind(self, fx_node, module_lookup: dict[str, Any]) -> OpKind:
        target = fx_node.target
        op_name = str(target)

        if fx_node.op == "call_module":
            module = module_lookup.get(target)
            module_name = module.__class__.__name__.lower() if module is not None else ""
            if "embedding" in module_name:
                return OpKind.EMBEDDING
            if "layernorm" in module_name or "rmsnorm" in module_name or "norm" in module_name:
                return OpKind.NORM
            if "linear" in module_name:
                return OpKind.MATMUL

        lowered = op_name.lower()
        if any(name in lowered for name in {"softmax"}):
            return OpKind.SOFTMAX
        if any(name in lowered for name in {"matmul", "bmm"}):
            return OpKind.BATCHED_MATMUL
        if any(name in lowered for name in {"cat", "concat"}):
            return OpKind.CONCAT
        if any(name in lowered for name in {"reshape", "view", "flatten"}):
            return OpKind.RESHAPE
        if any(name in lowered for name in {"transpose", "permute"}):
            return OpKind.TRANSPOSE
        return OpKind.ELEMENTWISE

    def _collect_input_tensors(self, fx_node) -> list[TensorDesc]:
        tensors: list[TensorDesc] = []
        for input_node in self._iter_input_nodes(fx_node):
            tensors.extend(self._tensor_descs_from_meta(input_node.meta.get("tensor_meta")))
        return tensors

    def _collect_output_tensors(self, fx_node) -> list[TensorDesc]:
        return self._tensor_descs_from_meta(fx_node.meta.get("tensor_meta"))

    def _iter_input_nodes(self, fx_node):
        for arg in fx_node.args:
            yield from self._walk_arg(arg)
        for _, value in fx_node.kwargs.items():
            yield from self._walk_arg(value)

    def _walk_arg(self, arg):
        if hasattr(arg, "op") and hasattr(arg, "name"):
            yield arg
            return
        if isinstance(arg, (tuple, list)):
            for item in arg:
                yield from self._walk_arg(item)
        elif isinstance(arg, dict):
            for item in arg.values():
                yield from self._walk_arg(item)

    def _input_fx_nodes(self, fx_node) -> list[str]:
        return [node.name for node in self._iter_input_nodes(fx_node)]

    def _tensor_descs_from_meta(self, tensor_meta) -> list[TensorDesc]:
        if tensor_meta is None:
            return []
        if isinstance(tensor_meta, (tuple, list)):
            descs: list[TensorDesc] = []
            for item in tensor_meta:
                descs.extend(self._tensor_descs_from_meta(item))
            return descs
        if not hasattr(tensor_meta, "shape") or not hasattr(tensor_meta, "dtype"):
            return []
        shape = tuple(int(dim) for dim in tensor_meta.shape)
        return [TensorDesc(shape=shape, dtype=self._to_dtype(str(tensor_meta.dtype)))]

    def _to_dtype(self, dtype: str) -> DType:
        normalized = dtype.replace("torch.", "")
        return {
            "float32": DType.FP32,
            "float": DType.FP32,
            "float16": DType.FP16,
            "half": DType.FP16,
            "bfloat16": DType.BF16,
            "int8": DType.INT8,
            "long": DType.INT8,
            "int64": DType.INT8,
        }.get(normalized, self.config.dtype)

    def _estimate_flops(self, op_kind: OpKind, inputs: list[TensorDesc], outputs: list[TensorDesc]) -> float:
        if not outputs:
            return 0.0
        output = outputs[0]
        if op_kind == OpKind.MATMUL and inputs:
            lhs = inputs[0]
            if lhs.shape and output.shape and lhs.shape[-1] > 0 and output.shape[-1] > 0:
                rows = max(output.numel // max(output.shape[-1], 1), 1)
                return 2 * rows * lhs.shape[-1] * output.shape[-1]
        if op_kind == OpKind.BATCHED_MATMUL and len(inputs) >= 2:
            lhs = inputs[0]
            rhs = inputs[1]
            if lhs.shape and rhs.shape and lhs.shape[-1] > 0:
                return 2 * output.numel * lhs.shape[-1]
        if op_kind == OpKind.SOFTMAX:
            return 5 * output.numel
        return 0.0
