"""CLI entry point."""

from __future__ import annotations

import argparse
import json
import tempfile

from ..backends.base import load_hardware_config
from ..backends import AscendBackend, NvidiaBackend
from ..calibration import load_backend_calibration, load_benchmark_rows, summarize_benchmark_rows
from ..frontend import ModelConfig, TorchFxGraphBuilder, TransformerSourceGraphBuilder
from ..reporting import compare_results, format_comparison, format_summary, result_to_dict
from ..sim import Simulator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="xPU simulator scaffold")
    parser.add_argument("--model-config", required=True, help="Path to model config JSON")
    parser.add_argument(
        "--model-family",
        choices=["transformer_source", "transformer_fx", "deepseek"],
        default="transformer_source",
        help="Frontend family to use. `transformer_fx` requires an executable PyTorch model class.",
    )
    parser.add_argument("--model-class", help="Model class name for torch.fx frontend")
    parser.add_argument("--backend", choices=["nvidia", "ascend", "compare"], default="nvidia")
    parser.add_argument("--device-config", help="Optional backend-specific hardware config JSON")
    parser.add_argument("--calibration-config", help="Optional backend-specific calibration JSON")
    parser.add_argument("--benchmark-csv", help="Optional benchmark CSV to summarize for calibration review")
    parser.add_argument("--nvidia-device-config", help="Optional NVIDIA hardware config JSON for compare mode")
    parser.add_argument("--nvidia-calibration-config", help="Optional NVIDIA calibration JSON for compare mode")
    parser.add_argument("--ascend-device-config", help="Optional Ascend hardware config JSON for compare mode")
    parser.add_argument("--ascend-calibration-config", help="Optional Ascend calibration JSON for compare mode")
    parser.add_argument("--model-source", help="Path to model source file for AST-based architecture extraction")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--dump-memory-events", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.benchmark_csv:
        summary = summarize_benchmark_rows(load_benchmark_rows(args.benchmark_csv))
        print(json.dumps(summary, indent=2))
        return
    config = ModelConfig.from_json(args.model_config)
    graph_builder = _make_graph_builder(args, config)
    graph = graph_builder.build_graph(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        layers=args.layers,
        layer_start=args.layer_start,
    )
    simulator = Simulator()

    if args.backend == "compare":
        results = {
            "nvidia": simulator.simulate(graph, _make_backend("nvidia", args.nvidia_device_config, args.nvidia_calibration_config)),
            "ascend": simulator.simulate(graph, _make_backend("ascend", args.ascend_device_config, args.ascend_calibration_config)),
        }
        if args.json_output:
            print(json.dumps(compare_results(graph, results), indent=2))
            return
        print(format_comparison(graph, results))
        return

    backend = _make_backend(args.backend, args.device_config, args.calibration_config)
    result = simulator.simulate(graph, backend)
    if args.json_output:
        payload = result_to_dict(graph, result)
        if not args.dump_memory_events:
            payload["memory"].pop("events", None)
        print(json.dumps(payload, indent=2))
        return
    print(format_summary(graph, result))


def _make_backend(name: str, device_config: str | None, calibration_config: str | None):
    hardware = load_hardware_config(device_config) if device_config else None
    calibration = load_backend_calibration(calibration_config) if calibration_config else None
    if name == "nvidia":
        return NvidiaBackend(hardware=hardware, calibration=calibration)
    if name == "ascend":
        return AscendBackend(hardware=hardware, calibration=calibration)
    raise ValueError(name)


def _make_graph_builder(args, config: ModelConfig):
    if args.model_family == "transformer_fx":
        if not args.model_source or not args.model_class:
            raise ValueError("--model-source and --model-class are required for transformer_fx")
        return TorchFxGraphBuilder(config, source_path=args.model_source, model_class=args.model_class)
    return TransformerSourceGraphBuilder(config, source_path=args.model_source)


if __name__ == "__main__":
    main()
