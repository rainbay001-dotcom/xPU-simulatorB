"""CLI entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from ..backends.base import load_hardware_config
from ..backends import AscendBackend, NvidiaBackend
from ..calibration import build_calibration_report, load_backend_calibration, load_benchmark_rows, summarize_benchmark_rows
from ..frontend import (
    BackendIrGraphBuilder,
    ModelConfig,
    SourceModelAnalyzer,
    TorchExportGraphBuilder,
    TorchFxGraphBuilder,
    TransformerSourceGraphBuilder,
)
from ..profiling import load_trace_events, summarize_trace_events
from ..reporting import (
    compare_results,
    diff_graphs,
    format_comparison,
    format_graph_diff,
    format_summary,
    result_to_dict,
    write_comparison_html_report,
    write_html_report,
)
from ..sim import Simulator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="xPU simulator scaffold")
    parser.add_argument("--model-config", required=True, help="Path to model config JSON")
    parser.add_argument(
        "--model-family",
        choices=["transformer_source", "transformer_fx", "transformer_export", "backend_ir", "deepseek"],
        default="transformer_source",
        help="Frontend family to use. FX/export require an executable PyTorch model class; backend_ir expects a lowered IR JSON.",
    )
    parser.add_argument("--model-class", help="Model class name for torch.fx frontend")
    parser.add_argument("--backend-ir", help="Path to backend/compiler IR JSON for backend_ir frontend")
    parser.add_argument("--backend", choices=["nvidia", "ascend", "compare"], default="nvidia")
    parser.add_argument("--device-config", help="Optional backend-specific hardware config JSON")
    parser.add_argument("--calibration-config", help="Optional backend-specific calibration JSON")
    parser.add_argument("--benchmark-csv", help="Optional benchmark CSV to summarize for calibration review")
    parser.add_argument("--trace-json", help="Optional profiler trace JSON to summarize")
    parser.add_argument("--export-architecture", help="Write extracted model architecture JSON to this path, or '-' for stdout")
    parser.add_argument("--html-report", help="Write simulation HTML report to this path. Default is an auto-generated filename in the current directory.")
    parser.add_argument("--calibration-report", action="store_true", help="Build a calibration recommendation report from --benchmark-csv")
    parser.add_argument("--compare-frontends", action="store_true", help="Compare AST/source frontend against FX/export frontend")
    parser.add_argument("--compare-target", choices=["fx", "export"], default="fx", help="Frontend to compare against the AST/source frontend")
    parser.add_argument("--nvidia-device-config", help="Optional NVIDIA hardware config JSON for compare mode")
    parser.add_argument("--nvidia-calibration-config", help="Optional NVIDIA calibration JSON for compare mode")
    parser.add_argument("--ascend-device-config", help="Optional Ascend hardware config JSON for compare mode")
    parser.add_argument("--ascend-calibration-config", help="Optional Ascend calibration JSON for compare mode")
    parser.add_argument("--model-source", help="Path to model source file for AST-based architecture extraction")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--mode", choices=["prefill", "decode"], default="prefill")
    parser.add_argument("--context-len", type=int, help="Existing KV-cache context length for decode mode")
    parser.add_argument("--enable-fusion", action="store_true", help="Enable supported graph-level kernel fusion passes")
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--dump-memory-events", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.trace_json:
        payload = summarize_trace_events(load_trace_events(args.trace_json))
        print(json.dumps(payload, indent=2))
        return
    if args.benchmark_csv:
        rows = load_benchmark_rows(args.benchmark_csv)
        summary = build_calibration_report(rows) if args.calibration_report else summarize_benchmark_rows(rows)
        print(json.dumps(summary, indent=2))
        return
    config = ModelConfig.from_json(args.model_config)
    if args.export_architecture:
        if not args.model_source:
            raise ValueError("--model-source is required for --export-architecture")
        payload = SourceModelAnalyzer().export_architecture(args.model_source, config=config)
        rendered = json.dumps(payload, indent=2)
        if args.export_architecture == "-":
            print(rendered)
        else:
            with open(args.export_architecture, "w") as handle:
                handle.write(rendered + "\n")
        return
    if args.compare_frontends:
        reference = TransformerSourceGraphBuilder(config, source_path=args.model_source).build_graph(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            layers=args.layers,
            layer_start=args.layer_start,
            mode=args.mode,
            context_len=args.context_len,
            enable_fusion=args.enable_fusion,
        )
        candidate_builder = _make_graph_builder(args, config, frontend_override=args.compare_target)
        candidate = candidate_builder.build_graph(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            layers=args.layers,
            layer_start=args.layer_start,
            mode=args.mode,
            context_len=args.context_len,
            enable_fusion=args.enable_fusion,
        )
        if args.json_output:
            print(json.dumps(diff_graphs(reference, candidate), indent=2))
            return
        print(format_graph_diff(reference, candidate, label_a="ast", label_b=args.compare_target))
        return
    graph_builder = _make_graph_builder(args, config)
    graph = graph_builder.build_graph(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        layers=args.layers,
        layer_start=args.layer_start,
        mode=args.mode,
        context_len=args.context_len,
        enable_fusion=args.enable_fusion,
    )
    simulator = Simulator()

    if args.backend == "compare":
        results = {
            "nvidia": simulator.simulate(graph, _make_backend("nvidia", args.nvidia_device_config, args.nvidia_calibration_config)),
            "ascend": simulator.simulate(graph, _make_backend("ascend", args.ascend_device_config, args.ascend_calibration_config)),
        }
        report_path = _resolve_html_report_path(args.html_report, graph.name, "compare")
        write_comparison_html_report(graph, results, report_path)
        if args.json_output:
            payload = compare_results(graph, results)
            payload["html_report_path"] = str(report_path)
            print(json.dumps(payload, indent=2))
            return
        print(format_comparison(graph, results))
        print(f"HTML report: {report_path}")
        return

    backend = _make_backend(args.backend, args.device_config, args.calibration_config)
    result = simulator.simulate(graph, backend)
    report_path = _resolve_html_report_path(args.html_report, graph.name, result.backend_name)
    write_html_report(graph, result, report_path)
    if args.json_output:
        payload = result_to_dict(graph, result)
        payload["html_report_path"] = str(report_path)
        if not args.dump_memory_events:
            payload["memory"].pop("events", None)
        print(json.dumps(payload, indent=2))
        return
    print(format_summary(graph, result))
    print("")
    print(f"HTML report: {report_path}")


def _make_backend(name: str, device_config: str | None, calibration_config: str | None):
    hardware = load_hardware_config(device_config) if device_config else None
    calibration = load_backend_calibration(calibration_config) if calibration_config else None
    if name == "nvidia":
        return NvidiaBackend(hardware=hardware, calibration=calibration)
    if name == "ascend":
        return AscendBackend(hardware=hardware, calibration=calibration)
    raise ValueError(name)


def _make_graph_builder(args, config: ModelConfig, frontend_override: str | None = None):
    frontend = frontend_override or args.model_family
    if frontend == "transformer_fx":
        if not args.model_source or not args.model_class:
            raise ValueError("--model-source and --model-class are required for transformer_fx")
        return TorchFxGraphBuilder(config, source_path=args.model_source, model_class=args.model_class)
    if frontend == "transformer_export":
        if not args.model_source or not args.model_class:
            raise ValueError("--model-source and --model-class are required for transformer_export")
        return TorchExportGraphBuilder(config, source_path=args.model_source, model_class=args.model_class)
    if frontend == "backend_ir":
        if not args.backend_ir:
            raise ValueError("--backend-ir is required for backend_ir")
        return BackendIrGraphBuilder(args.backend_ir)
    return TransformerSourceGraphBuilder(config, source_path=args.model_source)


def _resolve_html_report_path(requested: str | None, graph_name: str, backend_name: str) -> Path:
    if requested:
        return Path(requested)
    safe_name = graph_name.replace("/", "_")
    return Path.cwd() / f"{safe_name}_{backend_name}.html"


if __name__ == "__main__":
    main()
