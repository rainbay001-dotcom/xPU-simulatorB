"""HTML report generation for simulated runs."""

from __future__ import annotations

from collections import defaultdict
from html import escape
from pathlib import Path

from ..backends.base.types import SimulationResult
from ..ir.graph import Graph
from .breakdown import kernel_family, result_breakdown


def write_html_report(graph: Graph, result: SimulationResult, output_path: str | Path) -> str:
    path = Path(output_path)
    path.write_text(render_html_report(graph, result))
    return str(path)


def render_html_report(graph: Graph, result: SimulationResult) -> str:
    breakdown = result_breakdown(result)
    layer_kernels: dict[str, list[dict[str, object]]] = defaultdict(list)
    preamble: list[dict[str, object]] = []
    epilogue: list[dict[str, object]] = []

    for item in result.kernel_estimates:
        record = {
            "name": item.task_name,
            "total_time_us": item.total_time_us,
            "resource": item.resource,
            "bottleneck": item.bottleneck,
            "family": kernel_family(item.task_name),
        }
        if item.task_name.startswith("layer_"):
            layer_kernels["_".join(item.task_name.split("_")[:2])].append(record)
        elif item.task_name == "embedding":
            preamble.append(record)
        else:
            epilogue.append(record)

    architecture = graph.metadata.get("architecture", {})
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(graph.name)} simulation report</title>
  <style>
    body {{ font-family: Menlo, Monaco, Consolas, monospace; margin: 24px; color: #18222d; background: #f6f1e8; }}
    h1, h2, h3 {{ margin-bottom: 8px; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(280px, 1fr)); gap: 16px; margin-bottom: 24px; }}
    .card {{ background: #fffdf8; border: 1px solid #d8ccbb; padding: 16px; border-radius: 10px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 6px 8px; border-bottom: 1px solid #eadfce; text-align: left; }}
    th {{ background: #f0e6d7; }}
    details {{ margin: 10px 0; background: #fffdf8; border: 1px solid #d8ccbb; border-radius: 10px; padding: 10px 14px; }}
    summary {{ cursor: pointer; font-weight: 700; }}
    .muted {{ color: #6a7178; }}
  </style>
</head>
<body>
  <h1>{escape(graph.name)}</h1>
  <div class="grid">
    <div class="card">
      <h2>Run Summary</h2>
      <p>Backend: <strong>{escape(result.backend_name)}</strong> ({escape(result.device_name)})</p>
      <p>Total latency: <strong>{result.total_latency_us:.3f} us</strong></p>
      <p>Nodes / edges: <strong>{graph.node_count()}</strong> / <strong>{graph.edge_count()}</strong></p>
      <p>Peak live memory: <strong>{_format_bytes(result.memory_summary.peak_live_bytes if result.memory_summary else 0)}</strong></p>
    </div>
    <div class="card">
      <h2>Architecture</h2>
      <p>Model class: <strong>{escape(str(architecture.get("model_class")))}</strong></p>
      <p>Block class: <strong>{escape(str(architecture.get("block_class")))}</strong></p>
      <p>Attention class: <strong>{escape(str(architecture.get("attention_class")))}</strong></p>
      <p>Dense FFN class: <strong>{escape(str(architecture.get("dense_ffn_class")))}</strong></p>
      <p>MoE FFN class: <strong>{escape(str(architecture.get("moe_ffn_class")))}</strong></p>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <h2>Resource Breakdown</h2>
      {_resource_table(breakdown["by_resource"])}
    </div>
    <div class="card">
      <h2>Top Kernel Families</h2>
      {_family_table(breakdown["by_family"][:12])}
    </div>
  </div>

  <div class="card">
    <h2>Architecture Timeline</h2>
    {_kernel_table(preamble, "Preamble kernels")}
    {_layer_sections(layer_kernels, result.total_latency_us)}
    {_kernel_table(epilogue, "Output kernels")}
  </div>
</body>
</html>
"""


def _resource_table(rows: list[dict[str, object]]) -> str:
    body = "\n".join(
        f"<tr><td>{escape(str(row['resource']))}</td><td>{row['total_time_us']:.3f}</td><td>{row['share'] * 100:.2f}%</td></tr>"
        for row in rows
    )
    return f"<table><tr><th>Resource</th><th>Total us</th><th>Share</th></tr>{body}</table>"


def _family_table(rows: list[dict[str, object]]) -> str:
    body = "\n".join(
        f"<tr><td>{escape(str(row['family']))}</td><td>{row['count']}</td><td>{row['total_time_us']:.3f}</td><td>{row['share'] * 100:.2f}%</td></tr>"
        for row in rows
    )
    return f"<table><tr><th>Family</th><th>Count</th><th>Total us</th><th>Share</th></tr>{body}</table>"


def _kernel_table(rows: list[dict[str, object]], title: str) -> str:
    if not rows:
        return ""
    body = "\n".join(
        f"<tr><td>{escape(str(row['name']))}</td><td>{escape(str(row['family']))}</td><td>{row['total_time_us']:.3f}</td><td>{escape(str(row['resource']))}</td><td>{escape(str(row['bottleneck']))}</td></tr>"
        for row in rows
    )
    return f"<h3>{escape(title)}</h3><table><tr><th>Kernel</th><th>Family</th><th>Total us</th><th>Resource</th><th>Bottleneck</th></tr>{body}</table>"


def _layer_sections(layer_kernels: dict[str, list[dict[str, object]]], total_latency_us: float) -> str:
    sections: list[str] = []
    for layer_name in sorted(layer_kernels, key=lambda value: int(value.split("_")[1])):
        kernels = layer_kernels[layer_name]
        layer_total = sum(float(row["total_time_us"]) for row in kernels)
        share = (layer_total / total_latency_us * 100) if total_latency_us else 0.0
        body = "\n".join(
            f"<tr><td>{escape(str(row['name']))}</td><td>{escape(str(row['family']))}</td><td>{row['total_time_us']:.3f}</td><td>{escape(str(row['resource']))}</td><td>{escape(str(row['bottleneck']))}</td></tr>"
            for row in sorted(kernels, key=lambda row: float(row["total_time_us"]), reverse=True)
        )
        sections.append(
            f"<details><summary>{escape(layer_name)} <span class='muted'>({layer_total:.3f} us, {share:.2f}% of wall time)</span></summary>"
            f"<table><tr><th>Kernel</th><th>Family</th><th>Total us</th><th>Resource</th><th>Bottleneck</th></tr>{body}</table></details>"
        )
    return "\n".join(sections)


def _format_bytes(value: int) -> str:
    if value >= 1024 * 1024 * 1024:
        return f"{value / (1024 * 1024 * 1024):.2f} GiB"
    if value >= 1024 * 1024:
        return f"{value / (1024 * 1024):.2f} MiB"
    if value >= 1024:
        return f"{value / 1024:.2f} KiB"
    return f"{value} B"
