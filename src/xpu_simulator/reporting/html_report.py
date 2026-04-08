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
            "start_time_us": item.start_time_us,
            "end_time_us": item.end_time_us,
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
    :root {{
      --paper: #f5efe2;
      --ink: #1f2a2d;
      --muted: #677074;
      --frame: #cdbd9d;
      --card: #fffaf0;
      --compute: #c86b3c;
      --memory: #2e7d74;
      --communication: #7d5ab5;
      --bottleneck-compute: #f6d5bb;
      --bottleneck-memory: #cfe9e4;
      --bottleneck-communication: #ddd3f2;
      --layer-band: #e9ddc7;
      --preamble-band: #d7e2c4;
      --output-band: #e3d0b8;
    }}
    body {{ font-family: Georgia, 'Iowan Old Style', 'Palatino Linotype', serif; margin: 24px; color: var(--ink); background:
      radial-gradient(circle at top left, rgba(255,255,255,0.45), transparent 28%),
      linear-gradient(180deg, #f7f1e4 0%, #efe5d4 100%); }}
    h1, h2, h3 {{ margin-bottom: 8px; font-family: 'Baskerville', 'Times New Roman', serif; letter-spacing: 0.02em; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(280px, 1fr)); gap: 16px; margin-bottom: 24px; }}
    .card {{ background: linear-gradient(180deg, rgba(255,252,246,0.97), rgba(252,246,235,0.97)); border: 1px solid var(--frame); padding: 16px; border-radius: 14px; box-shadow: 0 8px 22px rgba(88, 69, 41, 0.08); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; font-family: Menlo, Monaco, Consolas, monospace; }}
    th, td {{ padding: 6px 8px; border-bottom: 1px solid #eadfce; text-align: left; }}
    th {{ background: #f0e6d7; }}
    details {{ margin: 10px 0; background: #fffdf8; border: 1px solid #d8ccbb; border-radius: 10px; padding: 10px 14px; }}
    summary {{ cursor: pointer; font-weight: 700; }}
    .muted {{ color: var(--muted); }}
    .arch-legend {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 12px 0 16px 0; }}
    .legend-chip {{ display: inline-flex; align-items: center; gap: 8px; padding: 6px 10px; border: 1px solid var(--frame); border-radius: 999px; background: rgba(255,255,255,0.55); font-size: 12px; font-family: Menlo, Monaco, Consolas, monospace; }}
    .swatch {{ width: 12px; height: 12px; border-radius: 999px; display: inline-block; }}
    .arch-graph {{ display: flex; flex-wrap: nowrap; gap: 18px; align-items: stretch; overflow-x: auto; padding: 8px 2px 18px 2px; }}
    .stage-box {{ min-width: 290px; max-width: 350px; background: linear-gradient(180deg, rgba(255,253,248,0.96), rgba(252,245,233,0.96)); border: 1px solid var(--frame); border-radius: 14px; padding: 0; box-shadow: 0 6px 16px rgba(96, 73, 44, 0.08); overflow: hidden; }}
    .stage-box h3 {{ margin: 0; }}
    .stage-header {{ padding: 12px 14px; border-bottom: 1px solid rgba(140,118,82,0.18); }}
    .stage-header.stage-layer {{ background: linear-gradient(180deg, var(--layer-band), #f0e6d7); }}
    .stage-header.stage-preamble {{ background: linear-gradient(180deg, var(--preamble-band), #e6edd7); }}
    .stage-header.stage-output {{ background: linear-gradient(180deg, var(--output-band), #eddcc8); }}
    .stage-body {{ padding: 10px 12px 12px 12px; }}
    .stage-meta {{ font-size: 12px; color: var(--muted); margin-top: 4px; margin-bottom: 2px; font-family: Menlo, Monaco, Consolas, monospace; }}
    .kernel-node {{ border-radius: 10px; padding: 10px; margin: 10px 0; border-left: 6px solid var(--compute); box-shadow: inset 0 0 0 1px rgba(120, 99, 66, 0.12); }}
    .kernel-node.resource-compute {{ border-left-color: var(--compute); }}
    .kernel-node.resource-memory {{ border-left-color: var(--memory); }}
    .kernel-node.resource-communication {{ border-left-color: var(--communication); }}
    .kernel-node.bottleneck-compute {{ background: linear-gradient(180deg, var(--bottleneck-compute), #fbf2e7); }}
    .kernel-node.bottleneck-memory {{ background: linear-gradient(180deg, var(--bottleneck-memory), #f0faf7); }}
    .kernel-node.bottleneck-communication {{ background: linear-gradient(180deg, var(--bottleneck-communication), #f6f2fd); }}
    .kernel-node strong {{ display: block; margin-bottom: 2px; }}
    .kernel-time {{ font-size: 18px; font-weight: 700; margin: 3px 0 6px 0; font-family: Menlo, Monaco, Consolas, monospace; }}
    .kernel-tags {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 6px; }}
    .tag {{ display: inline-block; padding: 2px 7px; border-radius: 999px; font-size: 11px; font-family: Menlo, Monaco, Consolas, monospace; background: rgba(255,255,255,0.58); border: 1px solid rgba(110,90,57,0.18); }}
    .kernel-meta {{ font-size: 12px; color: #5b6470; font-family: Menlo, Monaco, Consolas, monospace; }}
    .arrow {{ font-size: 34px; color: #9a8667; align-self: center; padding: 0 2px; }}
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
    <h2>Architecture Graph</h2>
    <p class="muted">Model stages are grouped by layer. Node border color shows resource class, and node fill color shows the dominant bottleneck.</p>
    {_graph_legend()}
    {_architecture_graph(preamble, layer_kernels, epilogue, result.total_latency_us)}
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


def _architecture_graph(
    preamble: list[dict[str, object]],
    layer_kernels: dict[str, list[dict[str, object]]],
    epilogue: list[dict[str, object]],
    total_latency_us: float,
) -> str:
    stages: list[str] = []
    if preamble:
        stages.append(_stage_box("Preamble", preamble, total_latency_us))
    for layer_name in sorted(layer_kernels, key=lambda value: int(value.split("_")[1])):
        stages.append(_stage_box(layer_name, layer_kernels[layer_name], total_latency_us))
    if epilogue:
        stages.append(_stage_box("Output", epilogue, total_latency_us))
    if not stages:
        return "<p class='muted'>No stage data available.</p>"
    return "<div class='arch-graph'>" + "<div class='arrow'>&rarr;</div>".join(stages) + "</div>"


def _stage_box(title: str, kernels: list[dict[str, object]], total_latency_us: float) -> str:
    stage_total = sum(float(row["total_time_us"]) for row in kernels)
    share = (stage_total / total_latency_us * 100) if total_latency_us else 0.0
    stage_kind = "stage-layer"
    if title == "Preamble":
        stage_kind = "stage-preamble"
    elif title == "Output":
        stage_kind = "stage-output"
    body = "\n".join(
        _kernel_node(row)
        for row in sorted(kernels, key=lambda row: float(row["start_time_us"]))
    )
    return (
        "<section class='stage-box'>"
        f"<div class='stage-header {stage_kind}'><h3>{escape(title)}</h3>"
        f"<div class='stage-meta'>{stage_total:.3f} us, {share:.2f}% of wall time, {len(kernels)} kernels</div></div>"
        f"<div class='stage-body'>{body}</div>"
        "</section>"
    )


def _kernel_node(row: dict[str, object]) -> str:
    resource = escape(str(row["resource"]))
    bottleneck = escape(str(row["bottleneck"]))
    return (
        f"<div class='kernel-node resource-{resource} bottleneck-{bottleneck}'>"
        f"<strong>{escape(str(row['name']))}</strong>"
        f"<div class='kernel-time'>{row['total_time_us']:.3f} us</div>"
        f"<div class='kernel-tags'><span class='tag'>{escape(str(row['family']))}</span><span class='tag'>{resource}</span><span class='tag'>{bottleneck}</span></div>"
        f"<div class='kernel-meta'>start={float(row['start_time_us']):.3f} us | end={float(row['end_time_us']):.3f} us</div>"
        "</div>"
    )


def _graph_legend() -> str:
    return (
        "<div class='arch-legend'>"
        "<div class='legend-chip'><span class='swatch' style='background: var(--compute)'></span>resource: compute</div>"
        "<div class='legend-chip'><span class='swatch' style='background: var(--memory)'></span>resource: memory</div>"
        "<div class='legend-chip'><span class='swatch' style='background: var(--communication)'></span>resource: communication</div>"
        "<div class='legend-chip'><span class='swatch' style='background: var(--bottleneck-compute)'></span>bottleneck: compute</div>"
        "<div class='legend-chip'><span class='swatch' style='background: var(--bottleneck-memory)'></span>bottleneck: memory</div>"
        "<div class='legend-chip'><span class='swatch' style='background: var(--bottleneck-communication)'></span>bottleneck: communication</div>"
        "</div>"
    )


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
