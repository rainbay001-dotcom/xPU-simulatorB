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
    cache_summary = result.cache_summary
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
    .timeline-shell {{ overflow-x: auto; padding: 8px 0; }}
    .timeline-svg text {{ font-family: Menlo, Monaco, Consolas, monospace; fill: #40372d; }}
    .timeline-axis {{ stroke: #8f7c61; stroke-width: 1; }}
    .timeline-grid {{ stroke: rgba(143,124,97,0.28); stroke-width: 1; stroke-dasharray: 3 4; }}
    .timeline-lane {{ fill: rgba(255,255,255,0.35); stroke: rgba(121,98,65,0.14); }}
    .timeline-bar {{ rx: 6; ry: 6; }}
    .timeline-bar.compute {{ fill: rgba(200,107,60,0.82); }}
    .timeline-bar.memory {{ fill: rgba(46,125,116,0.82); }}
    .timeline-bar.communication {{ fill: rgba(125,90,181,0.82); }}
    .timeline-label {{ font-size: 11px; }}
    .timeline-title {{ font-size: 12px; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>{escape(graph.name)}</h1>
  <div class="grid">
    <div class="card">
      <h2>Run Summary</h2>
      <p>Backend: <strong>{escape(result.backend_name)}</strong> ({escape(result.device_name)})</p>
      <p>Total latency: <strong>{result.total_latency_us:.3f} us</strong></p>
      <p>Mode: <strong>{escape(str(graph.metadata.get("mode", "prefill")))}</strong></p>
      <p>Fusion: <strong>{'enabled' if graph.metadata.get("fusion_requested", False) else 'disabled'}</strong>, fused nodes: <strong>{int(graph.metadata.get("fused_node_count", 0))}</strong></p>
      <p>Context / step tokens: <strong>{int(graph.metadata.get("context_len", graph.metadata.get("seq_len", 0)))}</strong> / <strong>{int(graph.metadata.get("step_tokens", graph.metadata.get("seq_len", 0)))}</strong></p>
      <p>Nodes / edges: <strong>{graph.node_count()}</strong> / <strong>{graph.edge_count()}</strong></p>
      <p>Peak live memory: <strong>{_format_bytes(result.memory_summary.peak_live_bytes if result.memory_summary else 0)}</strong></p>
      {_cache_block(cache_summary)}
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
    <h2>Timeline View</h2>
    <p class="muted">NSight-style simulated timeline grouped into compute, memory, and communication lanes.</p>
    {_timeline_view(result)}
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


def _cache_block(cache_summary) -> str:
    if cache_summary is None:
        return ""
    return (
        f"<p>KV cache total: <strong>{_format_bytes(cache_summary.kv_cache_total_bytes)}</strong></p>"
        f"<p>KV cache per layer: <strong>{_format_bytes(cache_summary.kv_cache_bytes_per_layer)}</strong></p>"
    )


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


def _timeline_view(result: SimulationResult) -> str:
    lane_order = ["compute", "memory", "communication"]
    lane_titles = {
        "compute": "Compute lane",
        "memory": "Memory lane",
        "communication": "Communication lane",
    }
    lane_items = {
        lane: [item for item in result.kernel_estimates if item.resource == lane]
        for lane in lane_order
    }
    width = 1200
    left_pad = 150
    right_pad = 36
    top_pad = 32
    lane_height = 88
    row_step = 16
    total_height = top_pad + (lane_height * len(lane_order)) + 40
    time_span = max(result.total_latency_us, max((item.end_time_us for item in result.kernel_estimates), default=0.0), 1.0)
    plot_width = width - left_pad - right_pad

    parts: list[str] = [
        f"<div class='timeline-shell'><svg class='timeline-svg' viewBox='0 0 {width} {total_height}' width='{width}' height='{total_height}' role='img' aria-label='Simulated execution timeline'>"
    ]
    tick_count = 8
    for tick in range(tick_count + 1):
        x = left_pad + (plot_width * tick / tick_count)
        time_us = time_span * tick / tick_count
        parts.append(f"<line class='timeline-grid' x1='{x:.2f}' y1='{top_pad - 10}' x2='{x:.2f}' y2='{total_height - 24}' />")
        parts.append(f"<text class='timeline-label' x='{x:.2f}' y='18' text-anchor='middle'>{time_us:.1f} us</text>")
    parts.append(f"<line class='timeline-axis' x1='{left_pad}' y1='{top_pad - 2}' x2='{width - right_pad}' y2='{top_pad - 2}' />")

    for lane_index, lane in enumerate(lane_order):
        y = top_pad + lane_index * lane_height
        parts.append(f"<rect class='timeline-lane' x='{left_pad}' y='{y}' width='{plot_width}' height='{lane_height - 10}' rx='10' ry='10' />")
        parts.append(f"<text class='timeline-title' x='12' y='{y + 22}'>{escape(lane_titles[lane])}</text>")
        parts.append(f"<text class='timeline-label' x='12' y='{y + 40}'>{len(lane_items[lane])} kernels</text>")

        items = sorted(lane_items[lane], key=lambda item: item.start_time_us)
        lane_bottom = y + lane_height - 24
        for item_index, item in enumerate(items):
            bar_x = left_pad + (item.start_time_us / time_span) * plot_width
            bar_width = max((item.total_time_us / time_span) * plot_width, 2.0)
            row = item_index % 3
            bar_y = y + 10 + row * row_step
            parts.append(
                f"<rect class='timeline-bar {lane}' x='{bar_x:.2f}' y='{bar_y:.2f}' width='{bar_width:.2f}' height='11'>"
                f"<title>{escape(item.task_name)} | start={item.start_time_us:.3f} us | end={item.end_time_us:.3f} us | total={item.total_time_us:.3f} us</title>"
                "</rect>"
            )
            if item_index < 12:
                label_x = min(bar_x + bar_width + 4, width - right_pad - 40)
                parts.append(
                    f"<text class='timeline-label' x='{label_x:.2f}' y='{bar_y + 9:.2f}'>{escape(_compact_label(item.task_name))}</text>"
                )
        parts.append(f"<line class='timeline-axis' x1='{left_pad}' y1='{lane_bottom:.2f}' x2='{width - right_pad}' y2='{lane_bottom:.2f}' />")

    parts.append("</svg></div>")
    return "".join(parts)


def _compact_label(name: str) -> str:
    if name.startswith("layer_"):
        pieces = name.split("_")
        if len(pieces) >= 3:
            return f"{pieces[0]}_{pieces[1]}:{pieces[2]}"
    return name


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
