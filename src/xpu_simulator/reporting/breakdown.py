"""Breakdown summaries for simulated runs."""

from __future__ import annotations

from collections import defaultdict

from ..backends.base.types import SimulationResult


def result_breakdown(result: SimulationResult) -> dict[str, object]:
    by_resource: dict[str, float] = defaultdict(float)
    by_family: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0.0, "total_time_us": 0.0})
    by_layer: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0.0, "total_time_us": 0.0})

    for item in result.kernel_estimates:
        by_resource[item.resource] += item.total_time_us

        family = kernel_family(item.task_name)
        by_family[family]["count"] += 1.0
        by_family[family]["total_time_us"] += item.total_time_us

        layer = layer_name(item.task_name)
        if layer is not None:
            by_layer[layer]["count"] += 1.0
            by_layer[layer]["total_time_us"] += item.total_time_us

    family_rows = [
        {
            "family": family,
            "count": int(values["count"]),
            "total_time_us": values["total_time_us"],
            "share": values["total_time_us"] / result.total_latency_us if result.total_latency_us else 0.0,
        }
        for family, values in by_family.items()
    ]
    family_rows.sort(key=lambda row: row["total_time_us"], reverse=True)

    layer_rows = [
        {
            "layer": layer,
            "count": int(values["count"]),
            "total_time_us": values["total_time_us"],
            "share": values["total_time_us"] / result.total_latency_us if result.total_latency_us else 0.0,
        }
        for layer, values in by_layer.items()
    ]
    layer_rows.sort(key=lambda row: int(row["layer"].split("_")[1]))

    resource_rows = [
        {
            "resource": resource,
            "total_time_us": total,
            "share": total / result.total_latency_us if result.total_latency_us else 0.0,
        }
        for resource, total in by_resource.items()
    ]
    resource_rows.sort(key=lambda row: row["total_time_us"], reverse=True)

    return {
        "by_resource": resource_rows,
        "by_family": family_rows,
        "by_layer": layer_rows,
    }


def format_breakdown_table(result: SimulationResult, top_families: int = 10, top_layers: int = 8) -> str:
    breakdown = result_breakdown(result)
    lines = ["Breakdown:"]
    lines.append("Resource        Total us      Share")
    for row in breakdown["by_resource"]:
        lines.append(f"{row['resource']:<14} {row['total_time_us']:>10.3f}  {row['share'] * 100:>6.2f}%")

    lines.append("")
    lines.append("Top families    Count   Total us      Share")
    for row in breakdown["by_family"][:top_families]:
        lines.append(
            f"{row['family']:<14} {row['count']:>5}  {row['total_time_us']:>10.3f}  {row['share'] * 100:>6.2f}%"
        )

    if breakdown["by_layer"]:
        lines.append("")
        lines.append("Top layers      Kernels Total us      Share")
        layer_rows = sorted(breakdown["by_layer"], key=lambda row: row["total_time_us"], reverse=True)[:top_layers]
        for row in layer_rows:
            lines.append(
                f"{row['layer']:<14} {row['count']:>7} {row['total_time_us']:>10.3f}  {row['share'] * 100:>6.2f}%"
            )
    return "\n".join(lines)


def kernel_family(name: str) -> str:
    if name == "embedding":
        return "embedding"
    if name == "lm_head":
        return "lm_head"
    if name.endswith("_fused_attention"):
        return "fused_attention"
    if name.endswith("_fused_router_dispatch"):
        return "fused_router_dispatch"
    if name.endswith("_kv_cache_read"):
        return "kv_cache_read"
    if name.endswith("_kv_cache_write"):
        return "kv_cache_write"
    if name.endswith("_softmax"):
        return "softmax"
    if name.endswith("_attn_scores") or name.endswith("_cross_attn_scores"):
        return "attn_scores"
    if name.endswith("_attn_out") or name.endswith("_cross_attn_out"):
        return "attn_out"
    if name.endswith("_attn_topk"):
        return "attn_topk"
    if name.endswith("_tensor_parallel_sync"):
        return "all_reduce"
    if name.endswith("_router"):
        return "router"
    if name.endswith("_dispatch"):
        return "dispatch"
    if name.endswith("_combine"):
        return "combine"
    if name.endswith("_shared_ffn_gate") or name.endswith("_ffn_gate"):
        return "gating"
    if name.endswith("_shared_ffn_down") or name.endswith("_down_proj") or name.endswith("_ffn_down"):
        return "ffn_down"
    if any(name.endswith(suffix) for suffix in ["_shared_ffn_w1", "_shared_up_proj", "_up_proj", "_ffn_w1"]):
        return "ffn_up"
    if any(name.endswith(suffix) for suffix in ["_shared_ffn_w3", "_shared_gate_proj", "_gate_proj", "_ffn_w3"]):
        return "ffn_gate_proj"
    if any(name.endswith(suffix) for suffix in ["_q_proj", "_k_proj", "_v_proj", "_qkv_proj", "_wq_a", "_wq_b", "_wkv_a", "_wkv_b"]):
        return "qkv_proj"
    if name.endswith("_wo") or name.endswith("_o_proj") or name.endswith("_out_proj"):
        return "attn_output_proj"
    if name.endswith("_attn_proj_merge"):
        return "attn_proj_merge"
    if name.endswith("_q_norm") or name.endswith("_kv_norm") or name.endswith("_attn_norm") or name.endswith("_ffn_norm"):
        return "norm"
    if name.endswith("_rope"):
        return "rope"
    if name.endswith("_expert_ffn"):
        return "expert_ffn"
    return "other"


def layer_name(name: str) -> str | None:
    if name.startswith("layer_"):
        return "_".join(name.split("_")[:2])
    return None
