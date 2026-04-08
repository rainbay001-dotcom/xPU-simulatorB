"""Calibration reporting helpers."""

from __future__ import annotations

from .csv_ingest import BenchmarkRow, summarize_benchmark_rows


def build_calibration_report(rows: list[BenchmarkRow]) -> dict[str, object]:
    summary = summarize_benchmark_rows(rows)
    recommendations: dict[str, dict[str, float]] = {}
    for op_name, values in summary.items():
        ratio = values["error_ratio"]
        recommended_utilization_scale = 1.0
        recommended_bandwidth_scale = 1.0
        if ratio > 1.0:
            recommended_utilization_scale = 1.0 / ratio
            recommended_bandwidth_scale = 1.0 / ratio
        elif ratio > 0.0:
            recommended_utilization_scale = 1.0 / ratio
            recommended_bandwidth_scale = 1.0 / ratio
        recommendations[op_name] = {
            "error_ratio": ratio,
            "recommended_utilization_scale": recommended_utilization_scale,
            "recommended_bandwidth_scale": recommended_bandwidth_scale,
        }
    return {
        "summary": summary,
        "recommendations": recommendations,
    }
