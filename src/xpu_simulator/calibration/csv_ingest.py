"""CSV benchmark ingestion for coarse calibration summaries."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchmarkRow:
    op_name: str
    measured_time_us: float
    predicted_time_us: float
    bytes_moved: float
    flops: float


def load_benchmark_rows(path: str | Path) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            rows.append(
                BenchmarkRow(
                    op_name=raw["op_name"],
                    measured_time_us=float(raw["measured_time_us"]),
                    predicted_time_us=float(raw["predicted_time_us"]),
                    bytes_moved=float(raw.get("bytes_moved", 0.0)),
                    flops=float(raw.get("flops", 0.0)),
                )
            )
    return rows


def summarize_benchmark_rows(rows: list[BenchmarkRow]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[BenchmarkRow]] = defaultdict(list)
    for row in rows:
        grouped[row.op_name].append(row)

    summary: dict[str, dict[str, float]] = {}
    for op_name, op_rows in grouped.items():
        mean_measured = sum(row.measured_time_us for row in op_rows) / len(op_rows)
        mean_predicted = sum(row.predicted_time_us for row in op_rows) / len(op_rows)
        ratio = mean_measured / mean_predicted if mean_predicted else 0.0
        summary[op_name] = {
            "samples": float(len(op_rows)),
            "mean_measured_time_us": mean_measured,
            "mean_predicted_time_us": mean_predicted,
            "error_ratio": ratio,
        }
    return summary
