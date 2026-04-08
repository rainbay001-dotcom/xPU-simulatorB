"""Helpers for loading device configuration from JSON files."""

from __future__ import annotations

import json
from pathlib import Path

from .types import HardwareConfig


def load_hardware_config(path: str | Path) -> HardwareConfig:
    raw = json.loads(Path(path).read_text())
    return HardwareConfig(
        name=raw["name"],
        architecture=raw["architecture"],
        compute_units=raw["compute_units"],
        peak_tflops=raw["peak_tflops"],
        mem_bandwidth_gbps=raw["mem_bandwidth_gbps"],
        launch_overhead_us=raw["launch_overhead_us"],
        on_chip_mem_mb=raw.get("on_chip_mem_mb", 0.0),
    )
