"""Load backend calibration and tuning parameters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BackendCalibration:
    utilization: dict[str, float]
    bandwidth_scale: dict[str, float]
    modifiers: dict[str, float]

    def utilization_for(self, op_name: str) -> float:
        return self.utilization.get(op_name, self.utilization.get("default", 0.25))

    def bandwidth_for(self, op_name: str) -> float:
        return self.bandwidth_scale.get(op_name, self.bandwidth_scale.get("default", 0.5))

    def modifier(self, key: str, default: float = 1.0) -> float:
        return self.modifiers.get(key, default)


def load_backend_calibration(path: str | Path) -> BackendCalibration:
    raw = json.loads(Path(path).read_text())
    return BackendCalibration(
        utilization=raw.get("utilization", {}),
        bandwidth_scale=raw.get("bandwidth_scale", {}),
        modifiers=raw.get("modifiers", {}),
    )
