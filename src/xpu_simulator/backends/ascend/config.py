"""Preset Ascend hardware configs."""

from pathlib import Path

from ...calibration import load_backend_calibration
from ..base.config_loader import load_hardware_config
from ..base.types import HardwareConfig

_DEVICE_PATH = Path(__file__).resolve().parents[4] / "configs" / "devices" / "ascend_910b.json"
_CALIBRATION_PATH = Path(__file__).resolve().parents[4] / "configs" / "calibration" / "ascend_default.json"

ASCEND_910B = load_hardware_config(_DEVICE_PATH)
ASCEND_910B_CALIBRATION = load_backend_calibration(_CALIBRATION_PATH)
