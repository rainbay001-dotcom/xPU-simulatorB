"""Preset Ascend hardware configs."""

from pathlib import Path

from ..base.config_loader import load_hardware_config
from ..base.types import HardwareConfig

_DEVICE_PATH = Path(__file__).resolve().parents[4] / "configs" / "devices" / "ascend_910b.json"

ASCEND_910B = load_hardware_config(_DEVICE_PATH)
