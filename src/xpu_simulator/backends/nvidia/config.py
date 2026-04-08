"""Preset NVIDIA hardware configs."""

from pathlib import Path

from ..base.config_loader import load_hardware_config
from ..base.types import HardwareConfig

_DEVICE_PATH = Path(__file__).resolve().parents[4] / "configs" / "devices" / "nvidia_a100.json"

NVIDIA_A100 = load_hardware_config(_DEVICE_PATH)
