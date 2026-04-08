"""Simplified Ascend backend."""

from __future__ import annotations

from ...ir.types import OpKind
from ...calibration import BackendCalibration
from ..base.backend import Backend
from ..base.types import HardwareConfig
from ..base.types import KernelEstimate, KernelTask
from .config import ASCEND_910B, ASCEND_910B_CALIBRATION


class AscendBackend(Backend):
    def __init__(self, hardware: HardwareConfig | None = None, calibration: BackendCalibration | None = None):
        super().__init__(hardware=hardware or ASCEND_910B)
        self.calibration = calibration or ASCEND_910B_CALIBRATION

    @property
    def name(self) -> str:
        return "ascend"

    def estimate_kernel(self, task: KernelTask) -> KernelEstimate:
        compute_time_us = self._estimate_compute_time(task)
        memory_time_us = self._estimate_memory_time(task)

        dominant = "compute" if compute_time_us >= memory_time_us else "memory"
        total_time_us = max(compute_time_us, memory_time_us) + self.hardware.launch_overhead_us
        return KernelEstimate(
            task_name=task.name,
            compute_time_us=compute_time_us,
            memory_time_us=memory_time_us,
            launch_overhead_us=self.hardware.launch_overhead_us,
            total_time_us=total_time_us,
            bottleneck=dominant,
            resource=task.resource,
        )

    def _estimate_compute_time(self, task: KernelTask) -> float:
        if task.flops <= 0:
            return 0.0
        utilization = self.calibration.utilization_for(task.op_kind.value)
        mode = str(task.attrs.get("execution_mode", "prefill"))
        if task.attrs.get("uses_fp8", False):
            utilization *= self.calibration.modifier("fp8_utilization_boost", 1.05)
        if task.attrs.get("moe", False):
            utilization *= self.calibration.modifier("moe_utilization_penalty", 0.95)
        attention_variant = task.attrs.get("attention_variant")
        if attention_variant == "gqa":
            utilization *= self.calibration.modifier("gqa_utilization_boost", 1.02)
        elif attention_variant == "mqa":
            utilization *= self.calibration.modifier("mqa_utilization_boost", 1.04)
        if mode == "decode" and task.op_kind in {OpKind.BATCHED_MATMUL, OpKind.SOFTMAX}:
            utilization *= self.calibration.modifier("decode_attention_utilization_penalty", 0.78)
        effective_tflops = max(self.hardware.peak_tflops * utilization, 1e-6)
        return task.flops / (effective_tflops * 1e6)

    def _estimate_memory_time(self, task: KernelTask) -> float:
        if task.bytes_moved <= 0:
            return 0.0
        bandwidth_scale = self.calibration.bandwidth_for(task.op_kind.value)
        mode = str(task.attrs.get("execution_mode", "prefill"))
        if task.attrs.get("moe", False):
            bandwidth_scale *= self.calibration.modifier("moe_bandwidth_penalty", 0.80)
        attention_variant = task.attrs.get("attention_variant")
        if attention_variant == "gqa":
            bandwidth_scale *= self.calibration.modifier("gqa_bandwidth_boost", 1.04)
        elif attention_variant == "mqa":
            bandwidth_scale *= self.calibration.modifier("mqa_bandwidth_boost", 1.06)
        if task.attrs.get("cache_op", False):
            action = str(task.attrs.get("cache_action", "read"))
            modifier = "cache_write_bandwidth_penalty" if action == "write" else "cache_read_bandwidth_penalty"
            bandwidth_scale *= self.calibration.modifier(modifier, 0.75 if action == "write" else 0.82)
        if mode == "decode" and task.op_kind in {OpKind.BATCHED_MATMUL, OpKind.SOFTMAX, OpKind.GATHER, OpKind.SCATTER}:
            bandwidth_scale *= self.calibration.modifier("decode_attention_bandwidth_penalty", 0.72)
        effective_bandwidth = max(self.hardware.mem_bandwidth_gbps * bandwidth_scale, 1e-6)
        return task.bytes_moved / (effective_bandwidth * 1e3)
