"""Simplified NVIDIA backend."""

from __future__ import annotations

from ...ir.types import OpKind
from ..base.backend import Backend
from ..base.types import HardwareConfig
from ..base.types import KernelEstimate, KernelTask
from .config import NVIDIA_A100


class NvidiaBackend(Backend):
    def __init__(self, hardware: HardwareConfig | None = None):
        super().__init__(hardware=hardware or NVIDIA_A100)

    @property
    def name(self) -> str:
        return "nvidia"

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
        utilization = {
            OpKind.MATMUL: 0.78,
            OpKind.BATCHED_MATMUL: 0.72,
            OpKind.OUTPUT: 0.70,
            OpKind.NORM: 0.25,
            OpKind.SOFTMAX: 0.22,
            OpKind.ELEMENTWISE: 0.18,
            OpKind.ROPE: 0.20,
            OpKind.ALL_REDUCE: 0.05,
        }.get(task.op_kind, 0.30)
        if task.attrs.get("uses_fp8", False):
            utilization *= 1.10
        effective_tflops = max(self.hardware.peak_tflops * utilization, 1e-6)
        return task.flops / (effective_tflops * 1e6)

    def _estimate_memory_time(self, task: KernelTask) -> float:
        if task.bytes_moved <= 0:
            return 0.0
        bandwidth_scale = {
            OpKind.EMBEDDING: 0.55,
            OpKind.MATMUL: 0.72,
            OpKind.BATCHED_MATMUL: 0.68,
            OpKind.SOFTMAX: 0.58,
            OpKind.NORM: 0.62,
            OpKind.ELEMENTWISE: 0.70,
            OpKind.TOPK: 0.42,
            OpKind.GATHER: 0.45,
            OpKind.SCATTER: 0.45,
            OpKind.ALL_REDUCE: 0.35,
            OpKind.OUTPUT: 0.60,
        }.get(task.op_kind, 0.55)
        if task.attrs.get("moe", False):
            bandwidth_scale *= 0.85
        effective_bandwidth = max(self.hardware.mem_bandwidth_gbps * bandwidth_scale, 1e-6)
        return task.bytes_moved / (effective_bandwidth * 1e3)
