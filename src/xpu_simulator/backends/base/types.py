"""Backend-facing shared types."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...ir.types import OpKind


@dataclass
class HardwareConfig:
    name: str
    architecture: str
    compute_units: int
    peak_tflops: float
    mem_bandwidth_gbps: float
    launch_overhead_us: float
    on_chip_mem_mb: float = 0.0


@dataclass
class KernelTask:
    name: str
    op_kind: OpKind
    flops: float
    bytes_moved: float
    attrs: dict[str, object] = field(default_factory=dict)
    resource: str = "compute"


@dataclass
class KernelEstimate:
    task_name: str
    compute_time_us: float
    memory_time_us: float
    launch_overhead_us: float
    total_time_us: float
    bottleneck: str
    resource: str = "compute"
    start_time_us: float = 0.0
    end_time_us: float = 0.0
    predecessors: list[str] = field(default_factory=list)


@dataclass
class MemoryEvent:
    node_name: str
    live_bytes: int
    peak_bytes: int


@dataclass
class MemorySummary:
    peak_live_bytes: int
    final_live_bytes: int
    events: list[MemoryEvent] = field(default_factory=list)


@dataclass
class CacheSummary:
    mode: str
    context_len: int
    step_tokens: int
    kv_cache_bytes_per_layer: int
    kv_cache_total_bytes: int


@dataclass
class SimulationResult:
    backend_name: str
    device_name: str
    total_latency_us: float
    kernel_estimates: list[KernelEstimate]
    critical_path: list[str] = field(default_factory=list)
    memory_summary: MemorySummary | None = None
    cache_summary: CacheSummary | None = None

    @property
    def kernel_count(self) -> int:
        return len(self.kernel_estimates)
