"""Backend-neutral IR enums."""

from __future__ import annotations

from enum import Enum


class DType(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"


class OpKind(str, Enum):
    EMBEDDING = "embedding"
    MATMUL = "matmul"
    BATCHED_MATMUL = "batched_matmul"
    NORM = "norm"
    SOFTMAX = "softmax"
    ELEMENTWISE = "elementwise"
    ROPE = "rope"
    TOPK = "topk"
    GATHER = "gather"
    SCATTER = "scatter"
    ALL_REDUCE = "all_reduce"
    CONCAT = "concat"
    RESHAPE = "reshape"
    TRANSPOSE = "transpose"
    OUTPUT = "output"
