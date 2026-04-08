"""Generic transformer/LLM source-driven frontend."""

from __future__ import annotations

from .deepseek import DeepSeekConfig, DeepSeekGraphBuilder
from .fx_frontend import TorchFxGraphBuilder
from .source_analysis import DeepSeekSourceAnalyzer


class ModelConfig(DeepSeekConfig):
    """Generic model config wrapper for source-driven transformer builders."""


class TransformerSourceGraphBuilder(DeepSeekGraphBuilder):
    """Generic source-driven graph builder for modern transformer-style LLMs."""


class SourceModelAnalyzer(DeepSeekSourceAnalyzer):
    """Generic AST-based analyzer for transformer/LLM model source code."""


class TransformerFxGraphBuilder(TorchFxGraphBuilder):
    """Optional torch.fx-based graph builder for executable transformer models."""
