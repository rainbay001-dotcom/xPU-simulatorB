"""Generic transformer/LLM source-driven frontend."""

from __future__ import annotations

from .backend_ir_frontend import BackendIrGraphBuilder
from .deepseek import DeepSeekConfig, DeepSeekGraphBuilder
from .export_frontend import TorchExportGraphBuilder
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


class TransformerExportGraphBuilder(TorchExportGraphBuilder):
    """Optional torch.export-based graph builder for executable transformer models."""


class TransformerBackendIrGraphBuilder(BackendIrGraphBuilder):
    """Frontend for lowered backend/compiler IR JSON."""
