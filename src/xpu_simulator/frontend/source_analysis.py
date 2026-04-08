"""Lightweight DeepSeek source analyzer using Python AST."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SourceFeatureSummary:
    classes: list[str]
    functions: list[str]
    uses_fp8: bool
    uses_moe: bool
    uses_rotary: bool
    uses_distributed: bool
    uses_topk: bool
    parallel_linears: list[str]
    block_components: list[str]
    transformer_components: list[str]

    def to_metadata(self) -> dict[str, object]:
        return {
            "classes": self.classes,
            "functions": self.functions,
            "uses_fp8": self.uses_fp8,
            "uses_moe": self.uses_moe,
            "uses_rotary": self.uses_rotary,
            "uses_distributed": self.uses_distributed,
            "uses_topk": self.uses_topk,
            "parallel_linears": self.parallel_linears,
            "block_components": self.block_components,
            "transformer_components": self.transformer_components,
        }


class DeepSeekSourceAnalyzer:
    def analyze(self, source_path: str | Path) -> SourceFeatureSummary:
        path = Path(source_path)
        tree = ast.parse(path.read_text())

        classes = sorted(node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
        functions = sorted(node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))

        names = set()
        attrs = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                attrs.add(node.attr)

        all_symbols = names | attrs | set(classes) | set(functions)
        parallel_linears = [name for name in classes if "ParallelLinear" in name]
        block_components = self._extract_self_assignments(tree, "Block")
        transformer_components = self._extract_self_assignments(tree, "Transformer")

        return SourceFeatureSummary(
            classes=classes,
            functions=functions,
            uses_fp8=any(symbol in all_symbols for symbol in {"fp8_gemm", "act_quant", "float8_e4m3fn", "fp8_index"}),
            uses_moe=any(symbol in all_symbols for symbol in {"Gate", "MoE", "topk", "n_routed_experts"}),
            uses_rotary=any(symbol in all_symbols for symbol in {"apply_rotary_emb", "precompute_freqs_cis", "rope_theta"}),
            uses_distributed=any(symbol in all_symbols for symbol in {"dist", "all_reduce", "broadcast", "world_size"}),
            uses_topk="topk" in all_symbols,
            parallel_linears=parallel_linears,
            block_components=block_components,
            transformer_components=transformer_components,
        )

    def _extract_self_assignments(self, tree: ast.AST, class_name: str) -> list[str]:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                components: list[str] = []
                for child in node.body:
                    if isinstance(child, ast.FunctionDef) and child.name == "__init__":
                        for stmt in ast.walk(child):
                            if isinstance(stmt, ast.Assign):
                                for target in stmt.targets:
                                    if (
                                        isinstance(target, ast.Attribute)
                                        and isinstance(target.value, ast.Name)
                                        and target.value.id == "self"
                                    ):
                                        components.append(target.attr)
                return sorted(set(components))
        return []
