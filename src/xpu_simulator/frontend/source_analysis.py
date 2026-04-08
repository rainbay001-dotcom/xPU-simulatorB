"""Source analysis and architecture extraction using Python AST."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ClassStructure:
    name: str
    assignments: dict[str, str] = field(default_factory=dict)
    conditional_assignments: dict[str, list[str]] = field(default_factory=dict)
    append_calls: dict[str, str] = field(default_factory=dict)


@dataclass
class ArchitectureDescriptor:
    model_class: str | None
    block_class: str | None
    attention_class: str | None
    dense_ffn_class: str | None
    moe_ffn_class: str | None
    embedding_attr: str | None
    layers_attr: str | None
    norm_attr: str | None
    head_attr: str | None
    block_norm_attrs: list[str]
    attention_traits: dict[str, object]
    dense_ffn_traits: dict[str, object]
    moe_ffn_traits: dict[str, object]

    def to_metadata(self) -> dict[str, object]:
        return {
            "model_class": self.model_class,
            "block_class": self.block_class,
            "attention_class": self.attention_class,
            "dense_ffn_class": self.dense_ffn_class,
            "moe_ffn_class": self.moe_ffn_class,
            "embedding_attr": self.embedding_attr,
            "layers_attr": self.layers_attr,
            "norm_attr": self.norm_attr,
            "head_attr": self.head_attr,
            "block_norm_attrs": self.block_norm_attrs,
            "attention_traits": self.attention_traits,
            "dense_ffn_traits": self.dense_ffn_traits,
            "moe_ffn_traits": self.moe_ffn_traits,
        }


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
    architecture: dict[str, object]

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
            "architecture": self.architecture,
        }


class DeepSeekSourceAnalyzer:
    def analyze(self, source_path: str | Path) -> SourceFeatureSummary:
        path = Path(source_path)
        tree = ast.parse(path.read_text())
        architecture = self._extract_architecture_from_tree(tree)

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

        return SourceFeatureSummary(
            classes=classes,
            functions=functions,
            uses_fp8=any(symbol in all_symbols for symbol in {"fp8_gemm", "act_quant", "float8_e4m3fn", "fp8_index"}),
            uses_moe=any(symbol in all_symbols for symbol in {"Gate", "MoE", "topk", "n_routed_experts"}),
            uses_rotary=any(symbol in all_symbols for symbol in {"apply_rotary_emb", "precompute_freqs_cis", "rope_theta"}),
            uses_distributed=any(symbol in all_symbols for symbol in {"dist", "all_reduce", "broadcast", "world_size"}),
            uses_topk="topk" in all_symbols,
            parallel_linears=parallel_linears,
            block_components=architecture.block_norm_attrs + [name for name in ["attn", "ffn"] if name],
            transformer_components=[
                value
                for value in [architecture.embedding_attr, architecture.layers_attr, architecture.norm_attr, architecture.head_attr]
                if value is not None
            ],
            architecture=architecture.to_metadata(),
        )

    def extract_architecture(self, source_path: str | Path) -> ArchitectureDescriptor:
        path = Path(source_path)
        tree = ast.parse(path.read_text())
        return self._extract_architecture_from_tree(tree)

    def _extract_architecture_from_tree(self, tree: ast.AST) -> ArchitectureDescriptor:
        structures = self._collect_class_structures(tree)
        model_class = self._infer_model_class(structures)
        model_structure = structures.get(model_class) if model_class else None
        layers_attr = self._infer_layers_attr(model_structure)
        block_class = None
        if model_structure and layers_attr:
            block_class = model_structure.append_calls.get(layers_attr)
        if block_class is None and "Block" in structures:
            block_class = "Block"
        block_structure = structures.get(block_class) if block_class else None

        attention_class = block_structure.assignments.get("attn") if block_structure else None
        ffn_variants = block_structure.conditional_assignments.get("ffn", []) if block_structure else []
        dense_ffn_class, moe_ffn_class = self._infer_ffn_variants(ffn_variants, structures)
        if dense_ffn_class is None and block_structure:
            assigned_ffn = block_structure.assignments.get("ffn")
            if assigned_ffn and self._class_traits(structures.get(assigned_ffn)).get("is_moe", False):
                moe_ffn_class = assigned_ffn
            else:
                dense_ffn_class = assigned_ffn

        embedding_attr = self._find_attr_by_constructor(model_structure, {"ParallelEmbedding", "Embedding"})
        norm_attr = self._find_attr_like(model_structure, {"norm"})
        head_attr = self._find_attr_by_constructor(
            model_structure,
            {"ColumnParallelLinear", "Linear", "RowParallelLinear"},
            preferred={"head", "lm_head"},
        )
        block_norm_attrs = [
            attr for attr in (block_structure.assignments.keys() if block_structure else [])
            if "norm" in attr
        ]

        return ArchitectureDescriptor(
            model_class=model_class,
            block_class=block_class,
            attention_class=attention_class,
            dense_ffn_class=dense_ffn_class,
            moe_ffn_class=moe_ffn_class,
            embedding_attr=embedding_attr,
            layers_attr=layers_attr,
            norm_attr=norm_attr,
            head_attr=head_attr,
            block_norm_attrs=sorted(block_norm_attrs),
            attention_traits=self._attention_traits(structures.get(attention_class)),
            dense_ffn_traits=self._class_traits(structures.get(dense_ffn_class)),
            moe_ffn_traits=self._class_traits(structures.get(moe_ffn_class)),
        )

    def _collect_class_structures(self, tree: ast.AST) -> dict[str, ClassStructure]:
        structures: dict[str, ClassStructure] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            structure = ClassStructure(name=node.name)
            init_fn = next(
                (child for child in node.body if isinstance(child, ast.FunctionDef) and child.name == "__init__"),
                None,
            )
            if init_fn is None:
                structures[node.name] = structure
                continue
            for stmt in ast.walk(init_fn):
                if isinstance(stmt, ast.Assign):
                    target = self._self_attr_target(stmt.targets)
                    if target is None:
                        continue
                    call_name = self._call_name(stmt.value)
                    if isinstance(stmt.value, ast.IfExp):
                        variants = [name for name in [self._call_name(stmt.value.body), self._call_name(stmt.value.orelse)] if name]
                        if variants:
                            structure.conditional_assignments[target] = variants
                    elif call_name:
                        structure.assignments[target] = call_name
                elif isinstance(stmt, ast.Expr):
                    self._maybe_record_append_call(structure, stmt.value)
            structures[node.name] = structure
        return structures

    def _self_attr_target(self, targets: list[ast.expr]) -> str | None:
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                return target.attr
        return None

    def _maybe_record_append_call(self, structure: ClassStructure, expr: ast.AST) -> None:
        if not isinstance(expr, ast.Call):
            return
        if not isinstance(expr.func, ast.Attribute) or expr.func.attr != "append":
            return
        owner = expr.func.value
        if not (
            isinstance(owner, ast.Attribute)
            and isinstance(owner.value, ast.Name)
            and owner.value.id == "self"
        ):
            return
        if not expr.args:
            return
        call_name = self._call_name(expr.args[0])
        if call_name:
            structure.append_calls[owner.attr] = call_name

    def _call_name(self, node: ast.AST | None) -> str | None:
        if node is None:
            return None
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                return func.id
            if isinstance(func, ast.Attribute):
                return func.attr
        if isinstance(node, ast.Name):
            return node.id
        return None

    def _infer_model_class(self, structures: dict[str, ClassStructure]) -> str | None:
        if "Transformer" in structures:
            return "Transformer"
        for name, structure in structures.items():
            if "layers" in structure.assignments or "layers" in structure.append_calls:
                return name
        return None

    def _infer_layers_attr(self, structure: ClassStructure | None) -> str | None:
        if structure is None:
            return None
        if "layers" in structure.append_calls:
            return "layers"
        for attr, ctor in structure.assignments.items():
            if ctor == "ModuleList" or "layer" in attr:
                return attr
        return None

    def _infer_ffn_variants(
        self,
        variants: list[str],
        structures: dict[str, ClassStructure],
    ) -> tuple[str | None, str | None]:
        dense = None
        moe = None
        for variant in variants:
            traits = self._class_traits(structures.get(variant))
            if traits.get("is_moe", False):
                moe = variant
            else:
                dense = variant
        return dense, moe

    def _find_attr_by_constructor(
        self,
        structure: ClassStructure | None,
        constructors: set[str],
        preferred: set[str] | None = None,
    ) -> str | None:
        if structure is None:
            return None
        preferred = preferred or set()
        for attr, ctor in structure.assignments.items():
            if attr in preferred and ctor in constructors:
                return attr
        for attr, ctor in structure.assignments.items():
            if ctor in constructors:
                return attr
        return None

    def _find_attr_like(self, structure: ClassStructure | None, fragments: set[str]) -> str | None:
        if structure is None:
            return None
        for attr in structure.assignments:
            if any(fragment in attr for fragment in fragments):
                return attr
        return None

    def _attention_traits(self, structure: ClassStructure | None) -> dict[str, object]:
        if structure is None:
            return {}
        projection_attrs = [
            attr for attr in structure.assignments
            if attr.startswith("w") and attr not in {"weight"}
        ]
        return {
            "has_q_norm": "q_norm" in structure.assignments,
            "has_kv_norm": "kv_norm" in structure.assignments,
            "has_output_proj": any(attr in structure.assignments for attr in {"wo", "out_proj", "o_proj"}),
            "has_indexer": "indexer" in structure.assignments,
            "projection_attrs": projection_attrs,
            "projection_count": len(projection_attrs),
        }

    def _class_traits(self, structure: ClassStructure | None) -> dict[str, object]:
        if structure is None:
            return {}
        attrs = set(structure.assignments)
        return {
            "is_moe": bool({"gate", "experts", "shared_experts"} & attrs),
            "has_gate_branch": {"w1", "w2", "w3"} <= attrs,
            "assignments": sorted(attrs),
        }
