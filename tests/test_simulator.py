import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from xpu_simulator.backends import AscendBackend, NvidiaBackend
from xpu_simulator.backends.base import load_hardware_config
from xpu_simulator.calibration import (
    build_calibration_report,
    load_backend_calibration,
    load_benchmark_rows,
    summarize_benchmark_rows,
)
from xpu_simulator.frontend import (
    BackendIrGraphBuilder,
    DeepSeekConfig,
    DeepSeekGraphBuilder,
    DeepSeekSourceAnalyzer,
    ModelConfig,
    SourceModelAnalyzer,
    TorchExportGraphBuilder,
    TorchFxGraphBuilder,
    TransformerSourceGraphBuilder,
)
from xpu_simulator.profiling import load_trace_events, summarize_trace_events
from xpu_simulator.reporting import compare_results, diff_graphs, format_summary, write_html_report
from xpu_simulator.ir.types import DType
from xpu_simulator.reporting import result_to_dict
from xpu_simulator.sim import Simulator


def make_config() -> DeepSeekConfig:
    return DeepSeekConfig(
        vocab_size=32000,
        dim=512,
        inter_dim=2048,
        moe_inter_dim=256,
        n_layers=4,
        n_dense_layers=1,
        n_heads=8,
        n_routed_experts=16,
        n_shared_experts=2,
        n_activated_experts=4,
        qk_rope_head_dim=64,
        v_head_dim=64,
        n_kv_heads=8,
        dtype=DType.BF16,
    )


class SimulatorScaffoldTests(unittest.TestCase):
    def test_source_analyzer_detects_deepseek_features(self) -> None:
        source_path = Path("/Users/ray/Documents/Codex/DeepSeek/DeepSeek-V3.2/inference/model.py")
        summary = DeepSeekSourceAnalyzer().analyze(source_path)
        self.assertTrue(summary.uses_fp8)
        self.assertTrue(summary.uses_rotary)
        self.assertTrue(summary.uses_distributed)
        self.assertIn("ColumnParallelLinear", summary.parallel_linears)
        self.assertIn("attn", summary.block_components)
        self.assertIn("layers", summary.transformer_components)

    def test_source_analyzer_handles_standard_transformer_naming(self) -> None:
        source_text = '''
class MLP:
    def __init__(self, config):
        self.gate_proj = Linear(config.hidden_size, config.intermediate_size)
        self.up_proj = Linear(config.hidden_size, config.intermediate_size)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size)

class SelfAttention:
    def __init__(self, config):
        self.q_proj = Linear(config.hidden_size, config.hidden_size)
        self.k_proj = Linear(config.hidden_size, config.hidden_size)
        self.v_proj = Linear(config.hidden_size, config.hidden_size)
        self.o_proj = Linear(config.hidden_size, config.hidden_size)

class DecoderLayer:
    def __init__(self, config):
        self.self_attn = SelfAttention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)

class ToyLM:
    def __init__(self, config):
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = ModuleList()
        self.layers.append(DecoderLayer(config))
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = Linear(config.hidden_size, config.vocab_size)
'''
        summary = SourceModelAnalyzer().analyze_text(source_text)
        architecture = summary.architecture
        self.assertEqual(architecture["model_class"], "ToyLM")
        self.assertEqual(architecture["block_class"], "DecoderLayer")
        self.assertEqual(architecture["attention_class"], "SelfAttention")
        self.assertEqual(architecture["dense_ffn_class"], "MLP")
        self.assertEqual(architecture["block_attention_attr"], "self_attn")
        self.assertEqual(architecture["block_ffn_attr"], "mlp")
        self.assertTrue(architecture["attention_traits"]["has_q_proj"])
        self.assertTrue(architecture["attention_traits"]["has_k_proj"])
        self.assertTrue(architecture["attention_traits"]["has_v_proj"])
        self.assertTrue(architecture["dense_ffn_traits"]["has_gate_branch"])

    def test_source_analyzer_detects_cross_attention_blocks(self) -> None:
        source_text = '''
class CrossAttention:
    def __init__(self, config):
        self.q_proj = Linear(config.hidden_size, config.hidden_size)
        self.k_proj = Linear(config.hidden_size, config.hidden_size)
        self.v_proj = Linear(config.hidden_size, config.hidden_size)
        self.o_proj = Linear(config.hidden_size, config.hidden_size)

class DecoderLayer:
    def __init__(self, config):
        self.self_attn = CrossAttention(config)
        self.encoder_attn = CrossAttention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size)
'''
        architecture = SourceModelAnalyzer().analyze_text(source_text).architecture
        self.assertEqual(architecture["block_cross_attention_attr"], "encoder_attn")

    def test_generic_frontend_aliases_match_deepseek_frontend(self) -> None:
        source_path = Path("/Users/ray/Documents/Codex/DeepSeek/DeepSeek-V3.2/inference/model.py")
        generic_summary = SourceModelAnalyzer().analyze(source_path)
        deepseek_summary = DeepSeekSourceAnalyzer().analyze(source_path)
        self.assertEqual(generic_summary.architecture, deepseek_summary.architecture)

        generic_config = ModelConfig.from_json(
            "/Users/ray/Documents/Codex/DeepSeek/DeepSeek-V3.2/inference/config_671B_v3.2.json"
        )
        graph = TransformerSourceGraphBuilder(generic_config, source_path=source_path).build_graph(
            batch_size=1,
            seq_len=16,
            layers=1,
        )
        self.assertGreater(graph.node_count(), 0)
        self.assertIn("architecture", graph.metadata)

    def test_source_analyzer_can_export_architecture_payload(self) -> None:
        source_text = '''
class MLP:
    def __init__(self, config):
        self.gate_proj = Linear(config.hidden_size, config.intermediate_size)
        self.up_proj = Linear(config.hidden_size, config.intermediate_size)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size)

class SelfAttention:
    def __init__(self, config):
        self.q_proj = Linear(config.hidden_size, config.hidden_size)
        self.k_proj = Linear(config.hidden_size, config.hidden_size)
        self.v_proj = Linear(config.hidden_size, config.hidden_size)
        self.o_proj = Linear(config.hidden_size, config.hidden_size)

class DecoderLayer:
    def __init__(self, config):
        self.self_attn = SelfAttention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size)
'''
        payload = SourceModelAnalyzer().export_architecture_text(source_text, config=make_config())
        self.assertIn("architecture", payload)
        self.assertIn("source_features", payload)
        self.assertIn("config", payload)
        self.assertEqual(payload["architecture"]["attention_class"], "SelfAttention")
        self.assertEqual(payload["config"]["n_heads"], 8)

    def test_generic_builder_uses_standard_projection_and_mlp_names(self) -> None:
        source_text = '''
class MLP:
    def __init__(self, config):
        self.gate_proj = Linear(config.hidden_size, config.intermediate_size)
        self.up_proj = Linear(config.hidden_size, config.intermediate_size)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size)

class SelfAttention:
    def __init__(self, config):
        self.q_proj = Linear(config.hidden_size, config.hidden_size)
        self.k_proj = Linear(config.hidden_size, config.hidden_size)
        self.v_proj = Linear(config.hidden_size, config.hidden_size)
        self.o_proj = Linear(config.hidden_size, config.hidden_size)

class DecoderLayer:
    def __init__(self, config):
        self.self_attn = SelfAttention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)

class ToyLM:
    def __init__(self, config):
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = ModuleList()
        self.layers.append(DecoderLayer(config))
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = Linear(config.hidden_size, config.vocab_size)
'''
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "toy_model.py"
            source_path.write_text(source_text)
            graph = TransformerSourceGraphBuilder(make_config(), source_path=source_path).build_graph(
                batch_size=1,
                seq_len=16,
                layers=1,
            )

        names = [node.name for node in graph.nodes]
        self.assertIn("layer_0_q_proj", names)
        self.assertIn("layer_0_k_proj", names)
        self.assertIn("layer_0_v_proj", names)
        self.assertIn("layer_0_attn_proj_merge", names)
        self.assertIn("layer_0_o_proj", names)
        self.assertIn("layer_0_up_proj", names)
        self.assertIn("layer_0_gate_proj", names)
        self.assertIn("layer_0_down_proj", names)
        self.assertIn("layer_0_q_proj", graph.predecessors("layer_0_attn_proj_merge"))
        self.assertIn("layer_0_k_proj", graph.predecessors("layer_0_attn_proj_merge"))
        self.assertIn("layer_0_v_proj", graph.predecessors("layer_0_attn_proj_merge"))

    def test_torch_fx_builder_traces_executable_model(self) -> None:
        source_text = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class ToyFxModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = torch.matmul(q, k.transpose(-1, -2))
        probs = F.softmax(scores, dim=-1)
        ctx = torch.matmul(probs, v)
        out = self.o_proj(ctx)
        out = self.norm(out)
        return self.head(out)
'''
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "toy_fx_model.py"
            source_path.write_text(source_text)
            graph = TorchFxGraphBuilder(make_config(), source_path=source_path, model_class="ToyFxModel").build_graph(
                batch_size=1,
                seq_len=8,
            )

        names = [node.name for node in graph.nodes]
        self.assertIn("embed", names)
        self.assertIn("q_proj", names)
        self.assertIn("softmax", names)
        self.assertIn("head", names)
        self.assertEqual(graph.metadata["frontend"], "torch_fx")
        self.assertGreater(graph.node_count(), 5)

    def test_generic_builder_adds_cross_attention_when_present(self) -> None:
        source_text = '''
class MLP:
    def __init__(self, config):
        self.gate_proj = Linear(config.hidden_size, config.intermediate_size)
        self.up_proj = Linear(config.hidden_size, config.intermediate_size)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size)

class CrossAttention:
    def __init__(self, config):
        self.q_proj = Linear(config.hidden_size, config.hidden_size)
        self.k_proj = Linear(config.hidden_size, config.hidden_size)
        self.v_proj = Linear(config.hidden_size, config.hidden_size)
        self.o_proj = Linear(config.hidden_size, config.hidden_size)

class DecoderLayer:
    def __init__(self, config):
        self.self_attn = CrossAttention(config)
        self.encoder_attn = CrossAttention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)

class Seq2SeqLM:
    def __init__(self, config):
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = ModuleList()
        self.layers.append(DecoderLayer(config))
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = Linear(config.hidden_size, config.vocab_size)
'''
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "seq2seq_model.py"
            source_path.write_text(source_text)
            graph = TransformerSourceGraphBuilder(make_config(), source_path=source_path).build_graph(
                batch_size=1,
                seq_len=16,
                layers=1,
            )
        names = [node.name for node in graph.nodes]
        self.assertIn("layer_0_cross_attn_scores", names)
        self.assertIn("layer_0_cross_attn_softmax", names)
        self.assertIn("layer_0_cross_attn_out", names)
        self.assertIn("layer_0_cross_attn_scores", graph.predecessors("layer_0_cross_attn_softmax"))
        self.assertIn("layer_0_cross_attn_softmax", graph.predecessors("layer_0_cross_attn_out"))

    def test_model_config_supports_generic_transformer_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                """
{
  "vocab_size": 50000,
  "hidden_size": 1024,
  "intermediate_size": 4096,
  "num_hidden_layers": 24,
  "num_attention_heads": 16,
  "num_key_value_heads": 4,
  "torch_dtype": "torch.float16"
}
"""
            )
            config = ModelConfig.from_json(config_path)

        self.assertEqual(config.dim, 1024)
        self.assertEqual(config.inter_dim, 4096)
        self.assertEqual(config.n_layers, 24)
        self.assertEqual(config.n_heads, 16)
        self.assertEqual(config.n_kv_heads, 4)
        self.assertEqual(config.dtype, DType.FP16)

    def test_torch_export_builder_exports_executable_model(self) -> None:
        source_text = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class ToyExportModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        x = self.proj(x)
        x = F.softmax(x, dim=-1)
        return self.head(x)
'''
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "toy_export_model.py"
            source_path.write_text(source_text)
            graph = TorchExportGraphBuilder(make_config(), source_path=source_path, model_class="ToyExportModel").build_graph(
                batch_size=1,
                seq_len=4,
            )
        self.assertEqual(graph.metadata["frontend"], "torch_export")
        self.assertGreater(graph.node_count(), 3)
        self.assertTrue(any(node.attrs.get("exported", False) for node in graph.nodes))

    def test_backend_ir_builder_loads_lowered_graph(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ir_path = Path(tmpdir) / "backend_ir.json"
            ir_path.write_text(
                """
{
  "name": "toy_backend_ir",
  "nodes": [
    {
      "name": "kernel_0",
      "op_kind": "matmul",
      "inputs": [{"shape": [1, 8, 16], "dtype": "bf16"}],
      "outputs": [{"shape": [1, 8, 16], "dtype": "bf16"}],
      "flops": 4096,
      "bytes_moved": 1024
    },
    {
      "name": "kernel_1",
      "op_kind": "softmax",
      "inputs": [{"shape": [1, 8, 16], "dtype": "bf16"}],
      "outputs": [{"shape": [1, 8, 16], "dtype": "bf16"}],
      "flops": 640,
      "bytes_moved": 512
    }
  ],
  "edges": [{"src": "kernel_0", "dst": "kernel_1"}]
}
"""
            )
            graph = BackendIrGraphBuilder(ir_path).build_graph()
        self.assertEqual(graph.metadata["frontend"], "backend_ir")
        self.assertEqual(graph.node_count(), 2)
        self.assertEqual(graph.edge_count(), 1)

    def test_graph_diff_reports_frontend_delta(self) -> None:
        source_path = Path("/Users/ray/Documents/Codex/DeepSeek/DeepSeek-V3.2/inference/model.py")
        ast_graph = TransformerSourceGraphBuilder(make_config(), source_path=source_path).build_graph(
            batch_size=1,
            seq_len=8,
            layers=1,
        )
        payload = diff_graphs(ast_graph, ast_graph)
        self.assertEqual(payload["node_delta"], 0)
        self.assertEqual(payload["edge_delta"], 0)

    def test_trace_ingestion_summarizes_profiler_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.json"
            trace_path.write_text(
                """
{
  "traceEvents": [
    {"name": "aten::matmul", "ph": "X", "dur": 12.5, "cat": "cpu_op"},
    {"name": "aten::matmul", "ph": "X", "dur": 10.0, "cat": "cpu_op"},
    {"name": "aten::softmax", "ph": "X", "dur": 4.0, "cat": "cpu_op"}
  ]
}
"""
            )
            events = load_trace_events(trace_path)
            summary = summarize_trace_events(events)
        self.assertEqual(summary["event_count"], 3)
        self.assertIn("aten::matmul", summary["ops"])
        self.assertGreater(summary["ops"]["aten::matmul"]["total_duration_us"], 20.0)

    def test_cli_can_export_architecture_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "architecture.json"
            subprocess.check_call(
                [
                    "python3",
                    "-m",
                    "xpu_simulator.cli.main",
                    "--model-config",
                    "/Users/ray/Documents/Codex/DeepSeek/DeepSeek-V3.2/inference/config_671B_v3.2.json",
                    "--model-source",
                    "/Users/ray/Documents/Codex/DeepSeek/DeepSeek-V3.2/inference/model.py",
                    "--export-architecture",
                    str(out_path),
                ],
                cwd="/Users/ray/Documents/Repo/xPU-simulatorB",
                env={"PYTHONPATH": "src"},
            )
            payload = json.loads(out_path.read_text())
        self.assertIn("architecture", payload)
        self.assertIn("source_features", payload)
        self.assertIn("config", payload)
        self.assertEqual(payload["architecture"]["model_class"], "Transformer")

    def test_calibration_report_produces_recommendations(self) -> None:
        rows = [
            load_benchmark_rows(Path(tempfile.mkdtemp()) / "missing.csv") if False else None
        ]
        del rows
        benchmark_rows = [
            type("Row", (), {
                "op_name": "softmax",
                "measured_time_us": 20.0,
                "predicted_time_us": 10.0,
                "bytes_moved": 1024.0,
                "flops": 512.0,
            })(),
            type("Row", (), {
                "op_name": "softmax",
                "measured_time_us": 10.0,
                "predicted_time_us": 10.0,
                "bytes_moved": 1024.0,
                "flops": 512.0,
            })(),
        ]
        report = build_calibration_report(benchmark_rows)
        self.assertIn("softmax", report["recommendations"])
        self.assertGreater(report["recommendations"]["softmax"]["error_ratio"], 1.0)

    def test_gqa_graph_uses_smaller_kv_projections(self) -> None:
        source_text = '''
class SelfAttention:
    def __init__(self, config):
        self.q_proj = Linear(config.hidden_size, config.hidden_size)
        self.k_proj = Linear(config.hidden_size, config.hidden_size)
        self.v_proj = Linear(config.hidden_size, config.hidden_size)
        self.o_proj = Linear(config.hidden_size, config.hidden_size)

class MLP:
    def __init__(self, config):
        self.gate_proj = Linear(config.hidden_size, config.intermediate_size)
        self.up_proj = Linear(config.hidden_size, config.intermediate_size)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size)

class DecoderLayer:
    def __init__(self, config):
        self.self_attn = SelfAttention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)

class ToyLM:
    def __init__(self, config):
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = ModuleList()
        self.layers.append(DecoderLayer(config))
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = Linear(config.hidden_size, config.vocab_size)
'''
        gqa_config = DeepSeekConfig(
            vocab_size=32000,
            dim=512,
            inter_dim=2048,
            moe_inter_dim=256,
            n_layers=1,
            n_dense_layers=1,
            n_heads=8,
            n_routed_experts=0,
            n_shared_experts=0,
            n_activated_experts=0,
            qk_rope_head_dim=64,
            v_head_dim=64,
            n_kv_heads=2,
            dtype=DType.BF16,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "toy_model.py"
            source_path.write_text(source_text)
            graph = TransformerSourceGraphBuilder(gqa_config, source_path=source_path).build_graph(
                batch_size=1,
                seq_len=16,
                layers=1,
            )

        q_proj = next(node for node in graph.nodes if node.name == "layer_0_q_proj")
        k_proj = next(node for node in graph.nodes if node.name == "layer_0_k_proj")
        attn_scores = next(node for node in graph.nodes if node.name == "layer_0_attn_scores")
        self.assertGreater(q_proj.outputs[0].size_bytes, k_proj.outputs[0].size_bytes)
        self.assertEqual(q_proj.attrs["attention_variant"], "gqa")
        self.assertEqual(q_proj.attrs["num_kv_heads"], 2)
        self.assertEqual(attn_scores.attrs["attention_variant"], "gqa")

    def test_graph_builder_creates_transformer_structure(self) -> None:
        graph = DeepSeekGraphBuilder(make_config()).build_graph(batch_size=1, seq_len=32, layers=2)
        names = [node.name for node in graph.nodes]
        self.assertIn("embedding", names)
        self.assertIn("layer_0_qkv_proj", names)
        self.assertIn("layer_1_router", names)
        self.assertIn("lm_head", names)
        self.assertGreater(graph.node_count(), 10)
        self.assertIn("op_histogram", graph.metadata)
        self.assertGreater(graph.edge_count(), graph.node_count() - 1)
        self.assertGreater(graph.metadata["branch_edge_count"], 0)

    def test_simulator_runs_on_both_backends(self) -> None:
        graph = DeepSeekGraphBuilder(make_config()).build_graph(batch_size=1, seq_len=16, layers=1)
        nvidia_result = Simulator().simulate(graph, NvidiaBackend())
        ascend_result = Simulator().simulate(graph, AscendBackend())
        self.assertGreater(nvidia_result.total_latency_us, 0)
        self.assertGreater(ascend_result.total_latency_us, 0)
        self.assertEqual(nvidia_result.kernel_count, graph.node_count())
        self.assertEqual(ascend_result.kernel_count, graph.node_count())
        self.assertGreaterEqual(nvidia_result.total_latency_us, max(item.total_time_us for item in nvidia_result.kernel_estimates))
        self.assertTrue(nvidia_result.critical_path)
        self.assertGreaterEqual(
            nvidia_result.kernel_estimates[-1].end_time_us,
            nvidia_result.kernel_estimates[0].end_time_us,
        )
        self.assertIsNotNone(nvidia_result.memory_summary)
        self.assertGreater(nvidia_result.memory_summary.peak_live_bytes, 0)

    def test_reporting_payload_contains_summary(self) -> None:
        source_path = Path("/Users/ray/Documents/Codex/DeepSeek/DeepSeek-V3.2/inference/model.py")
        graph = DeepSeekGraphBuilder(make_config(), source_path=source_path).build_graph(batch_size=2, seq_len=8, layers=1)
        result = Simulator().simulate(graph, NvidiaBackend())
        payload = result_to_dict(graph, result)
        self.assertEqual(payload["backend"], "nvidia")
        self.assertEqual(payload["graph"]["nodes"], graph.node_count())
        self.assertEqual(payload["total_latency_us"], result.total_latency_us)
        self.assertIn("source_features", payload["graph"]["metadata"])
        self.assertIn("critical_path", payload)
        self.assertIn("memory", payload)
        self.assertIn("breakdown", payload)
        self.assertIn("by_family", payload["breakdown"])
        self.assertGreater(payload["memory"]["peak_live_bytes"], 0)

    def test_text_summary_contains_breakdown_table(self) -> None:
        graph = DeepSeekGraphBuilder(make_config()).build_graph(batch_size=1, seq_len=8, layers=1)
        result = Simulator().simulate(graph, NvidiaBackend())
        summary = format_summary(graph, result)
        self.assertIn("Breakdown:", summary)
        self.assertIn("Top families", summary)

    def test_html_report_writer_generates_layer_annotated_report(self) -> None:
        graph = DeepSeekGraphBuilder(make_config()).build_graph(batch_size=1, seq_len=8, layers=1)
        result = Simulator().simulate(graph, AscendBackend())
        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / "report.html"
            write_html_report(graph, result, html_path)
            html = html_path.read_text()
        self.assertIn(graph.name, html)
        self.assertIn("Architecture Timeline", html)
        self.assertIn("layer_0", html)
        self.assertIn("layer_0_attn_scores", html)

    def test_layer_start_can_target_moe_layers(self) -> None:
        graph = DeepSeekGraphBuilder(make_config()).build_graph(batch_size=1, seq_len=8, layers=2, layer_start=2)
        names = [node.name for node in graph.nodes]
        self.assertIn("layer_2_router", names)
        self.assertIn("layer_3_shared_ffn_down", names)
        self.assertGreater(graph.metadata["moe_node_count"], 0)

    def test_memory_analysis_tracks_events(self) -> None:
        graph = DeepSeekGraphBuilder(make_config()).build_graph(batch_size=1, seq_len=8, layers=1)
        result = Simulator().simulate(graph, NvidiaBackend())
        self.assertEqual(len(result.memory_summary.events), graph.node_count())
        self.assertGreaterEqual(result.memory_summary.peak_live_bytes, result.memory_summary.final_live_bytes)

    def test_device_config_loader_and_compare_payload(self) -> None:
        repo_root = Path("/Users/ray/Documents/Repo/xPU-simulatorB")
        nvidia_hw = load_hardware_config(repo_root / "configs/devices/nvidia_a100.json")
        ascend_hw = load_hardware_config(repo_root / "configs/devices/ascend_910b.json")
        nvidia_cal = load_backend_calibration(repo_root / "configs/calibration/nvidia_default.json")
        ascend_cal = load_backend_calibration(repo_root / "configs/calibration/ascend_default.json")
        graph = DeepSeekGraphBuilder(make_config()).build_graph(batch_size=1, seq_len=8, layers=1)
        results = {
            "nvidia": Simulator().simulate(graph, NvidiaBackend(hardware=nvidia_hw, calibration=nvidia_cal)),
            "ascend": Simulator().simulate(graph, AscendBackend(hardware=ascend_hw, calibration=ascend_cal)),
        }
        payload = compare_results(graph, results)
        self.assertIn("fastest_backend", payload)
        self.assertIn("latencies_us", payload)
        self.assertEqual(set(payload["latencies_us"].keys()), {"nvidia", "ascend"})

    def test_softmax_now_has_compute_component(self) -> None:
        graph = DeepSeekGraphBuilder(make_config()).build_graph(batch_size=1, seq_len=32, layers=1)
        result = Simulator().simulate(graph, AscendBackend())
        softmax = next(item for item in result.kernel_estimates if item.task_name == "layer_0_softmax")
        self.assertGreater(softmax.compute_time_us, 0.0)
        self.assertEqual(softmax.resource, "memory")

    def test_csv_benchmark_ingestion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "bench.csv"
            with csv_path.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["op_name", "measured_time_us", "predicted_time_us", "bytes_moved", "flops"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "op_name": "softmax",
                        "measured_time_us": 12.0,
                        "predicted_time_us": 10.0,
                        "bytes_moved": 2048,
                        "flops": 1024,
                    }
                )
                writer.writerow(
                    {
                        "op_name": "softmax",
                        "measured_time_us": 18.0,
                        "predicted_time_us": 15.0,
                        "bytes_moved": 4096,
                        "flops": 2048,
                    }
                )
            rows = load_benchmark_rows(csv_path)
            summary = summarize_benchmark_rows(rows)
            self.assertEqual(len(rows), 2)
            self.assertIn("softmax", summary)
            self.assertGreater(summary["softmax"]["error_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
