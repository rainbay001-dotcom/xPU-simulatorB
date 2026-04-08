import csv
import tempfile
import unittest
from pathlib import Path

from xpu_simulator.backends import AscendBackend, NvidiaBackend
from xpu_simulator.backends.base import load_hardware_config
from xpu_simulator.calibration import load_backend_calibration, load_benchmark_rows, summarize_benchmark_rows
from xpu_simulator.frontend import DeepSeekConfig, DeepSeekGraphBuilder, DeepSeekSourceAnalyzer
from xpu_simulator.reporting import compare_results
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
        self.assertGreater(payload["memory"]["peak_live_bytes"], 0)

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
