# xPU-simulatorB

Modular simulator scaffold for estimating model execution on NVIDIA GPUs and Huawei Ascend NPUs.

It currently supports:
- source-driven transformer/LLM frontend
- optional `torch.fx` and `torch.export` frontends
- backend/compiler-IR ingestion
- dual backends for NVIDIA and Ascend
- kernel-level cost modeling
- DAG-aware scheduling
- peak live-memory analysis
- KV-cache accounting for `prefill` and `decode`
- JSON, text, and HTML reports

## Requirements

- Python 3.9+
- Run commands from the repo root:
  - `/Users/ray/Documents/Repo/xPU-simulatorB`
- Set `PYTHONPATH=src` for CLI usage

No GPU or NPU is required for the default AST/source frontend. `torch.fx` and `torch.export` require a runnable PyTorch model.

## Repo Layout

- `src/xpu_simulator/frontend/`: graph builders and architecture extraction
- `src/xpu_simulator/backends/`: NVIDIA and Ascend cost models
- `src/xpu_simulator/sim/`: scheduler and simulation engine
- `src/xpu_simulator/memory/`: liveness and KV-cache analysis
- `src/xpu_simulator/reporting/`: text, JSON, HTML, and breakdown rendering
- `configs/devices/`: hardware parameter packs
- `configs/calibration/`: backend calibration packs
- `tests/`: unit and integration-style tests

## Frontends

### `transformer_source`

Default frontend. Parses Python source with AST, extracts architecture structure, then builds a simulator graph.

Use this when:
- the model source is available
- the model is hard to execute or trace
- you want a stable architecture-level graph

Requires:
- `--model-config`
- usually `--model-source`

### `transformer_fx`

Builds a graph from `torch.fx` tracing.

Use this when:
- the model can be imported and run on CPU
- you want a graph closer to executable PyTorch behavior

Requires:
- `--model-config`
- `--model-source`
- `--model-class`

### `transformer_export`

Builds a graph from `torch.export`.

Use this when:
- the model exports cleanly
- you want a more normalized PyTorch graph

Requires:
- `--model-config`
- `--model-source`
- `--model-class`

### `backend_ir`

Loads a lowered graph from a JSON file.

Use this when:
- you already have compiler/backend IR in JSON form
- you want to bypass source extraction

Requires:
- `--model-config`
- `--backend-ir`

## Execution Modes

### `prefill`

Default mode. Simulates prompt processing over `seq_len` tokens.

Example:
- `--mode prefill --seq-len 4096`

### `decode`

Simulates one-token generation on top of an existing KV cache.

Use:
- `--mode decode`
- `--context-len <existing_context_tokens>`
- `--seq-len 1`

The simulator adds:
- `*_kv_cache_write`
- `*_kv_cache_read`
- KV-cache bytes per layer
- total KV-cache bytes for the simulated layer range

## Basic Usage

### Source frontend, single backend

```bash
cd /Users/ray/Documents/Repo/xPU-simulatorB

PYTHONPATH=src python3 -m xpu_simulator.cli.main \
  --model-config /Users/ray/Documents/Codex/DeepSeek/DeepSeek-V3.2/inference/config_671B_v3.2.json \
  --model-source /Users/ray/Documents/Codex/DeepSeek/DeepSeek-V3.2/inference/model.py \
  --model-family transformer_source \
  --backend nvidia \
  --layers 2
```

### Ascend run on MoE-heavy layers

```bash
PYTHONPATH=src python3 -m xpu_simulator.cli.main \
  --model-config /Users/ray/Documents/Codex/DeepSeek/DeepSeek-V3.2/inference/config_671B_v3.2.json \
  --model-source /Users/ray/Documents/Codex/DeepSeek/DeepSeek-V3.2/inference/model.py \
  --model-family transformer_source \
  --backend ascend \
  --layer-start 4 \
  --layers 2
```

### Full compare run: NVIDIA vs Ascend

```bash
PYTHONPATH=src python3 -m xpu_simulator.cli.main \
  --model-config /Users/ray/Documents/Codex/DeepSeek/DeepSeek-V3.2/inference/config_671B_v3.2.json \
  --model-source /Users/ray/Documents/Codex/DeepSeek/DeepSeek-V3.2/inference/model.py \
  --model-family transformer_source \
  --backend compare \
  --seq-len 128
```

### Decode-mode run with KV cache

```bash
PYTHONPATH=src python3 -m xpu_simulator.cli.main \
  --model-config /Users/ray/Documents/Codex/DeepSeek/DeepSeek-V3.2/inference/config_671B_v3.2.json \
  --model-source /Users/ray/Documents/Codex/DeepSeek/DeepSeek-V3.2/inference/model.py \
  --model-family transformer_source \
  --backend ascend \
  --mode decode \
  --context-len 128 \
  --seq-len 1 \
  --layers 1
```

## Output Modes

### Default text output

Default output includes:
- graph name
- backend/device
- mode, context, and step tokens
- node and edge counts
- total FLOPs and bytes
- estimated latency
- peak live memory
- KV-cache summary when applicable
- top kernels
- breakdown tables
- critical path preview

### JSON output

Use:

```bash
PYTHONPATH=src python3 -m xpu_simulator.cli.main \
  --model-config /path/to/config.json \
  --model-source /path/to/model.py \
  --backend nvidia \
  --json
```

JSON includes:
- graph metadata
- per-kernel timing
- critical path
- memory summary
- breakdown tables
- cache summary for decode runs
- HTML report path

To include the full memory event list:

```bash
--json --dump-memory-events
```

### HTML report

Every simulation run writes an HTML report automatically unless you override the path.

Default filename:
- `<graph_name>_<backend>.html`

Override:

```bash
--html-report /path/to/report.html
```

The HTML report includes:
- run summary
- architecture summary
- resource and kernel-family breakdowns
- architecture graph with per-kernel annotations
- NSight-style timeline lanes
- per-layer kernel tables
- KV-cache summary for decode runs

## Architecture Export

To export extracted model architecture JSON:

```bash
PYTHONPATH=src python3 -m xpu_simulator.cli.main \
  --model-config /Users/ray/Documents/Codex/DeepSeek/DeepSeek-V3.2/inference/config_671B_v3.2.json \
  --model-source /Users/ray/Documents/Codex/DeepSeek/DeepSeek-V3.2/inference/model.py \
  --export-architecture /tmp/deepseek_architecture.json
```

To print it to stdout:

```bash
--export-architecture -
```

## Frontend Comparison

To compare the AST/source frontend against `fx` or `export`:

```bash
PYTHONPATH=src python3 -m xpu_simulator.cli.main \
  --model-config /path/to/config.json \
  --model-source /path/to/model.py \
  --model-class MyModel \
  --compare-frontends \
  --compare-target fx
```

Targets:
- `fx`
- `export`

## FX and Export Notes

`torch.fx` and `torch.export` need:
- importable Python model code
- a runnable model constructor
- CPU-safe execution for tracing/export

They may fail when the model depends on:
- custom CUDA-only ops
- custom Ascend-only ops
- missing runtime modules
- distributed-only initialization

In those cases, use `transformer_source` instead.

## Calibration and Trace Utilities

### Benchmark CSV summary

```bash
PYTHONPATH=src python3 -m xpu_simulator.cli.main \
  --model-config /path/to/config.json \
  --benchmark-csv /path/to/bench.csv
```

### Calibration recommendation report

```bash
PYTHONPATH=src python3 -m xpu_simulator.cli.main \
  --model-config /path/to/config.json \
  --benchmark-csv /path/to/bench.csv \
  --calibration-report
```

### Profiler trace summary

```bash
PYTHONPATH=src python3 -m xpu_simulator.cli.main \
  --model-config /path/to/config.json \
  --trace-json /path/to/trace.json
```

## Device and Calibration Overrides

Single-backend run:

```bash
PYTHONPATH=src python3 -m xpu_simulator.cli.main \
  --model-config /path/to/config.json \
  --model-source /path/to/model.py \
  --backend ascend \
  --device-config /Users/ray/Documents/Repo/xPU-simulatorB/configs/devices/ascend_910b.json \
  --calibration-config /Users/ray/Documents/Repo/xPU-simulatorB/configs/calibration/ascend_default.json
```

Compare mode supports separate overrides:
- `--nvidia-device-config`
- `--nvidia-calibration-config`
- `--ascend-device-config`
- `--ascend-calibration-config`

## Testing

Run the test suite from repo root:

```bash
cd /Users/ray/Documents/Repo/xPU-simulatorB
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Current Limitations

- kernel models are still coarse analytic approximations
- accuracy still depends on calibration quality
- AST/source frontend builds a simulator-side graph, not a true runtime trace
- `fx` and `export` are only as good as the model’s ability to run on CPU
- multi-device and high-fidelity distributed simulation are still limited

## Recommended Workflow

For most model bring-up:

1. Start with `transformer_source`
2. Export architecture JSON
3. Run `prefill` on a small layer slice
4. Run `decode` on a small layer slice
5. Inspect HTML report
6. If the model is runnable, compare AST vs `fx` or `export`
7. Add calibration data once measured traces are available
