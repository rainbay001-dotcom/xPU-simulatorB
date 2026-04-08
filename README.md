# xPU-simulatorB

Modular simulator scaffold for estimating model execution on NVIDIA GPUs and
Huawei Ascend NPUs.

Current scope:
- common graph IR
- backend interface
- simple kernel-level cost model
- DAG-aware scheduling with basic compute / communication overlap
- peak live-memory analysis
- DeepSeek-oriented frontend that builds an abstract graph from config
- file-backed device parameter packs
- file-backed backend calibration packs
- cross-backend comparison mode

Quick start:

```bash
python -m xpu_simulator.cli.main \
  --model-config /Users/ray/Documents/Codex/DeepSeek/DeepSeek-V3.2/inference/config_671B_v3.2.json \
  --backend nvidia \
  --layers 2
```

To target MoE-heavy layers later in the stack:

```bash
PYTHONPATH=src python3 -m xpu_simulator.cli.main \
  --model-config /Users/ray/Documents/Codex/DeepSeek/DeepSeek-V3.2/inference/config_671B_v3.2.json \
  --model-source /Users/ray/Documents/Codex/DeepSeek/DeepSeek-V3.2/inference/model.py \
  --backend ascend \
  --layer-start 4 \
  --layers 2
```

JSON output without the full memory timeline:

```bash
PYTHONPATH=src python3 -m xpu_simulator.cli.main \
  --model-config /Users/ray/Documents/Codex/DeepSeek/DeepSeek-V3.2/inference/config_671B_v3.2.json \
  --model-source /Users/ray/Documents/Codex/DeepSeek/DeepSeek-V3.2/inference/model.py \
  --backend nvidia \
  --layers 2 \
  --json
```

Compare NVIDIA and Ascend on the same DeepSeek slice:

```bash
PYTHONPATH=src python3 -m xpu_simulator.cli.main \
  --model-config /Users/ray/Documents/Codex/DeepSeek/DeepSeek-V3.2/inference/config_671B_v3.2.json \
  --model-source /Users/ray/Documents/Codex/DeepSeek/DeepSeek-V3.2/inference/model.py \
  --backend compare \
  --layer-start 4 \
  --layers 2
```

Override backend calibration:

```bash
PYTHONPATH=src python3 -m xpu_simulator.cli.main \
  --model-config /Users/ray/Documents/Codex/DeepSeek/DeepSeek-V3.2/inference/config_671B_v3.2.json \
  --model-source /Users/ray/Documents/Codex/DeepSeek/DeepSeek-V3.2/inference/model.py \
  --backend ascend \
  --calibration-config /Users/ray/Documents/Repo/xPU-simulatorB/configs/calibration/ascend_default.json \
  --layer-start 4 \
  --layers 1
```
