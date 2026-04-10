#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_CONFIG="${MODEL_CONFIG:-/Users/ray/Documents/Codex/DeepSeek/DeepSeek-V3.2/inference/config_671B_v3.2.json}"
MODEL_SOURCE="${MODEL_SOURCE:-/Users/ray/Documents/Codex/DeepSeek/DeepSeek-V3.2/inference/model.py}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 {prefill-ascend|decode-ascend|compare|all}"
  echo "env overrides: MODEL_CONFIG=... MODEL_SOURCE=... PYTHON_BIN=..."
  exit 1
fi

run_cli() {
  (
    cd "${REPO_ROOT}"
    PYTHONPATH=src "${PYTHON_BIN}" -m xpu_simulator.cli.main "$@"
  )
}

ensure_inputs() {
  if [[ ! -f "${MODEL_CONFIG}" ]]; then
    echo "missing model config: ${MODEL_CONFIG}" >&2
    exit 1
  fi
  if [[ ! -f "${MODEL_SOURCE}" ]]; then
    echo "missing model source: ${MODEL_SOURCE}" >&2
    exit 1
  fi
}

run_prefill_ascend() {
  run_cli \
    --model-config "${MODEL_CONFIG}" \
    --model-source "${MODEL_SOURCE}" \
    --backend ascend \
    --seq-len 32 \
    --layers 1 \
    --enable-fusion
}

run_decode_ascend() {
  run_cli \
    --model-config "${MODEL_CONFIG}" \
    --model-source "${MODEL_SOURCE}" \
    --backend ascend \
    --mode decode \
    --context-len 128 \
    --seq-len 1 \
    --layers 1
}

run_compare() {
  run_cli \
    --model-config "${MODEL_CONFIG}" \
    --model-source "${MODEL_SOURCE}" \
    --backend compare \
    --seq-len 128 \
    --html-report "${REPO_ROOT}/deepseek_v32_compare_b1_s128.html"
}

ensure_inputs

case "$1" in
  prefill-ascend)
    run_prefill_ascend
    ;;
  decode-ascend)
    run_decode_ascend
    ;;
  compare)
    run_compare
    ;;
  all)
    run_prefill_ascend
    run_decode_ascend
    run_compare
    ;;
  *)
    echo "unknown target: $1" >&2
    exit 1
    ;;
esac
