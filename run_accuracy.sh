#!/usr/bin/env bash
# Accuracy checks for the paper (about 1.5-2 hours on one H100):
#   1. vLLM, all six models: dense vs hybrid on the same real SWE-agent replies and on GSM8K
#   2. SGLang, both hybrids: 32-bit (default) vs 16-bit recurrent state
# Needs the GPU to itself: start it after the retention run has finished.
# Run inside tmux, from the vLLM environment:  tmux new -s acc ; bash run_accuracy.sh
# Set STEPS="2" to run only one part. Finished pieces are skipped on restart.
set -uo pipefail
cd "$(dirname "$0")"
export HF_HOME=${HF_HOME:-/workspace/hf}
export SGLANG_ENABLE_JIT_DEEPGEMM=${SGLANG_ENABLE_JIT_DEEPGEMM:-0}
STEPS=${STEPS:-"1 2"}
OUT=results/accuracy
TRACE=traces/swe_smith_1500.json
mkdir -p "$OUT" results/logs traces
has() { case " $STEPS " in *" $1 "*) return 0;; *) return 1;; esac; }
BF16='{"mamba_ssm_dtype":"bfloat16"}'
MODELS=(
  "granite_1b_dense|ibm-granite/granite-4.0-1b"
  "granite_h1b|ibm-granite/granite-4.0-h-1b"
  "granite_micro_dense|ibm-granite/granite-4.0-micro"
  "granite_hmicro|ibm-granite/granite-4.0-h-micro"
  "devstral24b_dense|mistralai/Devstral-Small-2-24B-Instruct-2512"
  "nemo3_nano30b|nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
)
acc() {  # engine name variant args hf
  if [ -s "$OUT/${2}_${3}_summary.json" ]; then echo "[skip] $2 / $3"; return; fi
  echo; echo "======== accuracy: $2 / $3 ($1) ========"
  pkill -f "vllm serve" 2>/dev/null; pkill -f EngineCore 2>/dev/null; pkill -f "sglang.launch_server|sglang::" 2>/dev/null; sleep 8
  python accuracy_bench.py --engine "$1" --model "$5" --tag "$2" --variant "$3" --engine-args "$4" \
    --trace "$TRACE" --out-dir "$OUT" 2>&1 | tee "results/logs/acc_${2}_${3}.log" | grep -E "^\[|Error|error" || true
}

python -c "import pandas, pyarrow, huggingface_hub, transformers" 2>/dev/null || pip install -q pandas pyarrow huggingface_hub transformers
[ -s "$TRACE" ] || python prepare_traces.py --n 1500 --out "$TRACE" | tee results/logs/prepare_1500.log
python -c "from huggingface_hub import hf_hub_download as d; d('openai/gsm8k','main/test-00000-of-00001.parquet',repo_type='dataset')" \
  >/dev/null 2>&1 || echo "[warn] could not download GSM8K"

if has 1; then
  if ! python -c "from vllm import LLM" 2>results/logs/vllm_check.log; then
    echo "ERROR: vLLM does not import with $(which python):"; tail -5 results/logs/vllm_check.log; exit 1
  fi
  for m in "${MODELS[@]}"; do IFS='|' read -r name hf <<< "$m"; acc vllm "$name" vllm '{}' "$hf"; done
  pkill -f "vllm serve" 2>/dev/null; pkill -f EngineCore 2>/dev/null
fi

if has 2; then
  if [ ! -f /workspace/sgl/bin/activate ]; then echo "ERROR: no SGLang environment in /workspace/sgl"; exit 1; fi
  . /workspace/sgl/bin/activate
  python -c "import pandas, pyarrow, transformers" 2>/dev/null || pip install -q pandas pyarrow transformers
  acc sglang granite_hmicro sgl_fp32 '{}'    ibm-granite/granite-4.0-h-micro
  acc sglang granite_hmicro sgl_bf16 "$BF16" ibm-granite/granite-4.0-h-micro
  acc sglang nemo3_nano30b  sgl_fp32 '{}'    nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8
  acc sglang nemo3_nano30b  sgl_bf16 "$BF16" nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8
  pkill -f "sglang.launch_server|sglang::" 2>/dev/null
  deactivate
fi

python analyze_accuracy.py "$OUT" | tee "$OUT/summary.txt"
tar czf results_accuracy.tar.gz "$OUT"
echo "Done. Download results_accuracy.tar.gz"
