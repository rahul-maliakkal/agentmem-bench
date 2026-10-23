#!/usr/bin/env bash
# Experiment 8 (how many paused sessions keep their resume point) on vLLM and SGLang,
# plus the Devstral 256-agent concurrency point rerun with a longer time limit.
# Run inside tmux:  tmux new -s ckpt ; bash run_checkpoint.sh      (about 2 hours)
set -uo pipefail
cd "$(dirname "$0")"
export HF_HOME=${HF_HOME:-/workspace/hf}
mkdir -p results/logs
FINE='{"mamba_cache_mode":"align","prefix_match_unit":16}'

run() {  # name variant engine engine_args hf
  local out="results/ret_${1}_${2}.csv"
  if [ -s "${out%.csv}_summary.json" ]; then echo "[skip] $1 / $2 already done"; return; fi
  echo; echo "======== retention: $1 / $2 ($3) ========"
  python retention_bench.py --engine "$3" --model "$5" --variant "$2" --engine-args "$4" --out "$out" \
    2>&1 | tee "results/logs/ret_${1}_${2}.log" | grep -E "^N=|\[info\]|\[warn\]|Error|error|wrote" || true
}

# ---- vLLM
python -c "from vllm import LLM" 2>/dev/null || { echo "ERROR: run from the vLLM environment (deactivate the SGLang venv)"; exit 1; }
pkill -f "vllm serve" 2>/dev/null; pkill -f EngineCore 2>/dev/null; sleep 3
run granite_micro_dense default vllm '{}'    ibm-granite/granite-4.0-micro
run granite_hmicro      default vllm '{}'    ibm-granite/granite-4.0-h-micro
run granite_hmicro      fine    vllm "$FINE" ibm-granite/granite-4.0-h-micro
run devstral24b_dense   default vllm '{}'    mistralai/Devstral-Small-2-24B-Instruct-2512
run nemo3_nano30b       default vllm '{}'    nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8
run nemo3_nano30b       fine    vllm "$FINE" nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8

# ---- Devstral at 256 agents, with a 60-minute limit instead of 20
echo; echo "======== concurrency: devstral24b_dense / default / 256 agents (rerun) ========"
python concurrency_bench.py --model mistralai/Devstral-Small-2-24B-Instruct-2512 --tag devstral24b_dense \
  --variant default --agents 256 --redo --run-timeout 3600 --request-timeout 1800 --stall-s 600 \
  --out-dir results/conc_large 2>&1 | tee results/conc_large/devstral24b_dense_default_n256_rerun.log

# ---- SGLang (its own environment)
if [ -f /workspace/sgl/bin/activate ]; then
  . /workspace/sgl/bin/activate
  for m in "granite_hmicro|ibm-granite/granite-4.0-h-micro" "nemo3_nano30b|nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"; do
    IFS='|' read -r name hf <<< "$m"
    pkill -f "sglang_multiturn.py|retention_bench.py --engine sglang|sglang::" 2>/dev/null; sleep 3
    run "$name" sgl_auto  sglang '{}' "$hf"
    pkill -f "retention_bench.py --engine sglang|sglang::" 2>/dev/null; sleep 3
    run "$name" sgl_nobuf sglang '{"mamba_radix_cache_strategy":"no_buffer"}' "$hf"
  done
  deactivate
else
  echo "[skip] SGLang environment not found (run run_sglang.sh first)"
fi

python analyze_retention.py results/ret_*_summary.json --out results/ret_summary.csv | tee results/ret_summary.txt
python analyze_concurrency.py results/conc_large --out results/conc_large/summary.csv | tee results/conc_large/summary.txt
tar czf results_checkpoint.tar.gz results/ret_* results/logs/ret_* results/conc_large
echo "Done. Download results_checkpoint.tar.gz"
