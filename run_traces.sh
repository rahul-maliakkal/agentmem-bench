#!/usr/bin/env bash
# Replay real coding-agent sessions (SWE-agent + Claude 3.7 Sonnet on SWE-smith tasks).
#   1. download one shard of SWE-smith trajectories and build traces/swe_smith.json
#   2. Experiments 4-5 (repeated work per resumed turn) on vLLM with the real sessions
#   3. Experiment 6 (same, on SGLang default and exact modes)
#   4. Experiment 7 (agents per GPU) on vLLM, Granite-Micro dense vs Granite-H-Micro
# Run inside tmux:  tmux new -s traces ; bash run_traces.sh        (about 2.5-3 hours)
set -uo pipefail
cd "$(dirname "$0")"
export HF_HOME=${HF_HOME:-/workspace/hf}
mkdir -p results/logs results/traces traces
TRACE=traces/swe_smith.json
FINE='{"mamba_cache_mode":"align","prefix_match_unit":16}'
NOBUF='{"mamba_radix_cache_strategy":"no_buffer","disable_overlap_schedule":true}'

python -c "from vllm import LLM" 2>/dev/null || { echo "ERROR: run from the vLLM environment (deactivate the SGLang venv)"; exit 1; }

# ---- 1. trace file
if [ ! -s "$TRACE" ]; then
  python -c "import pyarrow, huggingface_hub" 2>/dev/null || pip install -q pyarrow huggingface_hub
  python prepare_traces.py --out "$TRACE" 2>&1 | tee results/traces/prepare.log || { echo "ERROR: could not build $TRACE"; exit 1; }
fi

# ---- 2. repeated work per resumed turn, vLLM
mt() {  # name hf variant args
  local out="results/traces/mt_${1}_${3}.csv"
  if [ -s "$out" ]; then echo "[skip] $out"; return; fi
  echo; echo "======== multi-turn (real sessions): $1 / $3 ========"
  pkill -f EngineCore 2>/dev/null; sleep 2
  python multiturn_bench.py --model "$2" --variant "$3" --engine-args "$4" --trace "$TRACE" --trace-sessions 12 \
    --out "$out" 2>&1 | tee "results/logs/trace_mt_${1}_${3}.log" | grep -E "\[info\]|session|wrote|Error" || true
}
mt granite_micro_dense ibm-granite/granite-4.0-micro default '{}'
mt granite_hmicro      ibm-granite/granite-4.0-h-micro default '{}'
mt granite_hmicro      ibm-granite/granite-4.0-h-micro fine "$FINE"
mt nemo3_nano30b       nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 default '{}'
mt nemo3_nano30b       nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 fine "$FINE"

# ---- 4. agents per GPU, vLLM (run before SGLang so the vLLM environment is still active)
for m in "granite_micro_dense|ibm-granite/granite-4.0-micro" "granite_hmicro|ibm-granite/granite-4.0-h-micro"; do
  IFS='|' read -r name hf <<< "$m"
  echo; echo "======== agents per GPU (real sessions): $name ========"
  pkill -f "vllm serve" 2>/dev/null; pkill -f EngineCore 2>/dev/null; sleep 5
  python concurrency_bench.py --model "$hf" --tag "$name" --variant trace --trace "$TRACE" --turns 20 \
    --agents 32,64,128,256 --run-timeout 1800 --out-dir results/traces/conc 2>&1 \
    | tee "results/traces/conc_${name}_client.log"
done
pkill -f "vllm serve" 2>/dev/null; pkill -f EngineCore 2>/dev/null

# ---- 3. repeated work per resumed turn, SGLang
if [ -f /workspace/sgl/bin/activate ]; then
  . /workspace/sgl/bin/activate
  export SGLANG_ENABLE_JIT_DEEPGEMM=${SGLANG_ENABLE_JIT_DEEPGEMM:-0}
  for m in "granite_hmicro|ibm-granite/granite-4.0-h-micro" "nemo3_nano30b|nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"; do
    IFS='|' read -r name hf <<< "$m"
    for v in "sgl_default|{}" "sgl_exact|$NOBUF"; do
      vname=${v%%|*}; vargs=${v#*|}; out="results/traces/mt_${name}_${vname}.csv"
      if [ -s "$out" ]; then echo "[skip] $out"; continue; fi
      echo; echo "======== multi-turn (real sessions, SGLang): $name / $vname ========"
      pkill -f "sglang_multiturn.py|sglang::" 2>/dev/null; sleep 3
      python sglang_multiturn.py --model "$hf" --variant "$vname" --engine-args "$vargs" --trace "$TRACE" \
        --trace-sessions 12 --out "$out" 2>&1 | tee "results/logs/trace_mt_${name}_${vname}.log" \
        | grep -E "\[info\]|session|wrote|Error|error" || true
    done
  done
  deactivate
else
  echo "[skip] SGLang environment not found; run run_sglang.sh once to create it"
fi

python analyze_multiturn.py results/traces/mt_*.csv --out results/traces/mt_summary.csv | tee results/traces/mt_summary.txt
python analyze_concurrency.py results/traces/conc --out results/traces/conc/summary.csv | tee results/traces/conc/summary.txt
tar czf results_traces.tar.gz results/traces results/logs/trace_*
echo "Done. Download results_traces.tar.gz"
