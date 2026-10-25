#!/usr/bin/env bash
# Coding-agent (SWE-agent) versions of the remaining experiments:
#   1. Experiment 7, agents per GPU: Devstral 24B vs Nemotron 30B on vLLM       (~2-2.5 h)
#   2. Experiment 8, resume points vs paused sessions, vLLM and SGLang           (~1.5 h)
#   3. Experiment 7 on SGLang: Granite-Micro pair (+ Nemotron)                   (~2-2.5 h)
#   4. Experiments 4-6 for the models not yet run on real sessions:
#      Granite-1B pair and Devstral (vLLM), Granite-H-1B (SGLang)                 (~45 min)
#   5. Experiment 7 for the Granite-1B pair on vLLM                               (~1.5 h)
# Run inside tmux, from the vLLM environment:  tmux new -s coding ; bash run_coding_agents.sh
# Set STEPS="1 2" to run only some steps. Finished pieces are skipped on restart.
set -uo pipefail
cd "$(dirname "$0")"
export HF_HOME=${HF_HOME:-/workspace/hf}
STEPS=${STEPS:-"1 2 4 3 5"}
mkdir -p results/logs results/traces/conc results/traces/conc_sgl traces
TRACE=traces/swe_smith.json          # 256 sessions (built by run_traces.sh)
TRACE1K=traces/swe_smith_1500.json   # 1,500 sessions, so 1,024 still fit after dropping over-long ones
FINE='{"mamba_cache_mode":"align","prefix_match_unit":16}'
NOBUF='{"mamba_radix_cache_strategy":"no_buffer","disable_overlap_schedule":true}'
BF16='{"mamba_ssm_dtype":"bfloat16"}'
NOBUF_BF16='{"mamba_radix_cache_strategy":"no_buffer","disable_overlap_schedule":true,"mamba_ssm_dtype":"bfloat16"}'
has() { case " $STEPS " in *" $1 "*) return 0;; *) return 1;; esac; }

if ! python -c "from vllm import LLM" 2>results/logs/vllm_check.log; then
  echo "ERROR: vLLM does not import with $(which python):"; tail -5 results/logs/vllm_check.log
  echo "Run: deactivate (if in the SGLang venv), or bash setup.sh (if vLLM is not installed)"; exit 1
fi
python -c "import pyarrow, huggingface_hub" 2>/dev/null || pip install -q pyarrow huggingface_hub
[ -s "$TRACE" ] || python prepare_traces.py --out "$TRACE" | tee results/traces/prepare.log
[ -s "$TRACE1K" ] || python prepare_traces.py --n 1500 --out "$TRACE1K" | tee results/traces/prepare_1500.log

# ---- 1. agents per GPU, large pair, vLLM
if has 1; then
  for m in "devstral24b_dense|mistralai/Devstral-Small-2-24B-Instruct-2512" "nemo3_nano30b|nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"; do
    IFS='|' read -r name hf <<< "$m"
    echo; echo "======== [1] agents per GPU (SWE-agent sessions): $name ========"
    pkill -f "vllm serve" 2>/dev/null; pkill -f EngineCore 2>/dev/null; sleep 5
    python concurrency_bench.py --model "$hf" --tag "$name" --variant trace --trace "$TRACE" --turns 20 \
      --agents 8,32,64,128,256 --run-timeout 3600 --request-timeout 1800 --stall-s 600 \
      --out-dir results/traces/conc 2>&1 | tee "results/traces/conc_${name}_client.log"
  done
  pkill -f "vllm serve" 2>/dev/null; pkill -f EngineCore 2>/dev/null
fi

# ---- 2. resume points vs paused sessions
ret() {  # name variant engine args hf
  local out="results/traces/ret_${1}_${2}.csv"
  if [ -s "${out%.csv}_summary.json" ]; then echo "[skip] $1 / $2"; return; fi
  echo; echo "======== [2] retention (SWE-agent sessions): $1 / $2 ($3) ========"
  python retention_bench.py --engine "$3" --model "$5" --variant "$2" --engine-args "$4" --trace "$TRACE1K" \
    --out "$out" 2>&1 | tee "results/logs/trace_ret_${1}_${2}.log" | grep -E "^N=|\[info\]|\[warn\]|Error|error|wrote" || true
}
if has 2; then
  pkill -f EngineCore 2>/dev/null; sleep 3
  ret granite_micro_dense default vllm '{}'    ibm-granite/granite-4.0-micro
  ret granite_hmicro      default vllm '{}'    ibm-granite/granite-4.0-h-micro
  ret granite_hmicro      fine    vllm "$FINE" ibm-granite/granite-4.0-h-micro
  ret devstral24b_dense   default vllm '{}'    mistralai/Devstral-Small-2-24B-Instruct-2512
  ret nemo3_nano30b       default vllm '{}'    nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8
  ret nemo3_nano30b       fine    vllm "$FINE" nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8
fi

# ---- 4. repeated work per resumed turn for the remaining models, vLLM
mt() {  # name hf variant args
  local out="results/traces/mt_${1}_${3}.csv"
  if [ -s "$out" ]; then echo "[skip] $out"; return; fi
  echo; echo "======== [4] multi-turn (SWE-agent sessions): $1 / $3 ========"
  pkill -f EngineCore 2>/dev/null; sleep 2
  python multiturn_bench.py --model "$2" --variant "$3" --engine-args "$4" --trace "$TRACE" --trace-sessions 12 \
    --out "$out" 2>&1 | tee "results/logs/trace_mt_${1}_${3}.log" | grep -E "\[info\]|session|wrote|Error" || true
}
if has 4; then
  mt granite_1b_dense ibm-granite/granite-4.0-1b default '{}'
  mt granite_h1b      ibm-granite/granite-4.0-h-1b default '{}'
  mt granite_h1b      ibm-granite/granite-4.0-h-1b fine "$FINE"
  mt devstral24b_dense mistralai/Devstral-Small-2-24B-Instruct-2512 default '{}'
fi

# ---- 5. agents per GPU, Granite-1B pair, vLLM
if has 5; then
  for m in "granite_1b_dense|ibm-granite/granite-4.0-1b" "granite_h1b|ibm-granite/granite-4.0-h-1b"; do
    IFS='|' read -r name hf <<< "$m"
    echo; echo "======== [5] agents per GPU (SWE-agent sessions): $name ========"
    pkill -f "vllm serve" 2>/dev/null; pkill -f EngineCore 2>/dev/null; sleep 5
    python concurrency_bench.py --model "$hf" --tag "$name" --variant trace --trace "$TRACE" --turns 20 \
      --agents 32,64,128,256 --run-timeout 1800 --out-dir results/traces/conc 2>&1 | tee "results/traces/conc_${name}_client.log"
  done
  pkill -f "vllm serve" 2>/dev/null; pkill -f EngineCore 2>/dev/null
fi

# ---- SGLang parts (2, 3 and 4) run in the SGLang environment
if (has 2 || has 3 || has 4) && [ -f /workspace/sgl/bin/activate ]; then
  . /workspace/sgl/bin/activate
  export SGLANG_ENABLE_JIT_DEEPGEMM=${SGLANG_ENABLE_JIT_DEEPGEMM:-0}
  export HF_HUB_OFFLINE=1   # models are already downloaded by the vLLM steps
  if has 2; then
    for m in "granite_hmicro|ibm-granite/granite-4.0-h-micro" "nemo3_nano30b|nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"; do
      IFS='|' read -r name hf <<< "$m"
      pkill -f "retention_bench.py --engine sglang|sglang::" 2>/dev/null; sleep 3
      ret "$name" sgl_default sglang '{}' "$hf"
      pkill -f "retention_bench.py --engine sglang|sglang::" 2>/dev/null; sleep 3
      ret "$name" sgl_exact sglang "$NOBUF" "$hf"
    done
  fi
  if has 4; then
    for v in "sgl_default|{}" "sgl_exact|$NOBUF"; do
      vname=${v%%|*}; vargs=${v#*|}; out="results/traces/mt_granite_h1b_${vname}.csv"
      if [ -s "$out" ]; then echo "[skip] $out"; continue; fi
      echo; echo "======== [4] multi-turn (SWE-agent sessions, SGLang): granite_h1b / $vname ========"
      pkill -f "sglang_multiturn.py|sglang::" 2>/dev/null; sleep 3
      python sglang_multiturn.py --model ibm-granite/granite-4.0-h-1b --variant "$vname" --engine-args "$vargs" \
        --trace "$TRACE" --trace-sessions 12 --out "$out" 2>&1 | tee "results/logs/trace_mt_granite_h1b_${vname}.log" \
        | grep -E "\[info\]|session|wrote|Error|error" || true
    done
  fi
  if has 3; then
    for r in "granite_micro_dense|ibm-granite/granite-4.0-micro|sgl_default|{}" \
             "granite_hmicro|ibm-granite/granite-4.0-h-micro|sgl_default|{}" \
             "granite_hmicro|ibm-granite/granite-4.0-h-micro|sgl_exact|$NOBUF" \
             "granite_hmicro|ibm-granite/granite-4.0-h-micro|sgl_default_bf16|$BF16" \
             "granite_hmicro|ibm-granite/granite-4.0-h-micro|sgl_exact_bf16|$NOBUF_BF16" \
             "nemo3_nano30b|nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8|sgl_default|{}"; do
      IFS='|' read -r name hf variant args <<< "$r"
      echo; echo "======== [3] agents per GPU on SGLang (SWE-agent sessions): $name / $variant ========"
      pkill -f "sglang.launch_server|sglang::" 2>/dev/null; sleep 10
      python concurrency_bench.py --engine sglang --model "$hf" --tag "$name" --variant "trace_$variant" \
        --engine-args "$args" --trace "$TRACE" --turns 20 --agents 32,64,128,256 --run-timeout 3600 \
        --out-dir results/traces/conc_sgl 2>&1 | tee "results/traces/conc_sgl_${name}_${variant}_client.log"
    done
    pkill -f "sglang.launch_server|sglang::" 2>/dev/null
  fi
  deactivate
elif has 2 || has 3 || has 4; then
  echo "[skip] SGLang parts: environment /workspace/sgl not found (run run_sglang.sh once)"
fi

python analyze_multiturn.py results/traces/mt_*.csv --out results/traces/mt_summary.csv | tee results/traces/mt_summary.txt
python analyze_retention.py results/traces/ret_*_summary.json --out results/traces/ret_summary.csv | tee results/traces/ret_summary.txt
python analyze_concurrency.py results/traces/conc --out results/traces/conc/summary.csv | tee results/traces/conc/summary.txt
[ -n "$(ls results/traces/conc_sgl/*_summary.json 2>/dev/null)" ] && \
  python analyze_concurrency.py results/traces/conc_sgl --out results/traces/conc_sgl/summary.csv | tee results/traces/conc_sgl/summary.txt
tar czf results_coding_agents.tar.gz results/traces/conc results/traces/conc_sgl results/traces/ret_* results/traces/mt_* \
  results/traces/*.log results/logs/trace_ret_* 2>/dev/null
echo "Done. Download results_coding_agents.tar.gz"
