#!/usr/bin/env bash
# Remaining SGLang runs on real SWE-agent sessions:
#   1. Experiments 4-6 (repeated work per turn): dense Granite-1B and Devstral on SGLang      (~20 min)
#   2. Experiment 7 (agents per GPU): Granite-1B pair (default, exact, 16-bit) and Devstral    (~3 h)
#   3. Experiment 7: Nemotron exact and 16-bit, so the large pair has the same SGLang settings (~2 h)
# Run inside tmux, from any environment:  tmux new -s sgl2 ; bash run_sglang_rest.sh
# Set STEPS="1 2" to skip part 3. Finished load levels are skipped on restart.
set -uo pipefail
cd "$(dirname "$0")"
export HF_HOME=${HF_HOME:-/workspace/hf}
export SGLANG_ENABLE_JIT_DEEPGEMM=${SGLANG_ENABLE_JIT_DEEPGEMM:-0}   # DeepGEMM needs NVCC >= 12.9
STEPS=${STEPS:-"1 2 3"}
has() { case " $STEPS " in *" $1 "*) return 0;; *) return 1;; esac; }
TRACE=traces/swe_smith.json
mkdir -p results/logs results/traces/conc_sgl
NOBUF='{"mamba_radix_cache_strategy":"no_buffer","disable_overlap_schedule":true}'
BF16='{"mamba_ssm_dtype":"bfloat16"}'
NOBUF_BF16='{"mamba_radix_cache_strategy":"no_buffer","disable_overlap_schedule":true,"mamba_ssm_dtype":"bfloat16"}'
G1=ibm-granite/granite-4.0-1b; GH1=ibm-granite/granite-4.0-h-1b
DEV=mistralai/Devstral-Small-2-24B-Instruct-2512; NEMO=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8

[ -f /workspace/sgl/bin/activate ] || { echo "ERROR: no SGLang environment in /workspace/sgl (run run_sglang.sh once)"; exit 1; }
[ -s "$TRACE" ] || { echo "ERROR: $TRACE missing (run run_traces.sh or run_coding_agents.sh first)"; exit 1; }
. /workspace/sgl/bin/activate
python -c "import sglang, aiohttp; print('sglang', sglang.__version__)" || { echo "ERROR: sglang does not import"; exit 1; }
for hf in $G1 $GH1 $DEV $NEMO; do
  python -c "from huggingface_hub import snapshot_download as d; d('$hf')" >/dev/null 2>&1 || echo "[warn] could not pre-download $hf"
done
export HF_HUB_OFFLINE=1
stop() { pkill -f "sglang.launch_server|sglang::|sglang_multiturn.py" 2>/dev/null; sleep 10; }

# ---- 1. repeated work per turn, dense baselines on SGLang
if has 1; then
  for m in "granite_1b_dense|$G1" "devstral24b_dense|$DEV"; do
    IFS='|' read -r name hf <<< "$m"; out="results/traces/mt_${name}_sgl_default.csv"
    if [ -s "$out" ]; then echo "[skip] $out"; continue; fi
    echo; echo "======== [1] multi-turn (SWE-agent sessions, SGLang): $name ========"
    stop
    python sglang_multiturn.py --model "$hf" --variant sgl_default --engine-args '{}' --trace "$TRACE" \
      --trace-sessions 12 --out "$out" 2>&1 | tee "results/logs/trace_mt_${name}_sgl_default.log" \
      | grep -E "\[info\]|session|wrote|Error|error" || true
  done
fi

conc() {  # name hf variant args agents
  echo; echo "======== agents per GPU on SGLang (SWE-agent sessions): $1 / $3 ========"
  stop
  python concurrency_bench.py --engine sglang --model "$2" --tag "$1" --variant "trace_$3" --engine-args "$4" \
    --trace "$TRACE" --turns 20 --agents "$5" --run-timeout 3600 --request-timeout 1800 --stall-s 600 \
    --out-dir results/traces/conc_sgl 2>&1 | tee -a "results/traces/conc_sgl_${1}_${3}_client.log"
}
# ---- 2. agents per GPU: Granite-1B pair and Devstral
if has 2; then
  conc granite_1b_dense  "$G1"  sgl_default      '{}'          32,64,128,256
  conc granite_h1b       "$GH1" sgl_default      '{}'          32,64,128,256
  conc granite_h1b       "$GH1" sgl_default_bf16 "$BF16"       32,64,128,256
  conc granite_h1b       "$GH1" sgl_exact        "$NOBUF"      32,64,128,256
  conc granite_h1b       "$GH1" sgl_exact_bf16   "$NOBUF_BF16" 32,64,128,256
  conc devstral24b_dense "$DEV" sgl_default      '{}'          8,32,64,128
fi
# ---- 3. agents per GPU: Nemotron with the remaining SGLang settings
if has 3; then
  conc nemo3_nano30b "$NEMO" sgl_default_bf16 "$BF16"       32,64,128,256
  conc nemo3_nano30b "$NEMO" sgl_exact        "$NOBUF"      32,64,128,256
  conc nemo3_nano30b "$NEMO" sgl_exact_bf16   "$NOBUF_BF16" 32,64,128,256
fi
stop
deactivate

python analyze_multiturn.py results/traces/mt_*.csv --out results/traces/mt_summary.csv | tee results/traces/mt_summary.txt
python analyze_concurrency.py results/traces/conc_sgl --out results/traces/conc_sgl/summary.csv | tee results/traces/conc_sgl/summary.txt
tar czf results_sglang_rest.tar.gz results/traces/conc_sgl results/traces/mt_* results/logs/trace_mt_* 2>/dev/null
echo "Done. Download results_sglang_rest.tar.gz"
