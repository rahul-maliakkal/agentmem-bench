#!/usr/bin/env bash
# Multi-turn agent replay for all models and caching variants. Run inside tmux:
#   tmux new -s mt ; bash run_multiturn.sh
# Each (model, variant) runs in a fresh process so caches never leak between runs.
set -uo pipefail
export HF_HOME=${HF_HOME:-/workspace/hf}
mkdir -p results/logs

if ! python -c "from vllm import LLM" 2>results/logs/preflight.log; then
  echo "ERROR: vLLM does not import:"; tail -5 results/logs/preflight.log; exit 1
fi

MODELS=(
  "granite_micro_dense|ibm-granite/granite-4.0-micro|dense"
  "granite_hmicro|ibm-granite/granite-4.0-h-micro|hybrid"
  "nemo3_nano30b|nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8|hybrid"
  "granite_1b_dense|ibm-granite/granite-4.0-1b|dense"
  "granite_h1b|ibm-granite/granite-4.0-h-1b|hybrid"
  "devstral24b_dense|mistralai/Devstral-Small-2-24B-Instruct-2512|dense"
)

# Caching variants. 'default' = plain prefix caching.
# The "fine" variant: align-mode Mamba caching with a 16-token match unit, which adds a
# checkpoint at the end of each prompt (not after the model reply).
VARIANTS_HYBRID=(
  'default|{}'
  'fine|{"mamba_cache_mode":"align","prefix_match_unit":16}'
)
# prefix_match_unit must divide the attention block size (528 for Qwen3.5, 4176 for Nemotron): 16 works for both.
VARIANTS_DENSE=('default|{}')

for m in "${MODELS[@]}"; do
  IFS='|' read -r name hf kind <<< "$m"
  if [ "$kind" = hybrid ]; then V=("${VARIANTS_HYBRID[@]}"); else V=("${VARIANTS_DENSE[@]}"); fi
  for v in "${V[@]}"; do
    vname=${v%%|*}; vargs=${v#*|}
    if [ -s "results/mt_${name}_${vname}.csv" ] && [ "${REDO:-0}" != 1 ]; then
      echo "[skip] $name / $vname already done (REDO=1 to rerun)"; continue
    fi
    echo; echo "======== $name / $vname ========"
    pkill -f EngineCore 2>/dev/null; sleep 2
    python multiturn_bench.py --model "$hf" --variant "$vname" --engine-args "$vargs" \
      --out "results/mt_${name}_${vname}.csv" 2>&1 | tee "results/logs/mt_${name}_${vname}.log" | \
      grep -E "session|wrote|skip|warn|Error|block size" || true
  done
done

echo; echo "======== summary ========"
python analyze_multiturn.py results/mt_*.csv --out results/mt_summary.csv --plot results/mt_waste.png \
  | tee results/mt_summary.txt
grep -h "Setting attention block size" results/logs/mt_*.log | sed 's/.*Setting/Setting/' | sort | uniq -c
tar czf results_multiturn.tar.gz results
echo "Done. Download results_multiturn.tar.gz"
