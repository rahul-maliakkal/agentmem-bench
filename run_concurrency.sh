#!/usr/bin/env bash
# How many agents fit on one GPU: Granite dense vs hybrid, two sizes.
# Run inside tmux:  tmux new -s conc ; bash run_concurrency.sh
# About 2-3 hours on one H100.
set -uo pipefail
export HF_HOME=${HF_HOME:-/workspace/hf}
# Every run writes to its own folder, so earlier results are never overwritten.
# Override with:  SUFFIX=granite_try2 bash run_concurrency.sh
SUFFIX=${SUFFIX:-$(date +%Y%m%d_%H%M)}
OUT=results/conc_$SUFFIX
mkdir -p "$OUT"

python -c "import vllm, aiohttp" 2>/dev/null || { echo "ERROR: vllm or aiohttp missing (bash setup.sh)"; exit 1; }

# name | HF id | kind
MODELS=(
  "granite_micro_dense|ibm-granite/granite-4.0-micro|dense"
  "granite_hmicro|ibm-granite/granite-4.0-h-micro|hybrid"
  "granite_1b_dense|ibm-granite/granite-4.0-1b|dense"
  "granite_h1b|ibm-granite/granite-4.0-h-1b|hybrid"
)
# Larger pair (Devstral 24B dense vs Nemotron-3-Nano-30B hybrid) adds about 1.5 hours.
# Include it with:  LARGE=1 bash run_concurrency.sh
if [ "${LARGE:-0}" = 1 ]; then
  MODELS+=(
    "devstral24b_dense|mistralai/Devstral-Small-2-24B-Instruct-2512|dense"
    "nemo3_nano30b|nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8|hybrid"
  )
fi
# Run only some models, e.g.:  ONLY="granite_hmicro granite_h1b" bash run_concurrency.sh
if [ -n "${ONLY:-}" ]; then
  KEEP=(); for m in "${MODELS[@]}"; do for o in $ONLY; do [ "${m%%|*}" = "$o" ] && KEEP+=("$m"); done; done
  MODELS=("${KEEP[@]}")
fi
# To resume a stopped run, reuse its folder:  SUFFIX=20261022_0208 bash run_concurrency.sh
# Load levels that already have a _summary.json are skipped.
AGENTS="8,32,64,128,256"
# Hybrids run twice: vLLM default caching, and the finer SSM checkpoint option.
FINE='{"mamba_cache_mode":"align","prefix_match_unit":16}'

for m in "${MODELS[@]}"; do
  IFS='|' read -r name hf kind <<< "$m"
  if [ "$kind" = hybrid ]; then VARIANTS=("default|{}" "fine|$FINE"); else VARIANTS=("default|{}"); fi
  for v in "${VARIANTS[@]}"; do
    vname=${v%%|*}; vargs=${v#*|}
    echo; echo "================ $name / $vname ================"
    pkill -f "vllm serve" 2>/dev/null; pkill -f EngineCore 2>/dev/null; sleep 5
    python concurrency_bench.py --model "$hf" --tag "$name" --variant "$vname" --engine-args "$vargs" \
      --agents "$AGENTS" --out-dir "$OUT" 2>&1 | tee "$OUT/${name}_${vname}_client.log"
  done
done

python analyze_concurrency.py "$OUT" --out "$OUT/summary.csv" --plot "$OUT/concurrency.png" | tee "$OUT/summary.txt"
tar czf "results_concurrency_$SUFFIX.tar.gz" "$OUT"
echo "Done. Download results_concurrency_$SUFFIX.tar.gz"
