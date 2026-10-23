#!/usr/bin/env bash
# Runs the full viability measurement. Use inside tmux:  tmux new -s bench; bash run_all.sh
# Edit MODELS to change what is measured. Format: short_name|hf_id|prefill_lengths
set -uo pipefail
export HF_HOME=${HF_HOME:-/workspace/hf}
mkdir -p results/logs

# Preflight: stop now instead of failing every step.
if ! python -c "from vllm import LLM, SamplingParams" 2>results/logs/preflight.log; then
  echo "ERROR: vLLM does not import. Last lines:"; tail -5 results/logs/preflight.log
  echo "Run: bash setup.sh   (it repairs the huggingface_hub version)"; exit 1
fi

MODELS=(
  "granite_micro_dense|ibm-granite/granite-4.0-micro|1000,2000,4000,8000,16000,32000"
  "granite_hmicro|ibm-granite/granite-4.0-h-micro|1000,2000,4000,8000,16000,32000,64000"
  "nemo3_nano30b|nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8|1000,2000,4000,8000,16000,32000,64000"
  "granite_1b_dense|ibm-granite/granite-4.0-1b|1000,2000,4000,8000,16000,32000"
  "granite_h1b|ibm-granite/granite-4.0-h-1b|1000,2000,4000,8000,16000,32000,64000"
  "devstral24b_dense|mistralai/Devstral-Small-2-24B-Instruct-2512|1000,2000,4000,8000,16000,32000"
)
CAPACITY_LENGTHS="2048,4096,8192,16384,32768"

echo "== 1. transfer bandwidth (GPU <-> CPU) =="
python transfer_bench.py --out results/transfer.csv 2>&1 | tee results/logs/transfer.log

RUNS=()
for m in "${MODELS[@]}"; do
  IFS='|' read -r name hf lengths <<< "$m"
  echo; echo "================ $name ($hf) ================"

  echo "== 2. analytic state sizes =="
  python state_sizes.py --model "$hf" --out "results/${name}_sizes.json" \
    2>&1 | tee "results/logs/${name}_sizes.log" || { echo "[skip] $name: config parse failed"; continue; }

  echo "== 3. what vLLM actually allocates =="
  python vllm_capacity.py --model "$hf" --lengths "$CAPACITY_LENGTHS" \
    --logdir "results/logs/${name}" --out "results/${name}_capacity.csv" \
    2>&1 | tee "results/logs/${name}_capacity.log"

  echo "== 4. recompute (prefill) cost =="
  python prefill_bench.py --model "$hf" --lengths "$lengths" --batch 8 \
    --out "results/${name}_prefill.csv" 2>&1 | tee "results/logs/${name}_prefill.log" \
    && RUNS+=(--model-run "${name}:results/${name}_sizes.json:results/${name}_prefill.csv")
done

echo; echo "== 5. cost table =="
if [ ${#RUNS[@]} -eq 0 ]; then
  echo "No model finished the prefill step -- nothing to analyze. Check results/logs/."; exit 1
fi
python analyze.py --transfer results/transfer.csv "${RUNS[@]}" \
  --out results/cost_table.csv --plot results/costs.png | tee results/summary.txt

tar czf results.tar.gz results
echo "Done. Download results.tar.gz (or look at results/summary.txt and results/costs.png)."
