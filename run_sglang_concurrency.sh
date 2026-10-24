#!/usr/bin/env bash
# Experiment 7 on SGLang: agents per GPU for dense Granite-Micro, Granite-H-Micro and Nemotron,
# with SGLang's default (extra_buffer) and exact (no_buffer) checkpoint modes.
# Run inside tmux:  tmux new -s sglc ; bash run_sglang_concurrency.sh      (about 1.5-2 hours)
set -uo pipefail
cd "$(dirname "$0")"
export HF_HOME=${HF_HOME:-/workspace/hf}
export SGLANG_ENABLE_JIT_DEEPGEMM=${SGLANG_ENABLE_JIT_DEEPGEMM:-0}   # DeepGEMM needs NVCC >= 12.9
OUT=results/conc_sgl
mkdir -p "$OUT" results/logs

if [ ! -f /workspace/sgl/bin/activate ]; then
  echo "Creating SGLang environment in /workspace/sgl (one time, ~10 min)..."
  python -m venv /workspace/sgl && . /workspace/sgl/bin/activate && pip install -q -U pip && pip install -q "sglang[all]" aiohttp \
    || { echo "ERROR: SGLang install failed"; exit 1; }
fi
. /workspace/sgl/bin/activate
python -c "import sglang, aiohttp; print('sglang', sglang.__version__)" | tee "$OUT/sglang_version.txt" \
  || { echo "ERROR: sglang or aiohttp does not import (pip install aiohttp in /workspace/sgl)"; exit 1; }

NOBUF='{"mamba_radix_cache_strategy":"no_buffer","disable_overlap_schedule":true}'
RUNS=(
  "granite_micro_dense|ibm-granite/granite-4.0-micro|sgl_default|{}"
  "granite_hmicro|ibm-granite/granite-4.0-h-micro|sgl_default|{}"
  "granite_hmicro|ibm-granite/granite-4.0-h-micro|sgl_exact|$NOBUF"
  "nemo3_nano30b|nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8|sgl_default|{}"
  "nemo3_nano30b|nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8|sgl_exact|$NOBUF"
)
for r in "${RUNS[@]}"; do
  IFS='|' read -r name hf variant args <<< "$r"
  echo; echo "================ $name / $variant (SGLang) ================"
  pkill -f "sglang.launch_server|sglang::" 2>/dev/null; pkill -f "vllm serve" 2>/dev/null; sleep 5
  python concurrency_bench.py --engine sglang --model "$hf" --tag "$name" --variant "$variant" \
    --engine-args "$args" --agents 8,32,64,128,256 --out-dir "$OUT" 2>&1 | tee "$OUT/${name}_${variant}_client.log"
done
pkill -f "sglang.launch_server|sglang::" 2>/dev/null

python analyze_concurrency.py "$OUT" --out "$OUT/summary.csv" --plot "$OUT/concurrency.png" | tee "$OUT/summary.txt"
tar czf results_sglang_concurrency.tar.gz "$OUT"
echo "Done. Download results_sglang_concurrency.tar.gz"
