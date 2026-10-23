#!/usr/bin/env bash
# SGLang multi-turn replay. Run from the benchmark folder, inside tmux.
# One-time install in a separate venv (keeps your vLLM env intact):
#   python -m venv /workspace/sgl && . /workspace/sgl/bin/activate && pip install -U pip && pip install "sglang[all]"
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p results/logs
if [ ! -f /workspace/sgl/bin/activate ]; then
  echo "Creating SGLang environment in /workspace/sgl (one time, ~10 min)..."
  python -m venv /workspace/sgl && . /workspace/sgl/bin/activate && pip install -q -U pip && pip install -q "sglang[all]" || { echo "ERROR: SGLang install failed"; exit 1; }
fi
. /workspace/sgl/bin/activate
python -c "import sglang; print('sglang', sglang.__version__)" | tee results/sglang_version.txt || { echo "ERROR: sglang does not import"; exit 1; }
export HF_HOME=${HF_HOME:-/workspace/hf}
# SGLang's DeepGEMM kernels need NVCC >= 12.9 to compile; turn them off so older CUDA
# toolkits work (standard kernels are used instead). Set SGLANG_ENABLE_JIT_DEEPGEMM=1 to override.
export SGLANG_ENABLE_JIT_DEEPGEMM=${SGLANG_ENABLE_JIT_DEEPGEMM:-0}
mkdir -p results/logs

MODELS=(
  "granite_hmicro|ibm-granite/granite-4.0-h-micro"
  "granite_h1b|ibm-granite/granite-4.0-h-1b"
  "nemo3_nano30b|nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
)
# auto      = SGLang's default (usually extra_buffer: state tracked every 256 decode tokens)
# nobuf     = no_buffer: a finished request hands its exact final state to the cache
VARIANTS=(
  'sgl_auto|{}'
  'sgl_nobuf|{"mamba_radix_cache_strategy":"no_buffer","disable_overlap_schedule":true}'
)

for m in "${MODELS[@]}"; do
  IFS='|' read -r name hf <<< "$m"
  for v in "${VARIANTS[@]}"; do
    vname=${v%%|*}; vargs=${v#*|}
    for G in 150 1500; do
      out="results/mt_${name}_${vname}_gen$G.csv"
      if [ -s "$out" ]; then echo "[skip] $out already exists"; continue; fi
      echo; echo "======== $name / $vname / gen $G ========"
      pkill -f "sglang_multiturn.py|sglang::" 2>/dev/null; sleep 3   # never match this script's own name
      python sglang_multiturn.py --model "$hf" --variant "${vname}_gen$G" --engine-args "$vargs" \
        --gen-len $G --out "results/mt_${name}_${vname}_gen$G.csv" 2>&1 \
        | tee "results/logs/mt_${name}_${vname}_gen$G.log" | grep -E "strategy|session|wrote|Error|error" || true
    done
  done
done

python analyze_multiturn.py results/mt_*.csv --out results/mt_summary.csv | tee results/mt_summary_sglang.txt
tar czf results_sglang.tar.gz results/mt_*sgl* results/sglang_version.txt results/mt_summary_sglang.txt results/logs/mt_*sgl* 2>/dev/null
echo "Done. Download results_sglang.tar.gz"
