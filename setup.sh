#!/usr/bin/env bash
# One-time setup on a RunPod pod. Run from the repo folder:  bash setup.sh
set -euo pipefail

# Keep model weights on the persistent volume so they survive pod restarts.
export HF_HOME=/workspace/hf
mkdir -p "$HF_HOME" results/logs
grep -q "HF_HOME=/workspace/hf" ~/.bashrc 2>/dev/null || echo 'export HF_HOME=/workspace/hf' >> ~/.bashrc

python -m pip install -U pip
# vLLM pins compatible torch / transformers / huggingface_hub. Do NOT upgrade those separately.
python -c "import vllm" 2>/dev/null || python -m pip install -U vllm
python -m pip install matplotlib

# Repair a too-new huggingface_hub (e.g. from an earlier run of this script):
# transformers refuses to import if it is outside the range it was built for.
if ! python -c "import transformers" 2>/dev/null; then
  echo "[fix] transformers failed to import; pinning huggingface_hub to <2.0"
  python -m pip install "huggingface_hub>=1.5.0,<2.0"
fi

# Hard check: vLLM must import, or nothing else will work.
if ! python -c "from vllm import LLM, SamplingParams" ; then
  echo "ERROR: vLLM does not import. Fix the error above before running run_all.sh." >&2
  exit 1
fi

# Log the environment with your results -- reviewers will ask.
{
  echo "date: $(date -u)"
  nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.current,pcie.link.width.current --format=csv
  python -c "import torch, vllm, transformers, huggingface_hub as h; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('vllm', vllm.__version__); print('transformers', transformers.__version__, 'hf_hub', h.__version__)"
} | tee results/env.txt

echo
echo "Setup OK. If a model is gated:  huggingface-cli login   (or export HF_TOKEN=...)"
echo "Then:  bash run_all.sh"
