#!/usr/bin/env bash
# Accuracy round 2 (about 3-4 hours on one H100). Four checks per model, all on real agent data:
#   1. needle      long-context retrieval at 4K-32K tokens (multi-key needle hidden in real tool output)
#   2. replaylong  likelihood of Claude's real next reply at 12K-31K-token histories
#   3. humaneval   HumanEval pass@1 (coding)
#   5. cachecheck  same prompt computed from scratch vs resumed from cache: are outputs identical?
# Runs: all six models on vLLM (default placement); vLLM align for the two hybrids (cache check only);
#       SGLang 32-bit and 16-bit state for Granite-H-Micro and Nemotron (needle + long replay),
#       SGLang default and exact placement (cache check).
# Run inside tmux, from the vLLM environment:  tmux new -s acc2 ; bash run_acc2.sh
# Finished pieces are skipped on restart. Upload results_acc2.tar.gz when done.
set -uo pipefail
cd "$(dirname "$0")"
export HF_HOME=${HF_HOME:-/workspace/hf}
export SGLANG_ENABLE_JIT_DEEPGEMM=${SGLANG_ENABLE_JIT_DEEPGEMM:-0}
OUT=results/accuracy2
mkdir -p $OUT results/logs
TRACE1K=traces/swe_smith_1500.json
FINE='{"mamba_cache_mode":"align","prefix_match_unit":16}'
NOBUF='{"mamba_radix_cache_strategy":"no_buffer","disable_overlap_schedule":true}'
BF16='{"mamba_ssm_dtype":"bfloat16"}'
G1=ibm-granite/granite-4.0-1b; GH1=ibm-granite/granite-4.0-h-1b; GM=ibm-granite/granite-4.0-micro; GHM=ibm-granite/granite-4.0-h-micro
DEV=mistralai/Devstral-Small-2-24B-Instruct-2512; NEMO=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8
stop() {  # kill any server and wait until the GPU memory is actually released (up to 3 minutes)
  pkill -f "vllm serve" 2>/dev/null; pkill -f EngineCore 2>/dev/null; pkill -f "sglang.launch_server|sglang::" 2>/dev/null
  for _ in $(seq 1 36); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "$used" ] && [ "$used" -lt 2000 ] && return 0
    sleep 5; pkill -9 -f "vllm serve|EngineCore|sglang::|sglang.launch_server" 2>/dev/null
  done
  echo "[warn] GPU still has ${used:-?} MiB in use after 3 minutes"
}
exec 9>/tmp/agentmem_runner.lock
flock -n 9 || { echo "ERROR: another run_acc2.sh is already running. Stop it first."; exit 1; }
if pgrep -f "run_final.sh" >/dev/null; then echo "ERROR: run_final.sh is still running (pgrep -af run_final). Stop it first."; exit 1; fi
python -c "from vllm import LLM" 2>results/logs/vllm_check.log || { echo "ERROR: vLLM does not import"; tail -5 results/logs/vllm_check.log; exit 1; }
python -c "import pandas, pyarrow, huggingface_hub" 2>/dev/null || pip install -q pandas pyarrow huggingface_hub
python -c "import vllm, transformers, torch; print('versions: vllm', vllm.__version__, 'transformers', transformers.__version__, 'torch', torch.__version__)" | tee $OUT/versions.txt
[ -s "$TRACE1K" ] || python prepare_traces.py --n 1500 --out "$TRACE1K" | tee results/logs/prepare_1500.log
python -c "from huggingface_hub import hf_hub_download as d; d('openai/openai_humaneval','openai_humaneval/test-00000-of-00001.parquet',repo_type='dataset')" \
  >/dev/null 2>&1 || echo "[warn] could not download HumanEval"
for hf in $G1 $GH1 $GM $GHM $DEV $NEMO; do
  python -c "from huggingface_hub import snapshot_download as d; d('$hf')" >/dev/null 2>&1 || echo "[warn] could not pre-download $hf"
done
export HF_HUB_OFFLINE=1
# quick check that long prompts are built correctly before spending GPU time
python accuracy2_bench.py --model "$GHM" --tag check --selftest --trace "$TRACE1K" 2>&1 | tail -2

acc2() {  # engine name variant args hf tasks
  if [ -s "$OUT/${2}_${3}_summary.json" ]; then echo "[skip] $2 / $3"; return; fi
  echo; echo "======== accuracy2: $2 / $3 ($1): $6 ========"
  stop
  python accuracy2_bench.py --engine "$1" --model "$5" --tag "$2" --variant "$3" --engine-args "$4" --tasks "$6" \
    --trace "$TRACE1K" --out-dir $OUT 2>&1 | tee "results/logs/acc2_${2}_${3}.log" | grep -E "^\[|Error|error" || true
}

for m in "granite_1b_dense|$G1" "granite_h1b|$GH1" "granite_micro_dense|$GM" "granite_hmicro|$GHM" "devstral24b_dense|$DEV" "nemo3_nano30b|$NEMO"; do
  IFS='|' read -r name hf <<< "$m"
  args='{}'; [ "$name" = nemo3_nano30b ] && args='{"max_num_seqs":256}'
  acc2 vllm "$name" vllm "$args" "$hf" needle,replaylong,humaneval,cachecheck
done
acc2 vllm granite_hmicro vllm_align "$FINE" "$GHM" cachecheck
acc2 vllm nemo3_nano30b  vllm_align '{"mamba_cache_mode":"align","prefix_match_unit":16,"max_num_seqs":256}' "$NEMO" cachecheck

if [ -f /workspace/sgl/bin/activate ]; then
  . /workspace/sgl/bin/activate
  python -c "import pandas, pyarrow, transformers, aiohttp" 2>/dev/null || pip install -q pandas pyarrow transformers aiohttp
  for m in "granite_hmicro|$GHM" "nemo3_nano30b|$NEMO"; do
    IFS='|' read -r name hf <<< "$m"
    acc2 sglang "$name" sgl_fp32  '{}'     "$hf" needle,replaylong,cachecheck
    acc2 sglang "$name" sgl_bf16  "$BF16"  "$hf" needle,replaylong
    acc2 sglang "$name" sgl_exact "$NOBUF" "$hf" cachecheck
  done
  deactivate
else
  echo "[skip] SGLang runs: no /workspace/sgl"
fi
stop
python analyze_accuracy2.py $OUT | tee $OUT/summary.txt
tar czf results_acc2.tar.gz $OUT results/logs/acc2_* 2>/dev/null
echo "Done. Upload results_acc2.tar.gz"
