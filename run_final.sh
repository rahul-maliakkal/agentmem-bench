#!/usr/bin/env bash
# Final batch for the paper (about 7-8 hours on one H100; each step can run on its own):
#   1. Accuracy: dense vs hybrid (vLLM, all six models) on real-session reply likelihood and GSM8K,
#      plus 32- vs 16-bit recurrent state on SGLang with greedy output-equality checks   (~2.5 h)
#   2. Branching agents: resume from 0-8 turns back, dense vs hybrid, vLLM and SGLang    (~1.5 h)
#   3. Longer sessions: memory per session at 64K and 128K tokens                         (~20 min)
#   4. Tool-pause sweep: agents per GPU with median tool pauses of 1, 10 and 30 s          (~2 h)
#   5. Memory-budget sweep: paused-session retention with less GPU memory                  (~40 min)
#   6. Model check: paused-session retention on a fine grid around the capacity the memory
#      rule predicts (dense Granite-Micro 65, Granite-H-Micro 469, Devstral 17, Nemotron 357) (~1 h)
# Default order: 1 6 2 3 4 5
# Run inside tmux, from the vLLM environment:  tmux new -s final ; bash run_final.sh
# Set STEPS="2 3" to run only some steps. Finished pieces are skipped on restart.
set -uo pipefail
cd "$(dirname "$0")"
export HF_HOME=${HF_HOME:-/workspace/hf}
export SGLANG_ENABLE_JIT_DEEPGEMM=${SGLANG_ENABLE_JIT_DEEPGEMM:-0}
STEPS=${STEPS:-"1 6 2 3 4 5"}
has() { case " $STEPS " in *" $1 "*) return 0;; *) return 1;; esac; }
mkdir -p results/traces/ret_fine results/logs results/accuracy results/branch results/capacity_long results/traces/conc_pause results/traces/ret_mem
TRACE=traces/swe_smith.json; TRACE1K=traces/swe_smith_1500.json
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
  echo "[warn] GPU still has ${used:-?} MiB in use after 3 minutes"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
}

if ! python -c "from vllm import LLM" 2>results/logs/vllm_check.log; then
  echo "ERROR: vLLM does not import with $(which python):"; tail -5 results/logs/vllm_check.log; exit 1
fi
python -c "import pandas, pyarrow, huggingface_hub, transformers" 2>/dev/null || pip install -q pandas pyarrow huggingface_hub "transformers<5"
python -c "import vllm, transformers, torch; print('versions: vllm', vllm.__version__, 'transformers', transformers.__version__, 'torch', torch.__version__)" | tee results/logs/versions_final.txt
[ -s "$TRACE" ] || python prepare_traces.py --out "$TRACE" | tee results/logs/prepare.log
[ -s "$TRACE1K" ] || python prepare_traces.py --n 1500 --out "$TRACE1K" | tee results/logs/prepare_1500.log
python -c "from huggingface_hub import hf_hub_download as d; d('openai/gsm8k','main/test-00000-of-00001.parquet',repo_type='dataset')" \
  >/dev/null 2>&1 || echo "[warn] could not download GSM8K"
for hf in $G1 $GH1 $GM $GHM $DEV $NEMO; do
  python -c "from huggingface_hub import snapshot_download as d; d('$hf')" >/dev/null 2>&1 || echo "[warn] could not pre-download $hf"
done
export HF_HUB_OFFLINE=1
SGL_OK=0; [ -f /workspace/sgl/bin/activate ] && SGL_OK=1
sgl_on() { . /workspace/sgl/bin/activate; python -c "import pandas, pyarrow, transformers" 2>/dev/null || pip install -q pandas pyarrow transformers; }

# ---- 1. accuracy
acc() {  # engine name variant args hf tasks
  if [ -s "results/accuracy/${2}_${3}_summary.json" ]; then echo "[skip] accuracy $2 / $3"; return; fi
  echo; echo "======== [1] accuracy: $2 / $3 ($1) ========"
  stop
  python accuracy_bench.py --engine "$1" --model "$5" --tag "$2" --variant "$3" --engine-args "$4" --tasks "$6" \
    --trace "$TRACE1K" --out-dir results/accuracy 2>&1 | tee "results/logs/acc_${2}_${3}.log" | grep -E "^\[|Error|error" || true
}
if has 1; then
  for m in "granite_1b_dense|$G1" "granite_h1b|$GH1" "granite_micro_dense|$GM" "granite_hmicro|$GHM" "devstral24b_dense|$DEV" "nemo3_nano30b|$NEMO"; do
    IFS='|' read -r name hf <<< "$m"
    args='{}'; [ "$name" = nemo3_nano30b ] && args='{"max_num_seqs":256}'   # Nemotron: only ~467 Mamba state slots fit
    acc vllm "$name" vllm "$args" "$hf" replay,gsm8k
  done
  if [ $SGL_OK = 1 ]; then
    sgl_on
    for m in "granite_hmicro|$GHM" "nemo3_nano30b|$NEMO"; do
      IFS='|' read -r name hf <<< "$m"
      acc sglang "$name" sgl_fp32       '{}'    "$hf" replay,gsm8k,greedy
      acc sglang "$name" sgl_fp32_rerun '{}'    "$hf" greedy
      acc sglang "$name" sgl_bf16       "$BF16" "$hf" replay,gsm8k,greedy
    done
    deactivate
  else echo "[skip] SGLang accuracy: no /workspace/sgl"; fi
  stop
  python analyze_accuracy.py results/accuracy | tee results/accuracy/summary.txt
fi

# ---- 6. model check: retention on a fine grid around the predicted capacity (vLLM default)
if has 6; then
  for m in "granite_micro_dense|$GM|40,48,56,64,72,80,96" "granite_hmicro|$GHM|320,384,416,448,480,512,576" \
           "devstral24b_dense|$DEV|8,12,16,20,24,28" "nemo3_nano30b|$NEMO|256,288,320,352,384,416,448"; do
    IFS='|' read -r name hf grid <<< "$m"; out="results/traces/ret_fine/ret_${name}_fine.csv"
    if [ -s "${out%.csv}_summary.json" ]; then echo "[skip] fine retention $name"; continue; fi
    echo; echo "======== [6] retention, fine grid ($grid): $name ========"
    stop
    python retention_bench.py --engine vllm --model "$hf" --variant default --engine-args '{}' --trace "$TRACE1K" \
      --sessions-list "$grid" --out "$out" 2>&1 | tee "results/logs/ret_fine_${name}.log" | grep -E "^N=|\[info\]|Error|error|wrote" || true
  done
fi

# ---- 2. branching agents
br() {  # engine name variant args hf
  local out="results/branch/br_${2}_${3}.csv"
  if [ -s "$out" ]; then echo "[skip] branch $2 / $3"; return; fi
  echo; echo "======== [2] branching: $2 / $3 ($1) ========"
  stop
  python branch_bench.py --engine "$1" --model "$5" --variant "$3" --engine-args "$4" --trace "$TRACE" --out "$out" \
    2>&1 | tee "results/logs/br_${2}_${3}.log" | grep -E "^back|^ *[0-9]+ |\[info\]|Error|error|wrote" || true
}
if has 2; then
  br vllm granite_micro_dense default '{}'    "$GM"
  br vllm granite_hmicro      default '{}'    "$GHM"
  br vllm granite_hmicro      fine    "$FINE" "$GHM"
  br vllm devstral24b_dense   default '{}'    "$DEV"
  br vllm nemo3_nano30b       default '{}'    "$NEMO"
  br vllm nemo3_nano30b       fine    "$FINE" "$NEMO"
  if [ $SGL_OK = 1 ]; then
    sgl_on
    br sglang granite_hmicro sgl_default '{}'     "$GHM"
    br sglang granite_hmicro sgl_exact   "$NOBUF" "$GHM"
    br sglang nemo3_nano30b  sgl_default '{}'     "$NEMO"
    br sglang nemo3_nano30b  sgl_exact   "$NOBUF" "$NEMO"
    deactivate
  fi
  stop
fi

# ---- 3. longer sessions: memory per session at 64K and 128K
if has 3; then
  for m in "granite_micro_dense|$GM" "granite_hmicro|$GHM" "devstral24b_dense|$DEV" "nemo3_nano30b|$NEMO"; do
    IFS='|' read -r name hf <<< "$m"; out="results/capacity_long/${name}.csv"
    if [ -s "$out" ]; then echo "[skip] capacity $name"; continue; fi
    echo; echo "======== [3] memory per session at 64K/128K: $name ========"
    stop
    python vllm_capacity.py --model "$hf" --lengths 65536,131072 --out "$out" 2>&1 | tee "results/logs/cap_long_${name}.log" | grep -E "MiB|run|Error|error" || true
  done
fi

# ---- 4. tool-pause sweep (agents per GPU, Granite-Micro pair, vLLM)
if has 4; then
  for p in 1 10 30; do
    for m in "granite_micro_dense|$GM" "granite_hmicro|$GHM"; do
      IFS='|' read -r name hf <<< "$m"
      echo; echo "======== [4] agents per GPU, median tool pause ${p}s: $name ========"
      stop
      python concurrency_bench.py --model "$hf" --tag "$name" --variant "trace_pause${p}" --trace "$TRACE" --turns 20 \
        --tool-median-s "$p" --tool-max-s "$((p * 10))" --agents 64,128,256 --run-timeout 3600 --request-timeout 1800 \
        --stall-s 900 --out-dir results/traces/conc_pause 2>&1 | tee -a "results/traces/conc_pause_${name}_${p}s_client.log" | grep -E "agents|Error|error" || true
    done
  done
  stop
  python analyze_concurrency.py results/traces/conc_pause --out results/traces/conc_pause/summary.csv | tee results/traces/conc_pause/summary.txt
fi

# ---- 5. memory-budget sweep (paused-session retention, Granite-Micro pair, vLLM)
if has 5; then
  for u in 0.55; do
    for m in "granite_micro_dense|$GM" "granite_hmicro|$GHM"; do
      IFS='|' read -r name hf <<< "$m"; out="results/traces/ret_mem/ret_${name}_util${u}.csv"
      if [ -s "${out%.csv}_summary.json" ]; then echo "[skip] retention $name util $u"; continue; fi
      echo; echo "======== [5] retention with gpu_util $u: $name ========"
      stop
      python retention_bench.py --engine vllm --model "$hf" --variant "util$u" --engine-args '{}' --gpu-util "$u" \
        --trace "$TRACE1K" --sessions-list 16,32,64,128,256,512 --out "$out" 2>&1 | tee "results/logs/ret_mem_${name}_${u}.log" \
        | grep -E "^N=|\[info\]|Error|error|wrote" || true
    done
  done
fi
stop
tar czf results_final.tar.gz results/traces/ret_fine results/accuracy results/branch results/capacity_long results/traces/conc_pause results/traces/ret_mem results/logs 2>/dev/null
echo "Done. Download results_final.tar.gz"
