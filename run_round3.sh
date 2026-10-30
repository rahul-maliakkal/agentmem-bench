#!/usr/bin/env bash
# Round 3 (reviewer follow-ups), about 18-20 hours on one H100, vLLM only. Steps, in the default order:
#   1. Large group at matched precision (FP8 weights + FP8 KV cache for every model):
#        Devstral 24B with FP8 KV (was BF16 KV), and gpt-oss-20b (transformer MoE, ~3.6B active) as the
#        control for Nemotron's sparse experts. Memory per session (2K-128K), agents per GPU (8-256, now
#        including 16), paused-session test on a fine grid, and the single-session reuse replay.      (~6 h)
#   3. Dense Granite-Micro with CPU offload of paused KV cache: paused-session test and agents per GPU. (~2 h)
#   2. Fine-grid agents per GPU with two extra repeats (seeds 1, 2) around each crossover.           (~9 h)
#   5. Close two gaps (~2-2.5 h): Devstral with FP8 KV at 24-48 agents on three seeds, and Nemotron at
#      112/120/128 agents on five seeds. Each seed stops at its first agent count over the 10 s p90 target.
#      Run alone with:  STEPS="5" bash run_round3.sh
#   4. gpt-oss-20b accuracy (GSM8K, HumanEval, long-context needle, long replay).                    (~1 h)
# Run inside tmux, from the vLLM environment:  tmux new -s r3 ; bash run_round3.sh
# STEPS="1 3" runs only those steps. Finished pieces are skipped on restart.
# Upload results_round3.tar.gz when done (it is also rebuilt after each step, so a partial upload works).
set -uo pipefail
cd "$(dirname "$0")"
export HF_HOME=${HF_HOME:-/workspace/hf}
STEPS=${STEPS:-"1 3 2 4"}
has() { case " $STEPS " in *" $1 "*) return 0;; *) return 1;; esac; }
OUT=results/round3
mkdir -p $OUT/capacity $OUT/conc $OUT/ret $OUT/mt $OUT/acc results/logs
TRACE=traces/swe_smith.json; TRACE1K=traces/swe_smith_1500.json
G1=ibm-granite/granite-4.0-1b; GH1=ibm-granite/granite-4.0-h-1b; GM=ibm-granite/granite-4.0-micro; GHM=ibm-granite/granite-4.0-h-micro
DEV=mistralai/Devstral-Small-2-24B-Instruct-2512; NEMO=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8
GPTOSS_BF16=${GPTOSS_BF16:-unsloth/gpt-oss-20b-BF16}; GPTOSS_MXFP4=openai/gpt-oss-20b
DEV_FP8KV='{"kv_cache_dtype":"fp8"}'
NEMO_ARGS='{"max_num_seqs":256}'
stop() {  # kill any server and wait until the GPU memory is actually released (up to 3 minutes)
  pkill -f "vllm serve" 2>/dev/null; pkill -f EngineCore 2>/dev/null; pkill -f "retention_bench.py|multiturn_bench.py|vllm_capacity.py" 2>/dev/null
  for _ in $(seq 1 36); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "$used" ] && [ "$used" -lt 2000 ] && return 0
    sleep 5; pkill -9 -f "vllm serve|EngineCore" 2>/dev/null
  done
  echo "[warn] GPU still has ${used:-?} MiB in use after 3 minutes"
}
pack() { tar czf results_round3.tar.gz $OUT results/logs/r3_* 2>/dev/null; }

exec 9>/tmp/agentmem_runner.lock
flock -n 9 || { echo "ERROR: another runner holds the lock. Stop it first."; exit 1; }
if pgrep -f "run_final.sh|run_acc2.sh" >/dev/null; then echo "ERROR: another runner is still running. Stop it first."; exit 1; fi
python -c "from vllm import LLM" 2>results/logs/vllm_check.log || { echo "ERROR: vLLM does not import"; tail -5 results/logs/vllm_check.log; exit 1; }
python -c "import vllm, transformers, torch; print('versions: vllm', vllm.__version__, 'transformers', transformers.__version__, 'torch', torch.__version__)" | tee $OUT/versions.txt
free -g | tee -a $OUT/versions.txt
[ -s "$TRACE" ] || python prepare_traces.py --out "$TRACE" | tee results/logs/prepare.log
[ -s "$TRACE1K" ] || python prepare_traces.py --n 1500 --out "$TRACE1K" | tee results/logs/prepare_1500.log
for hf in $G1 $GH1 $GM $GHM $DEV $NEMO $GPTOSS_BF16 $GPTOSS_MXFP4; do
  python -c "from huggingface_hub import snapshot_download as d; d('$hf')" >/dev/null 2>&1 || echo "[warn] could not pre-download $hf"
done
export HF_HUB_OFFLINE=1

# ---- pick how to serve gpt-oss-20b at the same precision as Nemotron (FP8 weights, FP8 KV cache)
# Tries, in order: BF16 checkpoint quantized to FP8 at load; then the released MXFP4 checkpoint with FP8 KV.
GPTOSS=""; GPTOSS_ARGS=""
pick_gptoss() {
  [ -s $OUT/gptoss_choice.txt ] && { read -r GPTOSS GPTOSS_ARGS < $OUT/gptoss_choice.txt; return; }
  for cand in "$GPTOSS_BF16|{\"quantization\":\"fp8\",\"kv_cache_dtype\":\"fp8\"}|fp8_weights_fp8_kv" \
              "$GPTOSS_MXFP4|{\"kv_cache_dtype\":\"fp8\"}|mxfp4_weights_fp8_kv"; do
    IFS='|' read -r hf args label <<< "$cand"
    stop
    echo "[gpt-oss] trying $hf with $args"
    if python vllm_capacity.py --model "$hf" --lengths 8192 --gpu-util 0.85 --engine-args "$args" \
         --logdir results/logs/r3_gptoss_probe_$label --out $OUT/capacity/probe_$label.csv > results/logs/r3_gptoss_probe_$label.log 2>&1; then
      GPTOSS=$hf; GPTOSS_ARGS=$args
      echo "$GPTOSS $GPTOSS_ARGS" > $OUT/gptoss_choice.txt
      echo "[gpt-oss] using $label: $hf $args" | tee -a $OUT/versions.txt
      return
    fi
    echo "[gpt-oss] $label failed; see results/logs/r3_gptoss_probe_$label.log"; tail -5 results/logs/r3_gptoss_probe_$label.log
  done
  echo "[gpt-oss] ERROR: no working configuration; gpt-oss runs are skipped" | tee -a $OUT/versions.txt
}

cap() {  # name hf args
  local out=$OUT/capacity/$1.csv
  [ -s "$out" ] && { echo "[skip] capacity $1"; return; }
  echo; echo "======== memory per session: $1 ========"; stop
  python vllm_capacity.py --model "$2" --engine-args "$3" --gpu-util 0.85 --lengths 2048,4096,8192,16384,32768,65536,131072 \
    --logdir results/logs/r3_cap_$1 --out "$out" 2>&1 | tee results/logs/r3_cap_$1.log | grep -E "MiB|run|Error|error" || true
}
conc() {  # name hf args variant agents seed [stop_p90]
  echo; echo "======== agents per GPU: $1 / $4 (agents $5, seed $6) ========"; stop
  python concurrency_bench.py --model "$2" --tag "$1" --variant "$4" --engine-args "$3" --trace "$TRACE" --turns 20 \
    --agents "$5" --seed "$6" --stop-p90 "${7:-0}" --run-timeout 3600 --request-timeout 1800 --stall-s 600 --out-dir $OUT/conc \
    2>&1 | tee -a results/logs/r3_conc_$1_$4.log | grep -E "agents|Error|error" || true
}
ret() {  # name hf args variant grid
  local out=$OUT/ret/ret_$1_$4.csv
  [ -s "${out%.csv}_summary.json" ] && { echo "[skip] pause test $1 / $4"; return; }
  echo; echo "======== paused sessions: $1 / $4 ($5) ========"; stop
  python retention_bench.py --engine vllm --model "$2" --variant "$4" --engine-args "$3" --trace "$TRACE1K" \
    --sessions-list "$5" --out "$out" 2>&1 | tee results/logs/r3_ret_$1_$4.log | grep -E "^N=|\[info\]|Error|error|wrote" || true
}
mt() {  # name hf args
  local out=$OUT/mt/mt_$1.csv
  [ -s "$out" ] && { echo "[skip] reuse replay $1"; return; }
  echo; echo "======== single-session reuse replay: $1 ========"; stop
  python multiturn_bench.py --model "$2" --variant default --engine-args "$3" --trace "$TRACE" --trace-sessions 12 \
    --out "$out" 2>&1 | tee results/logs/r3_mt_$1.log | grep -E "session|Error|error|wrote" | tail -3 || true
}

# ---- 1. large group at matched precision
if has 1; then
  cap devstral24b_fp8kv "$DEV" "$DEV_FP8KV"
  conc devstral24b_fp8kv "$DEV" "$DEV_FP8KV" trace 8,16,32,64,128 0
  ret devstral24b_fp8kv "$DEV" "$DEV_FP8KV" fine 16,24,32,40,48,56,64
  conc devstral24b_dense "$DEV" '{}' trace_s0_16 16 0            # the missing 16-agent point at BF16 KV, as in the paper
  pick_gptoss
  if [ -n "$GPTOSS" ]; then
    cap gptoss20b "$GPTOSS" "$GPTOSS_ARGS"
    conc gptoss20b "$GPTOSS" "$GPTOSS_ARGS" trace 8,16,32,64,96,128,192,256 0
    ret gptoss20b "$GPTOSS" "$GPTOSS_ARGS" fine 32,64,96,128,160,192,256,384
    mt gptoss20b "$GPTOSS" "$GPTOSS_ARGS"
  fi
  stop; pack
fi

# ---- 3. dense Granite-Micro with CPU offload of KV cache
if has 3; then
  memg=$(free -g | awk '/Mem:/{print $7}'); size=$(( memg / 2 )); [ "$size" -gt 96 ] && size=96; [ "$size" -lt 16 ] && size=16
  echo "[offload] CPU memory available ${memg} GiB; offloading up to ${size} GiB" | tee -a $OUT/versions.txt
  OFF_A="{\"kv_offloading_size\":$size,\"kv_offloading_backend\":\"native\"}"
  OFF_B="{\"kv_transfer_config\":{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"num_cpu_blocks\":$(( size * 1024 * 1024 / 1280 ))}}}"
  OFF=""
  for cand in "$OFF_A" "$OFF_B"; do   # find which offload flag this vLLM accepts
    stop
    if python retention_bench.py --engine vllm --model "$GM" --variant offload_probe --engine-args "$cand" --trace "$TRACE1K" \
         --sessions-list 16 --out $OUT/ret/probe_offload.csv > results/logs/r3_offload_probe.log 2>&1; then OFF=$cand; break; fi
    echo "[offload] engine args rejected: $cand"; tail -4 results/logs/r3_offload_probe.log
  done
  if [ -n "$OFF" ]; then
    echo "[offload] using $OFF" | tee -a $OUT/versions.txt
    # one process per level, so CPU-cached blocks from a smaller level cannot leak into the next
    for n in 64 128 256 512 1024; do ret granite_micro_dense "$GM" "$OFF" "offload_n$n" "$n"; done
    ret granite_micro_dense "$GM" '{}' "nooffload_timed" 64,128,256       # same test without offload, with resume timing
    ret granite_hmicro "$GHM" '{}' "timed" 64,128,256,384                 # hybrid, with resume timing
    conc granite_micro_dense "$GM" "$OFF" trace_offload 64,128,256 0
  else
    echo "[offload] ERROR: this vLLM accepts neither offload flag; step 3 skipped" | tee -a $OUT/versions.txt
  fi
  stop; pack
fi

# ---- 2. fine grid with repeats around each crossover (seed 0 = the runs already in the paper)
if has 2; then
  for seed in 1 2; do
    conc granite_1b_dense    "$G1"  '{}' "trace_s$seed" 64,80,96,112,128 "$seed"
    conc granite_micro_dense "$GM"  '{}' "trace_s$seed" 64,80,96,112,128 "$seed"
    conc granite_h1b         "$GH1" '{}' "trace_s$seed" 128,160,192,224,256 "$seed"
    conc granite_hmicro      "$GHM" '{}' "trace_s$seed" 128,160,192,224,256 "$seed"
    conc nemo3_nano30b       "$NEMO" '{}' "trace_s$seed" 64,80,96,112,128 "$seed"
    pack
  done
  for seed in 1 2; do conc devstral24b_dense "$DEV" '{}' "trace_s$seed" 8,12,16,24 "$seed"; done
  stop; pack
fi

# ---- 5. close two gaps: Devstral (FP8 KV) on three seeds; Nemotron at 112-128 on five seeds. Each seed stops
#         at its first agent count over the 10 s p90 target (later counts would fail too).
if has 5; then
  conc devstral24b_fp8kv "$DEV" "$DEV_FP8KV" gap_s0 24,40,48 0 10      # seed 0 already has 8,16,32,64,128
  for seed in 1 2; do conc devstral24b_fp8kv "$DEV" "$DEV_FP8KV" "gap_s$seed" 24,32,40,48 "$seed" 10; done
  pack
  conc nemo3_nano30b "$NEMO" '{}' gap_s0 112,120 0 10                  # seeds 0-2 already have 112 and/or 128
  for seed in 1 2; do conc nemo3_nano30b "$NEMO" '{}' "gap_s$seed" 120 "$seed" 10; done
  for seed in 3 4; do conc nemo3_nano30b "$NEMO" '{}' "gap_s$seed" 112,120,128 "$seed" 10; done
  stop; pack
fi

# ---- 4. gpt-oss-20b accuracy, same checks as the other models
if has 4; then
  [ -z "$GPTOSS" ] && pick_gptoss
  if [ -n "$GPTOSS" ] && [ ! -s $OUT/acc/gptoss20b_vllm_summary.json ]; then
    stop
    python accuracy_bench.py --engine vllm --model "$GPTOSS" --tag gptoss20b --variant vllm --engine-args "$GPTOSS_ARGS" \
      --tasks replay,gsm8k --trace "$TRACE1K" --out-dir $OUT/acc 2>&1 | tee results/logs/r3_acc1_gptoss.log | grep -E "^\[|Error|error" || true
    stop
    python accuracy2_bench.py --engine vllm --model "$GPTOSS" --tag gptoss20b --variant vllm2 --engine-args "$GPTOSS_ARGS" \
      --tasks needle,replaylong,humaneval --trace "$TRACE1K" --out-dir $OUT/acc 2>&1 | tee results/logs/r3_acc2_gptoss.log | grep -E "^\[|Error|error" || true
  fi
  stop; pack
fi
python analyze_concurrency.py $OUT/conc --out $OUT/conc/summary.csv > $OUT/conc/summary.txt 2>&1 || true
pack
echo "Done. Upload results_round3.tar.gz"
