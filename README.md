# Hybrid-model paused-session viability benchmark

Answers one go/no-go question before you build anything:

> For hybrid models, is a paused agent session's state split into a fixed SSM part and a
> growing KV part in a way that makes per-part keep/offload decisions worth it?

| Script | Measures | GPU? |
|---|---|---|
| `state_sizes.py` | Analytic fixed SSM state vs KV per token, from `config.json` | No |
| `vllm_capacity.py` | What vLLM *actually* allocates per session (includes padding) | Yes |
| `transfer_bench.py` | GPU↔CPU offload/reload time: contiguous block vs scattered pages | Yes |
| `prefill_bench.py` | Recompute cost: re-prefill time vs context length | Yes |
| `analyze.py` | Combines the above into the cost table + plot | No |
| `run_all.sh` | Runs everything for all models | Yes |

## 1. Create the pod on RunPod

1. **Storage first:** Storage → New Network Volume, ~150 GB, same region as the GPU you want.
   Model weights live here so you don't re-download after stopping the pod.
2. **Pod:** Pods → Deploy → **1× H100 SXM 80GB** (or H100 PCIe; note which, since PCIe
   bandwidth is one of your measurements). On-demand, not spot — a spot interruption mid-sweep
   wastes the run.
3. **Template:** a recent PyTorch template, or the `vllm/vllm-openai` image. Attach the network
   volume at `/workspace`. Container disk ~40 GB.
4. Start it and open a terminal (web terminal or SSH).

## 2. Run

```bash
cd /workspace
# upload this folder (runpodctl send/receive, scp, or git clone your repo)
cd hybrid-pause-bench
bash setup.sh                 # installs vLLM if missing, logs GPU/PCIe/versions to results/env.txt
huggingface-cli login         # only if a model is gated
tmux new -s bench             # so a dropped connection doesn't kill the run
bash run_all.sh
```

Detach with `Ctrl-b d`, reattach with `tmux attach -t bench`. Expect roughly 2–4 hours
for the three default models (most of it is model loading in the capacity sweep).

When done, download `results.tar.gz`. **Stop the pod** — you're billed per second while it runs;
the network volume keeps the weights (storage is billed separately).

## 3. Read the result

`results/summary.txt` and `results/costs.png`. The go signal:

- **State share crosses 50%** at agent-realistic contexts (a few thousand to tens of thousands of tokens), and
- **"keep st" ≫ "keep all"** — keeping only the fixed state fits many more paused sessions, and
- **reload KV ≪ recompute** — offloading KV is much cheaper than dropping the session.

If the fixed state is tiny everywhere (state share near 0%) the per-part decision doesn't matter
and the paper should pivot to a characterization study.

## 4. Check before trusting the numbers

- **Model IDs.** `run_all.sh` uses `Qwen/Qwen3.5-4B`, `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8`
  and `Qwen/Qwen3-4B` (dense control). Edit `MODELS` to add others (e.g. Nemotron-3-Nano-4B,
  Qwen3.5-122B-A10B on a bigger GPU).
- **State dtype.** `state_sizes.py` assumes an fp32 recurrent state (`--ssm-dtype`). Check what
  your vLLM version uses (`mamba_ssm_cache_dtype`) and re-run with the right value. The
  `vllm_capacity.py` fit (`fixed MiB + KiB/token`) is the ground truth — if it disagrees with the
  analytic numbers, find out why; that gap (e.g. block-size padding) may itself be a finding.
- **Config fields.** If `state_sizes.py` raises `config has none of (...)`, open the model's
  `config.json` and add the field name to the matching `need(...)` call.
- **vLLM log wording.** `vllm_capacity.py` parses vLLM's startup lines. If it warns it couldn't
  parse, look at `results/logs/<model>/capacity_<L>.log` and adjust the regexes at the top.
- **Page size.** `transfer_bench.py --page-kib` should match the real KV page size
  (block size × per-layer KV bytes per token). For hybrids vLLM may enlarge the block a lot —
  `vllm_capacity.py` prints the attention block size it chose.

## What this does not measure yet

Real agent behavior (tool durations, turn counts, concurrency under load). That's the next step:
replay SWE-Bench/BFCL traces against the Continuum vLLM fork with a hybrid model.

## Experiment 2: multi-turn agent replay (`run_multiturn.sh`)

Measures how much of each resumed agent turn vLLM recomputes with prefix caching on.
6 sessions x up to 16 turns; shared 4K-token system prompt; 150-token tool calls;
tool outputs drawn from a lognormal (median 800 tokens, 30-6000). Same lengths for every run.

```bash
tmux new -s mt
bash run_multiturn.sh          # ~30-60 min on one H100
```

Key column: **wasted** = tokens recomputed per resumed turn beyond the new tool output.
Dense should be ~0; hybrids show the cost of block-granular SSM checkpoints.
`fine_a` / `fine_b` try vLLM's finer SSM checkpoint option (the flag name changed across
versions; one of them may be rejected and skipped -- that's expected).

Test the pipeline without a GPU:
`python multiturn_bench.py --dry-run --block 4176 --out /tmp/x.csv`

## Experiment 3: how many agents fit on one GPU (`run_concurrency.sh`)

Runs each Granite model as a vLLM server and replays 8, 32, 64, 128 and 256 agents at once.
Each agent: 10 turns, 300-token tool calls, tool pauses of about 3 s (0.2-30 s), tool outputs of
about 800 tokens, shared 4,000-token system prompt. Hybrids run with default caching and the
finer checkpoint option. The server's /metrics are sampled every second.

```bash
tmux new -s conc
bash run_concurrency.sh        # about 2-3 hours on one H100
```

Read `results/conc_<suffix>/summary.txt` (suffix = date and time of the run, or set SUFFIX=name): throughput, p90 latency, preemptions and peak cache use as the
number of agents grows, and the largest agent count each model serves within a p90 latency of 10 s.
