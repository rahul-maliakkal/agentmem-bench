# AgentMem-Bench

Measures what switching from dense transformers to hybrid Mamba-attention models does to
GPU memory, reuse and capacity when serving AI agents. Agents alternate model calls with
tool calls, so their sessions pause, resume and share prefixes.

Hybrids make a session **cheap to hold**: a small fixed recurrent state plus a small KV
cache. They make it **costly to reuse**: a recurrent state cannot be rewound, so an engine
can only resume from points where it saved a full copy. This benchmark quantifies both
sides on production engines (vLLM and SGLang) with matched dense and hybrid models.

## Models

| Pair | Dense | Hybrid |
|---|---|---|
| Small | `ibm-granite/granite-4.0-1b` | `ibm-granite/granite-4.0-h-1b` |
| Medium | `ibm-granite/granite-4.0-micro` | `ibm-granite/granite-4.0-h-micro` |
| Large | `mistralai/Devstral-Small-2-24B-Instruct-2512` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` |

Tested on one NVIDIA H100 80GB (PCIe Gen5), vLLM 0.32, SGLang 0.5.22.

## Experiments

| # | Question | Script | Runner |
|---|---|---|---|
| 1 | How much GPU memory does one session take as context grows? | `vllm_capacity.py`, `state_sizes.py` | `run_all.sh` |
| 2 | How much of a hybrid session is fixed state, KV cache, padding? | `analyze.py` (from 1) | `run_all.sh` |
| 3 | Offload a paused session to CPU, or rebuild it? | `transfer_bench.py`, `prefill_bench.py` | `run_all.sh` |
| 4–5 | How much work is repeated when an agent resumes (vLLM default and align modes)? | `multiturn_bench.py` | `run_multiturn.sh` |
| 6 | Same, on SGLang (exact and every-256-token checkpoints) | `sglang_multiturn.py` | `run_sglang.sh` |
| 7 | How many concurrent agents does one GPU serve? | `concurrency_bench.py` | `run_concurrency.sh` |

`run_pending.sh` runs experiments 4–7 back to back. Every runner skips work that already
has results, so it is safe to restart.

## Workload

Simulated agent sessions with seeded random token IDs: a shared 4,000-token system prompt,
a 500-token task, model replies of 150 tokens (also 500 and 1,500; 300 in the concurrency
runs), and tool outputs drawn from a lognormal distribution (median 800 tokens, 30–6,000).
Concurrency runs add tool pauses (median 3 s, 0.2–30 s). Memory and repeated work depend
only on token counts and block sizes, not on token content.

## Setup and run

```bash
cd /workspace/agentmem-bench
bash setup.sh                 # installs vLLM if missing, records GPU and versions
tmux new -s bench             # always run inside tmux: a dropped SSH session kills the run
bash run_all.sh               # experiments 1-3
bash run_pending.sh           # experiments 4-7
```

Model weights go to `HF_HOME` (default `/workspace/hf`); budget about 150 GB of disk.
SGLang is installed into its own environment (`/workspace/sgl`) on first use.

Test the multi-turn pipeline without a GPU:
`python multiturn_bench.py --dry-run --block 4176 --out /tmp/x.csv`

## Outputs

- `results/summary.txt`, `results/cost_table.csv`: memory, transfer and rebuild costs.
- `results/mt_summary.txt`: tokens recomputed per resumed turn (`wasted` = beyond the new tokens).
- `results/conc_<suffix>/summary.txt`: throughput, p90 latency, preemptions, cache use and
  the largest agent count served within a 10 s p90 latency per model call. Load levels where
  the server stops responding are recorded as STALLED.

## Notes

- `prefix_match_unit` must divide the attention block size vLLM picks for a hybrid
  (400, 528 or 4,176 tokens here); 16 works for all of them.
- vLLM stores Granite-H recurrent state in 16-bit; `state_sizes.py --ssm-dtype` sets the
  analytic estimate. The measured capacity fit is the ground truth.
