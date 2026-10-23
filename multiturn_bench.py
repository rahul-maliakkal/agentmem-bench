#!/usr/bin/env python3
"""
Multi-turn agent replay: how much of each resumed turn does vLLM recompute?

Simulates agent sessions on one engine with prefix caching ON:
  turn 0 : [shared system prompt][task]                  -> model emits a tool call
  turn i : [... full history ...][tool output_i]         -> model emits next tool call

For every resumed turn it records:
  prompt_len       tokens in the full history sent to the engine
  cached_tokens    tokens vLLM reused from its prefix cache (RequestOutput.num_cached_tokens)
  recomputed       prompt_len - cached_tokens
  new_tokens       tokens that genuinely had to be computed (= this turn's tool output)
  wasted           recomputed - new_tokens   <- the cost of coarse state checkpoints
  resume_ms        wall time of the prefill + first token for the resumed turn

On a dense model `wasted` should stay below one KV block (~16 tokens). On hybrid
models vLLM can only restore SSM state at its (much larger) block boundaries,
so `wasted` is expected to approach the block size (528 / 4176 tokens).

Sessions run one at a time, so there is no eviction pressure: this isolates
checkpoint granularity. Identical lengths are used for every model/variant
(seeded), so results are directly comparable.

Usage:
  python multiturn_bench.py --model Qwen/Qwen3.5-4B --variant default \
      --engine-args '{}' --out results/mt_qwen35_4b_default.csv
  python multiturn_bench.py --dry-run --block 528 --out /tmp/fake.csv   # no GPU, tests the pipeline
"""
import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
import time


# ---------------------------------------------------------------- workload
def make_sessions(n_sessions, turns, sys_len, task_len, gen_len, tool_median,
                  tool_sigma, tool_min, tool_max, max_ctx, seed):
    """Lengths only; token ids are drawn later per model vocab."""
    rng = random.Random(seed)
    sessions = []
    for s in range(n_sessions):
        ctx = sys_len + task_len
        tl = []
        for _ in range(turns):
            t = int(min(tool_max, max(tool_min, rng.lognormvariate(math.log(tool_median), tool_sigma))))
            if ctx + gen_len + t + gen_len + 64 > max_ctx:
                break
            tl.append(t)
            ctx += gen_len + t
        sessions.append({"session": s, "task_len": task_len, "tools": tl})
    return sessions


class Tokens:
    def __init__(self, vocab, seed):
        self.rng = random.Random(seed)
        self.lo, self.hi = 1000, max(1001, min(vocab, 100000) - 1000)

    def take(self, n):
        return [self.rng.randrange(self.lo, self.hi) for _ in range(n)]


# ---------------------------------------------------------------- engines
class VllmEngine:
    def __init__(self, model, max_model_len, gpu_util, engine_args):
        from vllm import LLM, SamplingParams
        try:
            from vllm.inputs import TokensPrompt
        except ImportError:
            TokensPrompt = dict
        self.TP, self.SP = TokensPrompt, SamplingParams
        args = dict(model=model, max_model_len=max_model_len, gpu_memory_utilization=gpu_util,
                    enable_prefix_caching=True, trust_remote_code=True)
        args.update(engine_args)
        self.llm = LLM(**args)
        self.vocab = self.llm.get_tokenizer().vocab_size
        self.warned = False

    def run(self, ids, max_tokens):
        sp = self.SP(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)
        t0 = time.perf_counter()
        out = self.llm.generate([self.TP(prompt_token_ids=ids)], sp, use_tqdm=False)[0]
        ms = (time.perf_counter() - t0) * 1e3
        cached = getattr(out, "num_cached_tokens", None)
        if cached is None and not self.warned:
            print("[warn] RequestOutput.num_cached_tokens missing: only timings will be recorded", file=sys.stderr)
            self.warned = True
        return list(out.outputs[0].token_ids), cached, ms


class FakeEngine:
    """Simulates block-granular state restore, to test the pipeline without a GPU."""

    def __init__(self, block):
        self.block, self.vocab, self.cache = block, 150000, set()

    def run(self, ids, max_tokens):
        cached = 0
        for end in range(self.block, len(ids) + 1, self.block):
            if tuple(ids[:end]) in self.cache:
                cached = end
        full = ids + [7] * max_tokens
        for end in range(self.block, len(full) + 1, self.block):
            self.cache.add(tuple(full[:end]))
        ms = 0.02 * (len(ids) - cached) + 5
        return [7] * max_tokens, cached, ms


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--variant", default="default")
    ap.add_argument("--engine-args", default="{}", help="JSON dict passed to vllm.LLM(...)")
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--sessions", type=int, default=6)
    ap.add_argument("--turns", type=int, default=15)
    ap.add_argument("--sys-len", type=int, default=4000, help="shared system prompt + tool catalog")
    ap.add_argument("--task-len", type=int, default=500)
    ap.add_argument("--gen-len", type=int, default=150, help="tokens per tool call the model emits")
    ap.add_argument("--tool-median", type=int, default=800)
    ap.add_argument("--tool-sigma", type=float, default=1.0)
    ap.add_argument("--tool-min", type=int, default=30)
    ap.add_argument("--tool-max", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--block", type=int, default=528, help="dry-run only: simulated block size")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.dry_run:
        eng, model = FakeEngine(a.block), f"fake-block{a.block}"
    else:
        if not a.model:
            sys.exit("--model is required unless --dry-run")
        try:
            eng = VllmEngine(a.model, a.max_model_len, a.gpu_util, json.loads(a.engine_args))
        except (TypeError, ValueError) as e:
            sys.exit(f"[skip] engine rejected args {a.engine_args}: {e}")
        model = a.model

    sessions = make_sessions(a.sessions, a.turns, a.sys_len, a.task_len, a.gen_len, a.tool_median,
                             a.tool_sigma, a.tool_min, a.tool_max, a.max_model_len, a.seed)
    toks = Tokens(eng.vocab, a.seed)
    system = toks.take(a.sys_len)  # shared across sessions, like a real agent framework

    # warm up kernels / CUDA graphs on an unrelated prompt
    eng.run(Tokens(eng.vocab, 999).take(256), 4)

    rows = []
    for s in sessions:
        hist = system + toks.take(s["task_len"])
        new_tokens = s["task_len"]  # turn 0: system prompt may be cached, the task is new
        for turn in range(len(s["tools"]) + 1):
            # 1) resume: prefill the history + first token (this is what the agent waits for)
            _, cached, resume_ms = eng.run(hist, 1)
            row = {"model": model, "variant": a.variant, "session": s["session"], "turn": turn,
                   "prompt_len": len(hist), "new_tokens": new_tokens,
                   "cached_tokens": cached if cached is not None else "",
                   "resume_ms": round(resume_ms, 2)}
            if cached is not None:
                rec = len(hist) - cached
                row["recomputed"] = rec
                row["wasted"] = max(0, rec - new_tokens)
            rows.append(row)
            # 2) the model writes its tool call (decode); its KV/state now exists on the GPU
            gen, _, _ = eng.run(hist, a.gen_len)
            hist = hist + gen
            # 3) the tool returns; its output is appended for the next turn
            if turn < len(s["tools"]):
                t = s["tools"][turn]
                hist = hist + toks.take(t)
                new_tokens = t
        last = [r for r in rows if r["session"] == s["session"]]
        w = [r["wasted"] for r in last if r.get("wasted") != "" and r["turn"] > 0 and "wasted" in r]
        print(f"session {s['session']}: {len(last)} turns, final ctx {len(hist):,}, "
              f"mean wasted/turn {statistics.mean(w):.0f}" if w else f"session {s['session']} done", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    keys = ["model", "variant", "session", "turn", "prompt_len", "new_tokens", "cached_tokens",
            "recomputed", "wasted", "resume_ms"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
