#!/usr/bin/env python3
"""
Recompute cost: how long it takes to re-prefill a dropped session's context.

Runs single-request prefill (max_tokens=1, prefix caching OFF) at a range of
context lengths, plus an optional batched run to see throughput under load.

Usage:
  python prefill_bench.py --model Qwen/Qwen3.5-4B --lengths 1000,4000,16000,64000 \
      --out results/qwen35_4b_prefill.csv
"""
import argparse
import csv
import os
import random
import statistics
import time

from vllm import LLM, SamplingParams

try:
    from vllm.inputs import TokensPrompt
except ImportError:  # older vLLM
    TokensPrompt = dict


def make_prompt(length, vocab, rng):
    # Random ids from a safe middle range of the vocab (avoid special tokens).
    lo, hi = 1000, max(1001, min(vocab, 100000) - 1000)
    return TokensPrompt(prompt_token_ids=[rng.randrange(lo, hi) for _ in range(length)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", default="1000,2000,4000,8000,16000,32000,64000")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--batch", type=int, default=0, help="also time a batch of N prompts per length")
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    lengths = [int(x) for x in a.lengths.split(",")]
    llm = LLM(model=a.model, max_model_len=max(lengths) + 16, gpu_memory_utilization=a.gpu_util,
              enable_prefix_caching=False, trust_remote_code=True)
    vocab = llm.get_tokenizer().vocab_size
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    rng = random.Random(0)

    llm.generate([make_prompt(512, vocab, rng)], sp, use_tqdm=False)  # warmup

    rows = []
    for L in lengths:
        ts = []
        for _ in range(a.repeats):
            p = make_prompt(L, vocab, rng)  # fresh tokens each time: no accidental reuse
            t0 = time.perf_counter()
            llm.generate([p], sp, use_tqdm=False)
            ts.append((time.perf_counter() - t0) * 1e3)
        med = statistics.median(ts)
        row = {"model": a.model, "context": L, "batch": 1, "median_ms": round(med, 2),
               "tokens_per_s": round(L / (med / 1e3))}
        rows.append(row)
        print(f"L={L:>7}  single  {med:9.1f} ms  ({row['tokens_per_s']:,} tok/s)")

        if a.batch > 1:
            ps = [make_prompt(L, vocab, rng) for _ in range(a.batch)]
            t0 = time.perf_counter()
            llm.generate(ps, sp, use_tqdm=False)
            ms = (time.perf_counter() - t0) * 1e3
            rows.append({"model": a.model, "context": L, "batch": a.batch, "median_ms": round(ms, 2),
                         "tokens_per_s": round(L * a.batch / (ms / 1e3))})
            print(f"L={L:>7}  batch{a.batch:<3} {ms:9.1f} ms")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
