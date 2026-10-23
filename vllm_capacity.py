#!/usr/bin/env python3
"""
Measure what vLLM actually allocates per session, by sweeping max_model_len
and reading vLLM's own startup report:
  "Available KV cache memory: X GiB"
  "Maximum concurrency for N tokens per request: Y x"

bytes_per_session(L) ~= available_bytes / concurrency(L)

This includes the Mamba/GDN state, block-size padding (e.g. vLLM rounding the
attention block up to the Mamba page size) and allocator overhead -- the real
number your policy has to work with. Compare it with state_sizes.py.

Each length runs in a fresh subprocess so GPU memory is fully released.

Usage:
  python vllm_capacity.py --model Qwen/Qwen3.5-4B --lengths 2048,8192,32768 \
      --out results/qwen35_4b_capacity.csv
"""
import argparse
import csv
import os
import re
import subprocess
import sys

CHILD = r"""
import sys
from vllm import LLM
LLM(model=sys.argv[1], max_model_len=int(sys.argv[2]),
    gpu_memory_utilization=float(sys.argv[3]), enforce_eager=True,
    enable_prefix_caching=False, trust_remote_code=True)
print("CHILD_DONE", flush=True)
"""

RE_AVAIL = re.compile(r"Available KV cache memory:\s*([\d.]+)\s*GiB")
RE_CONC = re.compile(r"Maximum concurrency for ([\d,]+) tokens per request:\s*([\d.]+)x")
RE_TOKENS = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")
RE_BLOCK = re.compile(r"Setting attention block size to (\d+) tokens")


def run_one(model, length, util, logdir):
    p = subprocess.run([sys.executable, "-c", CHILD, model, str(length), str(util)],
                       capture_output=True, text=True)
    log = p.stdout + p.stderr
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, f"capacity_{length}.log"), "w") as f:
        f.write(log)
    avail = RE_AVAIL.findall(log)
    conc = RE_CONC.findall(log)
    toks = RE_TOKENS.findall(log)
    block = RE_BLOCK.findall(log)
    if p.returncode != 0 or not avail or not conc:
        print(f"[warn] L={length}: could not parse (exit {p.returncode}); see {logdir}/capacity_{length}.log",
              file=sys.stderr)
        tail = [l for l in log.strip().splitlines() if l.strip()][-6:]
        print("       last log lines:\n         " + "\n         ".join(tail), file=sys.stderr)
        return None
    avail_gib = float(avail[-1])
    concurrency = float(conc[-1][1])
    return {
        "max_model_len": length,
        "available_kv_GiB": avail_gib,
        "max_concurrency": concurrency,
        "bytes_per_session": avail_gib * 2**30 / concurrency,
        "MiB_per_session": avail_gib * 1024 / concurrency,
        "kv_cache_tokens": int(toks[-1].replace(",", "")) if toks else "",
        "attn_block_size": int(block[-1]) if block else "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", default="2048,4096,8192,16384,32768")
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--logdir", default="results/logs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = []
    for L in [int(x) for x in a.lengths.split(",")]:
        print(f"[run] {a.model} max_model_len={L}", flush=True)
        r = run_one(a.model, L, a.gpu_util, a.logdir)
        if r:
            r["model"] = a.model
            rows.append(r)
            print(f"      {r['MiB_per_session']:.1f} MiB/session, concurrency {r['max_concurrency']:.1f}x"
                  + (f", attn block {r['attn_block_size']}" if r["attn_block_size"] else ""))
    if not rows:
        sys.exit("no lengths parsed -- check the logs; vLLM's log wording may have changed")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # Fit bytes/session = fixed + per_token * L  (least squares)
    if len(rows) >= 2:
        xs = [r["max_model_len"] for r in rows]; ys = [r["bytes_per_session"] for r in rows]
        n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        fixed = my - slope * mx
        print(f"\nfit: fixed ~{fixed/2**20:.1f} MiB/session + {slope/1024:.2f} KiB/token")
        print("compare with state_sizes.py: fixed_state_bytes and kv_bytes_per_token")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
