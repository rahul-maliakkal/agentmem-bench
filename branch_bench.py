#!/usr/bin/env python3
"""
Branching agents: how much of the history is reused when an agent resumes from an EARLIER turn.

Agents often go back: retry the last tool call after an error, try another branch of a plan,
or backtrack a few steps (tree search, best-of-N). The new request shares the history only up
to an earlier turn. A dense KV cache can reuse any page of that prefix; a hybrid model can only
resume from a point where the engine saved a full copy of the recurrent state.

For each real SWE-agent session we replay K turns normally (so the engine caches what it would
in a real run), then send branch requests that go back d turns (d = 0 retries the latest tool
call with a different result): prompt = history up to the model's reply at turn K-d, followed by
a different tool output. We record how many prompt tokens came from the cache.

  reusable  = length of the shared prefix (history up to and including that reply)
  cached    = tokens the engine reused
  reused    = cached / reusable

Usage:
  python branch_bench.py --engine vllm   --model ibm-granite/granite-4.0-h-micro --variant default --out ...
  python branch_bench.py --engine sglang --model ... --variant sgl_exact --engine-args '{"mamba_radix_cache_strategy":"no_buffer","disable_overlap_schedule":true}' --out ...
"""
import argparse
import csv
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["vllm", "sglang"], default="vllm")
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", default="default")
    ap.add_argument("--engine-args", default="{}")
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--mem-frac", type=float, default=0.80)
    ap.add_argument("--trace", default="traces/swe_smith.json")
    ap.add_argument("--sessions", type=int, default=10)
    ap.add_argument("--turns", type=int, default=12, help="turns replayed before branching")
    ap.add_argument("--backs", default="0,1,2,4,8", help="how many turns to go back")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    backs = [int(x) for x in a.backs.split(",")]

    if a.dry_run:
        from multiturn_bench import FakeEngine
        eng = FakeEngine(528)
    elif a.engine == "vllm":
        from multiturn_bench import VllmEngine
        eng = VllmEngine(a.model, a.max_model_len, a.gpu_util, json.loads(a.engine_args))
    else:
        from sglang_multiturn import SglEngine
        eng = SglEngine(a.model, a.max_model_len, a.mem_frac, json.loads(a.engine_args))
    from traces import load_trace, describe
    system, sessions = load_trace(a.trace, eng.tokenizer, a.max_model_len, None, None)
    sessions = [s for s in sessions if len(s["turns"]) >= a.turns + 1]
    rng = random.Random(a.seed)
    rng.shuffle(sessions)
    sessions = sessions[:a.sessions]
    print("[info] " + describe(system, sessions), flush=True)
    pool = [t["tool"] for s in sessions for t in s["turns"]]  # alternative tool outputs come from other turns

    eng.run(list(range(1000, 1256)), 4)  # warm-up
    rows = []
    for s in sessions:
        hist = system + list(s["task"])
        cut = []  # history length right after the model's reply at each turn
        for turn in range(a.turns):
            g = s["turns"][turn]["gen_len"]
            gen, _, _ = eng.run(hist, g)
            hist = hist + gen
            cut.append(list(hist))
            hist = hist + s["turns"][turn]["tool"]
            eng.run(hist, 1)  # the next turn's resume, as in a real run
        for d in backs:
            k = a.turns - 1 - d
            if k < 0:
                continue
            prefix = cut[k]
            alt = rng.choice(pool)
            if len(prefix) + len(alt) + 64 > a.max_model_len:
                alt = alt[: max(16, a.max_model_len - len(prefix) - 64)]
            _, cached, ms = eng.run(prefix + alt, 1)
            c = cached or 0
            rows.append({"model": a.model, "variant": a.variant, "engine": a.engine, "session": s["session"],
                         "back": d, "reusable": len(prefix), "cached": c,
                         "reused": round(min(1.0, c / len(prefix)), 4), "recomputed": len(prefix) + len(alt) - c,
                         "ms": round(ms, 1)})
        last = [r for r in rows if r["session"] == s["session"]]
        print(f"session {s['session']}: " + ", ".join(f"back {r['back']}: {100*r['reused']:.0f}%" for r in last), flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nback  mean reused  mean recomputed tokens")
    for d in backs:
        r = [x for x in rows if x["back"] == d]
        if r:
            print(f"{d:4d}  {100*statistics.mean(x['reused'] for x in r):10.1f}%  {statistics.mean(x['recomputed'] for x in r):10.0f}")
    print(f"wrote {a.out}")
    if hasattr(eng, "close"):
        eng.close()


if __name__ == "__main__":
    main()
