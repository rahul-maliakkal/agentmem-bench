"""Summarize results/accuracy/*_summary.json into one table."""
import glob
import json
import os
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "results/accuracy"
rows = []
for f in sorted(glob.glob(os.path.join(d, "*_summary.json"))):
    s = json.load(open(f))
    r, g = s.get("replay") or {}, s.get("gsm8k") or {}
    rows.append((s["tag"], s["variant"], r.get("pairs"), r.get("nats_per_token"), r.get("bits_per_char"),
                 g.get("accuracy"), g.get("stderr"), g.get("n")))
print(f"{'model':22} {'variant':10} {'pairs':>5} {'nats/tok':>9} {'bits/char':>9} {'GSM8K':>7} {'+-':>6} {'n':>4}")
fmt = lambda v, p: "-" if v is None else f"{v:.{p}f}"
for t, v, n, npt, bpc, acc, se, gn in rows:
    print(f"{t:22} {v:10} {n or 0:5d} {fmt(npt,4):>9} {fmt(bpc,4):>9} {fmt(acc and 100*acc,1):>7} {fmt(se and 100*se,1):>6} {gn or 0:4d}")
print("\nReplay: log-likelihood of Claude's actual next reply in real SWE-agent sessions (lower is better;"
      " bits/char is comparable across tokenizers). GSM8K: greedy, exact match on the final number.")
