"""Summarize results/accuracy2/*_summary.json (needle, long-context replay, HumanEval, cache check)."""
import csv, glob, json, os, statistics, sys
from collections import defaultdict

d = sys.argv[1] if len(sys.argv) > 1 else "results/accuracy2"
S = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "*_summary.json")))]
f1 = lambda v: "-" if v is None else f"{100*v:.1f}"

print("Long-context retrieval (multi-key needle in real tool output), accuracy % by context length")
lens = sorted({int(k) for s in S for k in (s.get("needle") or {}).get("by_length", {})})
print(f"{'model':22} {'variant':12} " + " ".join(f"{str(L//1000)+'K':>6}" for L in lens) + f" {'all':>6} {'distr.':>6}")
for s in S:
    n = s.get("needle")
    if not n:
        continue
    rows = list(csv.DictReader(open(os.path.join(d, f"{s['tag']}_{s['variant']}_needle.csv"))))
    dist = statistics.mean(int(r["distractor"]) and not int(r["correct"]) for r in rows)
    print(f"{s['tag']:22} {s['variant']:12} " + " ".join(f"{f1(n['by_length'].get(str(L))):>6}" for L in lens)
          + f" {f1(n['accuracy']):>6} {f1(dist):>6}")
print("distr.: answered with a distractor key's value instead.\n")

print("By depth (all lengths), accuracy %")
for s in S:
    if not s.get("needle"):
        continue
    rows = list(csv.DictReader(open(os.path.join(d, f"{s['tag']}_{s['variant']}_needle.csv"))))
    by = defaultdict(list)
    for r in rows:
        by[r["depth"]].append(int(r["correct"]))
    print(f"{s['tag']:22} {s['variant']:12} " + "  ".join(f"{k}: {f1(statistics.mean(v))}" for k, v in sorted(by.items())))

print("\nReplay at long contexts (12K-31K tokens): bits per character of Claude's next reply (lower is better)")
for s in S:
    r = s.get("replaylong")
    if r:
        print(f"{s['tag']:22} {s['variant']:12} pairs {r['pairs']:4d}  bits/char {r['bits_per_char']}")

print("\nHumanEval pass@1 %")
for s in S:
    h = s.get("humaneval")
    if h:
        print(f"{s['tag']:22} {s['variant']:12} {f1(h['pass_at_1']):>6} +- {f1(h['stderr'])}  (n={h['n']}, errors={h['errors']})")

print("\nCache check: same prompt cold (cache cleared) vs warm (resumed from cache), greedy, one request at a time")
print(f"{'model':22} {'variant':12} {'n':>4} {'identical':>9} {'shared':>7} {'med |dlp|':>10} {'max |dlp|':>10} {'cached cold/warm':>17} {'reset':>6}")
for s in S:
    c = s.get("cachecheck")
    if c:
        print(f"{s['tag']:22} {s['variant']:12} {c['n']:4d} {f1(c['identical']):>8}% {f1(c['mean_shared_prefix']):>6}% "
              f"{c['median_first_lp_diff']!s:>10} {c['max_first_lp_diff']!s:>10} {str(c['median_cached_cold'])+'/'+str(c['median_cached_warm']):>17} {str(c['reset_ok']):>6}")
print("|dlp|: difference in the log-probability of the first generated token. shared: share of output before the first difference.")
