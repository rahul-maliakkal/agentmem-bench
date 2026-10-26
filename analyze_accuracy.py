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


# ---- greedy output equality across settings of the same model
def _load_greedy(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[r["item"]] = r["text"]
    return out


def _prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


greedy = {}
for f in sorted(glob.glob(os.path.join(d, "*_greedy.jsonl"))):
    stem = os.path.basename(f)[:-len("_greedy.jsonl")]
    s = json.load(open(os.path.join(d, stem + "_summary.json")))
    greedy.setdefault(s["tag"], {})[s["variant"]] = _load_greedy(f)
if greedy:
    print("\nGreedy output equality (same prompts, temperature 0):")
    print(f"{'model':22} {'A':14} {'B':14} {'n':>4} {'identical':>9} {'mean shared prefix':>19}")
    for tag, vs in greedy.items():
        names = sorted(vs)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                A, B = vs[names[i]], vs[names[j]]
                keys = [k for k in A if A[k] is not None and B.get(k) is not None]
                if not keys:
                    continue
                same = sum(A[k] == B[k] for k in keys)
                frac = [(_prefix(A[k], B[k]) / max(1, max(len(A[k]), len(B[k])))) for k in keys]
                print(f"{tag:22} {names[i]:14} {names[j]:14} {len(keys):4d} {100*same/len(keys):8.1f}% "
                      f"{100*sum(frac)/len(frac):18.1f}%")
    print("Shared prefix: share of the output (in characters) before the two settings first differ.")
