#!/usr/bin/env python3
"""
Summarize multi-turn replay results (results/mt_*.csv).

Per model x variant, over resumed turns (turn >= 1):
  new tokens/turn       what genuinely had to be computed (tool output)
  recomputed/turn       what vLLM actually recomputed
  wasted/turn           recomputed - new  (cost of coarse state checkpoints)
  waste ratio           recomputed / new
  resume ms (p50, p90)  wall time the agent waits to resume
  full-hit turns        % of turns where only the new tokens were computed (wasted < 32)

Usage:
  python analyze_multiturn.py results/mt_*.csv --out results/mt_summary.csv --plot results/mt_waste.png
"""
import argparse
import csv
import statistics
from collections import defaultdict


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = min(len(xs) - 1, max(0, round(q / 100 * (len(xs) - 1))))
    return xs[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--plot")
    a = ap.parse_args()

    groups = defaultdict(list)
    for p in a.csvs:
        with open(p) as f:
            rd = csv.DictReader(f)
            if "turn" not in (rd.fieldnames or []):
                continue  # e.g. mt_summary.csv matched by the glob
            for r in rd:
                if int(r["turn"]) >= 1:
                    groups[(r["model"].split("/")[-1], r["variant"])].append(r)

    rows = []
    for (model, variant), rs in sorted(groups.items()):
        has = all(r["wasted"] != "" for r in rs)
        new = [int(r["new_tokens"]) for r in rs]
        ms = [float(r["resume_ms"]) for r in rs]
        row = {"model": model, "variant": variant, "turns": len(rs),
               "mean_new": round(statistics.mean(new)),
               "resume_ms_p50": round(pct(ms, 50), 1), "resume_ms_p90": round(pct(ms, 90), 1)}
        if has:
            rec = [int(r["recomputed"]) for r in rs]
            wst = [int(r["wasted"]) for r in rs]
            row.update({"mean_recomputed": round(statistics.mean(rec)),
                        "mean_wasted": round(statistics.mean(wst)),
                        "p90_wasted": pct(wst, 90),
                        "waste_ratio": round(sum(rec) / max(1, sum(new)), 2),
                        "full_hit_pct": round(100 * sum(w < 32 for w in wst) / len(wst))})
        rows.append(row)

    keys = ["model", "variant", "turns", "mean_new", "mean_recomputed", "mean_wasted", "p90_wasted",
            "waste_ratio", "full_hit_pct", "resume_ms_p50", "resume_ms_p90"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

    print(f"{'model':<34}{'variant':<12}{'new':>7}{'recomp':>8}{'wasted':>8}{'p90 w':>8}{'ratio':>7}{'hit%':>6}{'p50 ms':>8}{'p90 ms':>8}")
    for r in rows:
        print(f"{r['model'][:33]:<34}{r['variant']:<12}{r['mean_new']:>7}{str(r.get('mean_recomputed','-')):>8}"
              f"{str(r.get('mean_wasted','-')):>8}{str(r.get('p90_wasted','-')):>8}{str(r.get('waste_ratio','-')):>7}"
              f"{str(r.get('full_hit_pct','-')):>6}{r['resume_ms_p50']:>8}{r['resume_ms_p90']:>8}")
    print(f"\nwrote {a.out}")
    print("Read: 'wasted' = tokens recomputed per resumed turn beyond the tool output itself.")
    print("Dense should be ~0. If hybrids show hundreds/thousands, coarse SSM checkpoints are a real per-turn cost.")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.5))
        labels = [f"{r['model'][:22]}\n{r['variant']}" for r in rows if "mean_wasted" in r]
        new = [r["mean_new"] for r in rows if "mean_wasted" in r]
        wst = [r["mean_wasted"] for r in rows if "mean_wasted" in r]
        x = range(len(labels))
        ax.bar(x, new, color="#4C78A8", label="new tokens (necessary)")
        ax.bar(x, wst, bottom=new, color="#E45756", label="recomputed beyond necessary")
        ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("tokens prefilled per resumed turn"); ax.set_title("Work to resume an agent turn")
        ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(a.plot, dpi=150)
        print(f"wrote {a.plot}")


if __name__ == "__main__":
    main()
