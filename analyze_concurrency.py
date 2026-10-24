#!/usr/bin/env python3
"""
Summarize concurrency runs: one row per model x variant x number of agents,
plus the largest agent count each configuration served within a latency target.

Usage:
  python analyze_concurrency.py results/conc --out results/conc/summary.csv --plot results/conc/concurrency.png
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

COLS = ["tag", "variant", "agents", "status", "agents_finished", "agents_unfinished", "turns_per_s", "gen_tok_per_s",
        "latency_p50_s", "latency_p90_s", "jct_p50_s", "jct_p90_s", "mean_ttft_s", "preemptions",
        "prefix_hit_rate", "reused_token_frac", "peak_cache_usage", "peak_waiting", "peak_gpu_mem_mib"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--plot")
    ap.add_argument("--p90-target-s", type=float, default=10.0,
                    help="latency target for 'agents served': p90 time per model call")
    a = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(os.path.join(a.dir, "*_summary.json"))):
        with open(p) as f:
            s = json.load(f)
        base = os.path.basename(p)[: -len("_summary.json")]
        tag, _, n = base.rpartition("_n")
        s["tag"] = tag[: -len("_" + s["variant"])] if tag.endswith("_" + s["variant"]) else tag
        rows.append(s)
    if not rows:
        raise SystemExit(f"no *_summary.json in {a.dir}")
    rows.sort(key=lambda r: (r["tag"], r["variant"], r["agents"]))

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    def fmt(v, nd=2):
        return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))

    hdr = f"{'config':<30}{'agents':>7}{'turns/s':>9}{'p90 lat':>9}{'p90 job':>9}{'ttft':>8}{'preempt':>9}{'reuse':>7}{'cache%':>8}{'unfin':>7}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{(r['tag'] + '/' + r['variant'])[:29]:<30}{r['agents']:>7}{fmt(r['turns_per_s']):>9}"
              f"{fmt(r['latency_p90_s']):>9}{fmt(r['jct_p90_s'], 0):>9}{fmt(r['mean_ttft_s'], 3):>8}"
              f"{fmt(r['preemptions'], 0):>9}{fmt(r['reused_token_frac']):>7}"
              f"{fmt(100 * r['peak_cache_usage'] if r['peak_cache_usage'] is not None else None, 0):>8}"
              f"{r['agents_unfinished']:>7}" + ("   STALLED (server stopped responding)" if r.get("status") == "stalled" else ""))

    served = defaultdict(int)
    for r in rows:
        okay = (r.get("status", "ok") == "ok" and r["agents_unfinished"] == 0 and r["latency_p90_s"] is not None
                and r["latency_p90_s"] <= a.p90_target_s)
        if okay:
            served[(r["tag"], r["variant"])] = max(served[(r["tag"], r["variant"])], r["agents"])
    print(f"\nLargest agent count served with p90 call latency <= {a.p90_target_s:.0f} s and no unfinished agents:")
    for (tag, var), n in sorted(served.items()):
        print(f"  {tag}/{var}: {n}")
    print("\nCompare dense vs hybrid at the same size: turns/s and p90 latency as agents grow,")
    print("when preemptions start, and peak cache use. 'reuse' = share of prompt tokens served from cache.")

    if a.plot:
        try:
            import matplotlib
        except ImportError:
            print("matplotlib not installed; skipping the plot")
            return
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        groups = defaultdict(list)
        for r in rows:
            groups[f"{r['tag']}/{r['variant']}"].append(r)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for name, rs in groups.items():
            x = [r["agents"] for r in rs]
            axes[0].plot(x, [r["turns_per_s"] for r in rs], marker="o", label=name)
            axes[1].plot(x, [r["latency_p90_s"] for r in rs], marker="o", label=name)
            axes[2].plot(x, [r["preemptions"] or 0 for r in rs], marker="o", label=name)
        for ax, t, y in ((axes[0], "Throughput", "agent turns per second"),
                         (axes[1], "p90 latency per model call", "seconds"),
                         (axes[2], "Preemptions (memory pressure)", "count")):
            ax.set_xscale("log", base=2); ax.set_title(t); ax.set_xlabel("concurrent agents"); ax.set_ylabel(y)
            ax.grid(alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
        axes[0].legend(fontsize=7, frameon=False)
        fig.tight_layout(); fig.savefig(a.plot, dpi=150)
        print(f"wrote {a.plot}")


if __name__ == "__main__":
    main()
