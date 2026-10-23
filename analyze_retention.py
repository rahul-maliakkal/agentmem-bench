#!/usr/bin/env python3
"""Summarize retention runs (Experiment 8): share of paused sessions whose resume point
survives as the number of paused sessions grows.

Usage: python analyze_retention.py results/ret_*_summary.json --out results/ret_summary.csv
"""
import argparse
import csv
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-target", type=float, default=90.0)
    a = ap.parse_args()
    rows = []
    for p in sorted(a.files):
        with open(p) as f:
            d = json.load(f)
        for lv in d["levels"]:
            rows.append(dict(lv, cache_info=json.dumps(d.get("info", {}))))
    if not rows:
        raise SystemExit("no results")
    keys = sorted({k for r in rows for k in r})
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    hdr = f"{'model':<34}{'variant':<12}{'N':>6}{'full%':>8}{'prompt%':>9}{'sys%':>7}{'reused':>8}{'recomp':>8}"
    print(hdr); print("-" * len(hdr))
    best = {}
    for r in rows:
        name = (r["model"].split("/")[-1][:33], r["variant"])
        print(f"{name[0]:<34}{name[1]:<12}{r['n_sessions']:>6}{r['pct_kept_full']:>8}{r['pct_kept_prompt']:>9}"
              f"{r['pct_kept_sys']:>7}{str(r['mean_reused_frac']):>8}{str(r['mean_recomputed']):>8}")
        if r["pct_kept_full"] >= a.keep_target:
            best[name] = max(best.get(name, 0), r["n_sessions"])
        else:
            best.setdefault(name, 0)
    print(f"\nLargest number of paused sessions with >= {a.keep_target:.0f}% of resume points fully kept:")
    for (m, v), n in sorted(best.items()):
        print(f"  {m} / {v}: {n}")
    print("\nfull% = resume point (system + task + reply) still cached; prompt% = at least the old prompt;")
    print("sys% = at least the shared system prompt; recomp = tokens recomputed per resume.")


if __name__ == "__main__":
    main()
