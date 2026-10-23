#!/usr/bin/env python3
"""
Combine the measurements into the cost table for paused sessions.

For each model and context length:
  memory held  : fixed SSM state vs KV cache
  resume cost  : reload state (contiguous), reload KV (paged-gather),
                 recompute (full re-prefill)
  capacity     : paused sessions that fit in a memory budget if you keep
                 everything vs keep only the fixed state on GPU

Usage:
  python analyze.py --transfer results/transfer.csv \
      --model-run qwen35_4b:results/qwen35_4b_sizes.json:results/qwen35_4b_prefill.csv \
      --model-run qwen3_4b:results/qwen3_4b_sizes.json:results/qwen3_4b_prefill.csv \
      --out results/cost_table.csv --plot results/costs.png
"""
import argparse
import csv
import json
import math


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def transfer_model(rows, layout, direction):
    pts = sorted((float(r["size_MiB"]) * 2**20, float(r["ms"])) for r in rows
                 if r["layout"] == layout and r["direction"] == direction)
    if not pts:
        raise ValueError(f"no transfer rows for {layout}/{direction}")

    def ms_for(nbytes):
        if nbytes <= 0:
            return 0.0
        if nbytes <= pts[0][0]:
            return pts[0][1] * nbytes / pts[0][0]
        if nbytes >= pts[-1][0]:
            return pts[-1][1] * nbytes / pts[-1][0]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= nbytes <= x1:
                t = (math.log(nbytes) - math.log(x0)) / (math.log(x1) - math.log(x0))
                return y0 + t * (y1 - y0)
    return ms_for


def prefill_model(rows):
    pts = sorted((int(r["context"]), float(r["median_ms"])) for r in rows if int(r["batch"]) == 1)

    def ms_for(L):
        if L <= pts[0][0]:
            return pts[0][1] * L / pts[0][0]
        if L >= pts[-1][0]:  # extrapolate with last segment's slope
            (x0, y0), (x1, y1) = pts[-2], pts[-1]
            return y1 + (y1 - y0) / (x1 - x0) * (L - x1)
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= L <= x1:
                return y0 + (y1 - y0) * (L - x0) / (x1 - x0)
    return ms_for, pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transfer", required=True)
    ap.add_argument("--model-run", action="append", required=True,
                    help="name:sizes.json:prefill.csv (repeat per model)")
    ap.add_argument("--contexts", default="1000,2000,4000,8000,16000,32000,64000,128000")
    ap.add_argument("--budget-gib", type=float, default=40.0,
                    help="GPU memory available for paused sessions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--plot")
    a = ap.parse_args()

    tr = load_csv(a.transfer)
    reload_contig = transfer_model(tr, "contiguous", "reload")
    reload_paged = transfer_model(tr, "paged-gather", "reload")
    contexts = [int(x) for x in a.contexts.split(",")]
    budget = a.budget_gib * 2**30

    rows = []
    for spec in a.model_run:
        name, sizes_path, prefill_path = spec.split(":", 2)
        with open(sizes_path) as f:
            s = json.load(f)
        recompute, _ = prefill_model(load_csv(prefill_path))
        fixed = s["fixed_state_bytes"]
        # Prefer what vLLM actually allocates (capacity sweep) over the analytic estimate:
        # engines may store the SSM state or KV cache in a different dtype than assumed.
        cap_path = sizes_path.replace("_sizes.json", "_capacity.csv")
        try:
            pts = [(int(r["max_model_len"]), float(r["bytes_per_session"])) for r in load_csv(cap_path)][-3:]
            if len(pts) >= 2:
                mx = sum(a for a, _ in pts) / len(pts); my = sum(b for _, b in pts) / len(pts)
                slope = sum((a - mx) * (b - my) for a, b in pts) / sum((a - mx) ** 2 for a, _ in pts)
                fixed = max(0.0, my - slope * mx)
                s = dict(s, kv_bytes_per_token=slope, swa_bytes_per_token=0)
                print(f"[{name}] using measured sizes: fixed {fixed/2**20:.1f} MiB, KV {slope/1024:.1f} KiB/token")
        except FileNotFoundError:
            pass
        for L in contexts:
            swa_tokens = min(L, s["swa_window_tokens"]) if s.get("swa_window_tokens") else L
            kv = s["kv_bytes_per_token"] * L + s.get("swa_bytes_per_token", 0) * swa_tokens
            total = fixed + kv
            r = {
                "model": name, "context": L,
                "state_MiB": round(fixed / 2**20, 2), "kv_MiB": round(kv / 2**20, 2),
                "state_share": round(fixed / total, 3) if total else 0,
                "reload_state_ms": round(reload_contig(fixed), 3),
                "reload_kv_ms": round(reload_paged(kv), 3),
                "recompute_ms": round(recompute(L), 2),
                "sessions_keep_all": int(budget // total) if total else 0,
                "sessions_keep_state_only": int(budget // fixed) if fixed else "",
            }
            r["reload_all_ms"] = round(r["reload_state_ms"] + r["reload_kv_ms"], 3)
            r["recompute_over_reload"] = round(r["recompute_ms"] / r["reload_all_ms"], 1) if r["reload_all_ms"] else ""
            rows.append(r)

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out}\n")

    hdr = f"{'model':<14}{'ctx':>8}{'state MiB':>11}{'KV MiB':>9}{'state%':>8}{'rl state':>10}{'rl KV':>9}{'recomp':>10}{'keep all':>10}{'keep st':>9}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['model']:<14}{r['context']:>8}{r['state_MiB']:>11.1f}{r['kv_MiB']:>9.1f}"
              f"{100*r['state_share']:>7.0f}%{r['reload_state_ms']:>9.2f}m{r['reload_kv_ms']:>8.2f}m"
              f"{r['recompute_ms']:>9.0f}m{r['sessions_keep_all']:>10}{str(r['sessions_keep_state_only']):>9}")
    print("\n(m = ms; 'keep all' / 'keep st' = paused sessions fitting in the budget)")
    print("Go signal: state share crosses 50% inside realistic agent contexts, and keeping only the")
    print("state fits many more sessions than keeping everything while reload_kv << recompute.")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = list(dict.fromkeys(r["model"] for r in rows))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for n in names:
            rs = [r for r in rows if r["model"] == n]
            x = [r["context"] for r in rs]
            axes[0].plot(x, [r["state_MiB"] for r in rs], "--", label=f"{n} state")
            axes[0].plot(x, [r["kv_MiB"] for r in rs], "-", label=f"{n} KV")
            axes[1].plot(x, [r["reload_all_ms"] for r in rs], "-", label=f"{n} reload")
            axes[1].plot(x, [r["recompute_ms"] for r in rs], ":", label=f"{n} recompute")
        for ax, t, y in ((axes[0], "Memory held per paused session", "MiB"),
                         (axes[1], "Cost to resume a paused session", "ms")):
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(t)
            ax.set_xlabel("context tokens"); ax.set_ylabel(y); ax.grid(alpha=.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(a.plot, dpi=150)
        print(f"wrote {a.plot}")


if __name__ == "__main__":
    main()
