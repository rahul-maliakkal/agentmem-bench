#!/usr/bin/env python3
"""
GPU <-> pinned CPU transfer cost for the two kinds of session state:

  contiguous : one block of S bytes (how a recurrent state is laid out)
  paged-loop : S bytes split into pages of P bytes, copied one by one
               (naive offload of scattered KV pages)
  paged-gather: same pages, gathered into a staging buffer on GPU, then
               one copy (what a good offload connector does)

Times both directions (offload = D2H, reload = H2D) with CUDA events.

Usage:
  python transfer_bench.py --out results/transfer.csv
"""
import argparse
import csv
import os
import statistics

import torch


def time_ms(fn, iters):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    fn(); torch.cuda.synchronize()  # warmup
    times = []
    for _ in range(iters):
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


def bench(total_bytes, page_bytes, iters, dev):
    out = []
    # contiguous
    g = torch.empty(total_bytes, dtype=torch.uint8, device=dev)
    c = torch.empty(total_bytes, dtype=torch.uint8, pin_memory=True)
    out.append(("contiguous", "offload", time_ms(lambda: c.copy_(g, non_blocking=True), iters)))
    out.append(("contiguous", "reload", time_ms(lambda: g.copy_(c, non_blocking=True), iters)))
    del g, c

    n_pages = max(1, total_bytes // page_bytes)
    # Scattered pages inside a larger pool, as in a paged KV allocator.
    pool = torch.empty((n_pages * 4, page_bytes), dtype=torch.uint8, device=dev)
    idx = torch.randperm(n_pages * 4, device=dev)[:n_pages]
    idx_list = idx.tolist()
    host = torch.empty((n_pages, page_bytes), dtype=torch.uint8, pin_memory=True)
    staging = torch.empty((n_pages, page_bytes), dtype=torch.uint8, device=dev)

    def loop_off():
        for j, i in enumerate(idx_list):
            host[j].copy_(pool[i], non_blocking=True)

    def loop_on():
        for j, i in enumerate(idx_list):
            pool[i].copy_(host[j], non_blocking=True)

    def gather_off():
        torch.index_select(pool, 0, idx, out=staging)
        host.copy_(staging, non_blocking=True)

    def gather_on():
        staging.copy_(host, non_blocking=True)
        pool.index_copy_(0, idx, staging)

    loop_iters = max(3, iters // 4)
    out.append(("paged-loop", "offload", time_ms(loop_off, loop_iters)))
    out.append(("paged-loop", "reload", time_ms(loop_on, loop_iters)))
    out.append(("paged-gather", "offload", time_ms(gather_off, iters)))
    out.append(("paged-gather", "reload", time_ms(gather_on, iters)))
    del pool, host, staging
    torch.cuda.empty_cache()
    return out, n_pages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes-mib", default="1,4,16,64,256,1024")
    ap.add_argument("--page-kib", type=int, default=64,
                    help="KV page size; ~ block_size * per-layer KV bytes/token")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert torch.cuda.is_available(), "needs a GPU"
    dev = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    rows = []
    for mib in [float(x) for x in a.sizes_mib.split(",")]:
        total = int(mib * 2**20)
        res, n_pages = bench(total, a.page_kib * 1024, a.iters, dev)
        for layout, direction, ms in res:
            gbps = total / (ms / 1e3) / 1e9
            rows.append({"size_MiB": mib, "layout": layout, "direction": direction,
                         "pages": n_pages if layout != "contiguous" else 1,
                         "page_KiB": a.page_kib, "ms": round(ms, 4), "GB_per_s": round(gbps, 2)})
            print(f"{mib:>8.0f} MiB  {layout:<13} {direction:<8} {ms:9.3f} ms  {gbps:7.2f} GB/s")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
