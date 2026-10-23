#!/usr/bin/env python3
"""
How many agents fit on one GPU? Dense vs hybrid under concurrent agent load.

Starts (or connects to) a vLLM OpenAI-compatible server, then runs N agent
sessions at the same time. Each agent loops over turns:

    send full history -> model writes a tool call (gen_len tokens)
    -> tool runs (sleep, lognormal duration) -> tool output appended -> next turn

While it runs, a sampler scrapes the server's /metrics every second
(KV/state cache usage, running/waiting requests, preemptions, prefix-cache
hits) and GPU memory from nvidia-smi.

Outputs, per run (model x variant x N):
  <prefix>_requests.csv  one row per model call: latency, prompt length, cached tokens
  <prefix>_metrics.csv   one row per second: cache usage, queue, preemptions, GPU memory
  <prefix>_summary.json  throughput, latency percentiles, job completion time,
                         preemptions, prefix hit rate, peak cache usage

Usage (server started for you):
  python concurrency_bench.py --model ibm-granite/granite-4.0-h-micro --variant default \
      --agents 8,32,64,128 --out-dir results/conc
Usage (server already running):
  python concurrency_bench.py --url http://localhost:8000 --model <name> --agents 32 --out-dir results/conc
"""
import argparse
import asyncio
import csv
import json
import math
import os
import random
import re
import shlex
import statistics
import subprocess
import sys
import time

import aiohttp


# ---------------------------------------------------------------- server
def engine_args_to_cli(engine_args):
    out = []
    for k, v in engine_args.items():
        flag = "--" + k.replace("_", "-")
        if v is True:
            out.append(flag)
        elif v is False or v is None:
            continue
        else:
            out += [flag, str(v)]
    return out


def start_server(model, port, max_len, gpu_util, engine_args, log_path):
    cmd = ["vllm", "serve", model, "--port", str(port), "--max-model-len", str(max_len),
           "--gpu-memory-utilization", str(gpu_util), "--enable-prefix-caching",
           "--enable-prompt-tokens-details", "--served-model-name", "bench",
           "--trust-remote-code"] + engine_args_to_cli(engine_args)
    env = dict(os.environ, VLLM_SERVER_DEV_MODE="1")  # enables /reset_prefix_cache
    print("[server] " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    log = open(log_path, "w")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)


async def wait_healthy(url, proc, timeout=900):
    t0 = time.time()
    async with aiohttp.ClientSession() as s:
        while time.time() - t0 < timeout:
            if proc is not None and proc.poll() is not None:
                raise RuntimeError("server exited during startup; see its log")
            try:
                async with s.get(url + "/health", timeout=aiohttp.ClientTimeout(total=5)) as r:
                    if r.status == 200:
                        return
            except Exception:
                pass
            await asyncio.sleep(3)
    raise RuntimeError("server did not become healthy in time")


# ---------------------------------------------------------------- metrics
METRIC_KEYS = {
    "cache_usage": re.compile(r"^vllm:(kv_cache_usage_perc|gpu_cache_usage_perc)\b"),
    "running": re.compile(r"^vllm:num_requests_running\b"),
    "waiting": re.compile(r"^vllm:num_requests_waiting\b"),
    "preemptions": re.compile(r"^vllm:num_preemptions(_total)?\b"),
    "prefix_queries": re.compile(r"^vllm:(gpu_)?prefix_cache_queries(_total)?\b"),
    "prefix_hits": re.compile(r"^vllm:(gpu_)?prefix_cache_hits(_total)?\b"),
    "prompt_tokens": re.compile(r"^vllm:prompt_tokens(_total)?\b"),
    "gen_tokens": re.compile(r"^vllm:generation_tokens(_total)?\b"),
    "ttft_sum": re.compile(r"^vllm:time_to_first_token_seconds_sum\b"),
    "ttft_count": re.compile(r"^vllm:time_to_first_token_seconds_count\b"),
}


def parse_metrics(text):
    vals = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        for key, rx in METRIC_KEYS.items():
            if rx.match(line):
                try:
                    vals[key] = vals.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
    return vals


async def gpu_mem_mib():
    # Run nvidia-smi without blocking the event loop: if the GPU hangs, nvidia-smi can hang
    # too, and a blocking call would freeze the whole benchmark (and its watchdog).
    try:
        p = await asyncio.create_subprocess_exec(
            "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        try:
            out, _ = await asyncio.wait_for(p.communicate(), timeout=5)
        except asyncio.TimeoutError:
            try:
                p.kill()
            except Exception:
                pass
            return None
        lines = out.decode().strip().splitlines()
        return float(lines[0]) if lines else None
    except Exception:
        return None


async def sampler(url, t0, rows, stop):
    async with aiohttp.ClientSession() as s:
        while not stop.is_set():
            row = {"t": round(time.time() - t0, 2)}
            try:
                async with s.get(url + "/metrics", timeout=aiohttp.ClientTimeout(total=5)) as r:
                    row.update(parse_metrics(await r.text()))
            except Exception:
                pass
            row["gpu_mem_mib"] = await gpu_mem_mib()
            rows.append(row)
            try:
                await asyncio.wait_for(stop.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass


async def scrape_once(url):
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(url + "/metrics", timeout=aiohttp.ClientTimeout(total=15)) as r:
                return parse_metrics(await r.text())
    except Exception as e:
        print(f"[warn] could not read /metrics ({type(e).__name__}); server may be unresponsive", file=sys.stderr)
        return {}


# ---------------------------------------------------------------- agents
class Rng:
    def __init__(self, seed, vocab):
        self.r = random.Random(seed)
        self.lo, self.hi = 1000, max(1001, min(vocab, 100000) - 1000)

    def toks(self, n):
        return [self.r.randrange(self.lo, self.hi) for _ in range(n)]

    def lognorm(self, median, sigma, lo, hi):
        return min(hi, max(lo, self.r.lognormvariate(math.log(median), sigma)))


async def agent(aid, a, url, session, system, vocab, max_ctx, req_rows, job_rows, t0, warn):
    rng = Rng(a.seed * 100003 + aid, vocab)
    await asyncio.sleep(rng.r.uniform(0, a.stagger_s))  # agents arrive over a few seconds
    hist = system + rng.toks(a.task_len)
    start = time.time()
    turns_done = 0
    for turn in range(a.turns):
        if len(hist) + a.gen_len + 64 > max_ctx:
            break
        body = {"model": "bench", "prompt": hist, "max_tokens": a.gen_len, "temperature": 0.0,
                "ignore_eos": True, "return_token_ids": True}
        ts = time.time()
        try:
            async with session.post(url + "/v1/completions", json=body,
                                    timeout=aiohttp.ClientTimeout(total=a.request_timeout)) as r:
                data = await r.json()
                status = r.status
        except Exception as e:
            req_rows.append({"agent": aid, "turn": turn, "prompt_len": len(hist), "status": f"error:{type(e).__name__}",
                             "latency_s": round(time.time() - ts, 3), "start_s": round(ts - t0, 2)})
            break
        lat = time.time() - ts
        if status != 200:
            req_rows.append({"agent": aid, "turn": turn, "prompt_len": len(hist), "status": f"http{status}",
                             "latency_s": round(lat, 3), "start_s": round(ts - t0, 2)})
            break
        ch = data["choices"][0]
        gen = ch.get("token_ids")
        if not gen:
            if not warn.get("ids"):
                print("[warn] server did not return token_ids; appending placeholder tokens "
                      "(prefix reuse of the model's own reply will be understated)", file=sys.stderr)
                warn["ids"] = True
            gen = rng.toks(a.gen_len)
        usage = data.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        req_rows.append({"agent": aid, "turn": turn, "prompt_len": len(hist), "status": "ok",
                         "cached_tokens": cached if cached is not None else "",
                         "latency_s": round(lat, 3), "start_s": round(ts - t0, 2)})
        hist = hist + list(gen)
        turns_done += 1
        if turn < a.turns - 1:
            await asyncio.sleep(rng.lognorm(a.tool_median_s, a.tool_sigma, a.tool_min_s, a.tool_max_s))
            hist = hist + rng.toks(int(rng.lognorm(a.tool_median_tok, 1.0, 30, a.tool_max_tok)))
    job_rows.append({"agent": aid, "turns": turns_done, "jct_s": round(time.time() - start, 2),
                     "final_ctx": len(hist)})


async def run_load(a, url, n_agents, vocab, out_prefix):
    seed_rng = random.Random(a.seed * 7 + n_agents)
    system = [seed_rng.randrange(1000, max(1001, min(vocab, 100000) - 1000)) for _ in range(a.sys_len)]
    try:  # start every run with an empty prefix cache
        async with aiohttp.ClientSession() as s:
            async with s.post(url + "/reset_prefix_cache", timeout=aiohttp.ClientTimeout(total=30)) as r:
                pass
    except Exception:
        pass
    before = await scrape_once(url)
    req_rows, job_rows, met_rows, warn = [], [], [], {}
    stop = asyncio.Event()
    t0 = time.time()
    samp = asyncio.create_task(sampler(url, t0, met_rows, stop))
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = [asyncio.create_task(agent(i, a, url, session, system, vocab, a.max_model_len,
                                           req_rows, job_rows, t0, warn)) for i in range(n_agents)]
        # Watchdog: if the server stops answering (no call finishes for stall_s while agents
        # are still waiting), stop this load level instead of waiting out every request timeout.
        stalled = False
        last_n, last_t = 0, time.time()
        while True:
            done, pending = await asyncio.wait(tasks, timeout=10)
            if not pending or time.time() - t0 > a.run_timeout:
                break
            if len(req_rows) != last_n:
                last_n, last_t = len(req_rows), time.time()
            elif time.time() - last_t > a.stall_s:
                stalled = True
                print(f"[STALL] no model call finished for {a.stall_s:.0f} s with {len(pending)} agents "
                      f"still waiting; stopping this load level", flush=True)
                try:
                    with open(out_prefix + "_stall_diag.txt", "w") as f:
                        for cmd in (["nvidia-smi"], ["ps", "aux"]):
                            f.write("$ " + " ".join(cmd) + "\n")
                            try:
                                f.write(subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout)
                            except Exception as e:
                                f.write(f"failed: {e}\n")
                except Exception:
                    pass
                break
        for p in pending:
            p.cancel()
    wall = time.time() - t0
    stop.set()
    try:
        await asyncio.wait_for(samp, timeout=30)
    except asyncio.TimeoutError:
        samp.cancel()
    after = await scrape_once(url)

    def delta(k):
        return (after.get(k, 0) - before.get(k, 0)) if k in after else None

    ok = [r for r in req_rows if r["status"] == "ok"]
    lats = sorted(r["latency_s"] for r in ok)
    jcts = sorted(j["jct_s"] for j in job_rows)

    def pct(xs, q):
        return xs[min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))] if xs else None

    cached = [int(r["cached_tokens"]) for r in ok if r.get("cached_tokens") not in ("", None)]
    prompt = [r["prompt_len"] for r in ok if r.get("cached_tokens") not in ("", None)]
    ttft_n = delta("ttft_count")
    summary = {
        "model": a.model, "variant": a.variant, "agents": n_agents, "wall_s": round(wall, 1),
        "status": "stalled" if stalled else "ok",
        "agents_finished": len(job_rows), "agents_unfinished": len(pending),
        "calls_ok": len(ok), "calls_failed": len(req_rows) - len(ok),
        "turns_per_s": round(len(ok) / wall, 3) if wall else None,
        "gen_tok_per_s": round(delta("gen_tokens") / wall, 1) if delta("gen_tokens") is not None else None,
        "latency_p50_s": pct(lats, 50), "latency_p90_s": pct(lats, 90), "latency_p99_s": pct(lats, 99),
        "jct_p50_s": pct(jcts, 50), "jct_p90_s": pct(jcts, 90),
        "mean_ttft_s": round(delta("ttft_sum") / ttft_n, 4) if ttft_n else None,
        "preemptions": delta("preemptions"),
        "prefix_hit_rate": round(delta("prefix_hits") / delta("prefix_queries"), 4)
        if delta("prefix_queries") else None,
        "reused_token_frac": round(sum(cached) / sum(prompt), 4) if prompt else None,
        "peak_cache_usage": max((m.get("cache_usage", 0) for m in met_rows), default=None),
        "peak_waiting": max((m.get("waiting", 0) for m in met_rows), default=None),
        "peak_gpu_mem_mib": max((m["gpu_mem_mib"] for m in met_rows if m.get("gpu_mem_mib")), default=None),
    }

    def write(path, rows):
        if not rows:
            return
        keys = sorted({k for r in rows for k in r})
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)

    write(out_prefix + "_requests.csv", req_rows)
    write(out_prefix + "_metrics.csv", met_rows)
    write(out_prefix + "_jobs.csv", job_rows)
    with open(out_prefix + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ---------------------------------------------------------------- main
async def amain(a):
    if a.vocab:
        vocab = a.vocab
    else:
        from transformers import AutoTokenizer
        vocab = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True).vocab_size
    os.makedirs(a.out_dir, exist_ok=True)
    tag = a.tag or a.model.split("/")[-1]
    proc = None
    url = a.url
    if not url:
        url = f"http://localhost:{a.port}"
        proc = start_server(a.model, a.port, a.max_model_len, a.gpu_util, json.loads(a.engine_args),
                            os.path.join(a.out_dir, f"{tag}_{a.variant}_server.log"))
    try:
        await wait_healthy(url, proc)
        print("[server] healthy", flush=True)
        for n in [int(x) for x in a.agents.split(",")]:
            prefix = os.path.join(a.out_dir, f"{tag}_{a.variant}_n{n}")
            if os.path.exists(prefix + "_summary.json") and not a.redo:
                print(f"[skip] {tag} / {a.variant} / {n} agents already done", flush=True)
                continue
            print(f"[run] {tag} / {a.variant} / {n} agents ...", flush=True)
            s = await run_load(a, url, n, vocab, prefix)
            if s.get("status") == "stalled":
                print(f"      STALLED after {s['calls_ok']} calls; server log: {tag}_{a.variant}_server.log. "
                      f"Skipping larger agent counts for this configuration.", flush=True)
                break
            print(f"      turns/s {s['turns_per_s']}, p90 latency {s['latency_p90_s']} s, "
                  f"p90 job time {s['jct_p90_s']} s, preemptions {s['preemptions']}, "
                  f"peak cache {s['peak_cache_usage']}, unfinished {s['agents_unfinished']}", flush=True)
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", help="short name used in output files")
    ap.add_argument("--variant", default="default")
    ap.add_argument("--engine-args", default="{}", help="JSON of extra vLLM engine args")
    ap.add_argument("--url", help="use an already running server instead of starting one")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--agents", default="8,32,64,128,256")
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--sys-len", type=int, default=4000)
    ap.add_argument("--task-len", type=int, default=500)
    ap.add_argument("--gen-len", type=int, default=300)
    ap.add_argument("--tool-median-tok", type=int, default=800)
    ap.add_argument("--tool-max-tok", type=int, default=6000)
    ap.add_argument("--tool-median-s", type=float, default=3.0)
    ap.add_argument("--tool-sigma", type=float, default=1.0)
    ap.add_argument("--tool-min-s", type=float, default=0.2)
    ap.add_argument("--tool-max-s", type=float, default=30.0)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--request-timeout", type=float, default=300)
    ap.add_argument("--run-timeout", type=float, default=1200)
    ap.add_argument("--stagger-s", type=float, default=10.0, help="spread agent start times over this many seconds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stall-s", type=float, default=180,
                    help="stop a load level if no model call finishes for this many seconds")
    ap.add_argument("--redo", action="store_true", help="rerun load levels that already have results")
    ap.add_argument("--vocab", type=int, help="skip loading the tokenizer (testing)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    asyncio.run(amain(a))


if __name__ == "__main__":
    main()
