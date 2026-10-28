#!/usr/bin/env python3
"""
Second round of accuracy checks: is "equal accuracy" robust, especially at agent-length contexts?

Four tasks, run against one served model (same server launcher as accuracy_bench.py):

  needle      Long-context retrieval built from real agent histories. The haystack is real SWE-agent tool
              output (file listings, tracebacks, diffs, test logs). Four "export KEY=value" lines with
              different keys are hidden in it (RULER-style multi-key needle); the model is asked for the value
              of one of them. Context 4K, 8K, 16K, 32K tokens; needle at 10%, 50%, 90% depth. Greedy, exact match.
  replaylong  Same as accuracy_bench.py's replay (log-likelihood of Claude's actual next reply), but only for
              positions whose history is 12K-31K tokens long.
  humaneval   HumanEval (164 problems), greedy, pass@1 by running the official tests in a subprocess.
  cachecheck  Does reusing cached state change what the model outputs? For each real agent prompt: clear the
              prefix cache, generate (cold: computed from scratch), then send the same prompt again (warm:
              resumed from whatever the engine cached). Concurrency 1, greedy. Records whether the outputs are
              identical and the difference in the log-probabilities of the generated tokens.

Usage:
  python accuracy2_bench.py --engine vllm --model ibm-granite/granite-4.0-h-micro --tag granite_hmicro --variant vllm
  python accuracy2_bench.py --engine sglang --model ... --variant sgl_bf16 --engine-args '{"mamba_ssm_dtype":"bfloat16"}'
Writes results/accuracy2/<tag>_<variant>_{needle.csv,replaylong.csv,humaneval.csv,cachecheck.csv,summary.json}.
"""
import argparse
import asyncio
import csv
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time

import aiohttp

import concurrency_bench as cb
from accuracy_bench import chat_ids, prompt_logprobs, generate, run_all

KEY_WORDS = ["ALPINE", "BRONZE", "CEDAR", "DELTA", "EMBER", "FALCON", "GARNET", "HARBOR", "IVORY", "JUNIPER",
             "KESTREL", "LANTERN", "MAPLE", "NEBULA", "ORCHID", "PEPPER", "QUARTZ", "RAVEN", "SAFFRON", "TUNDRA"]


# ---------------------------------------------------------------- needle
def needle_items(trace, tok, lengths, depths, per_cell, seed, max_model_len):
    rng = random.Random(seed)
    tools = [t["tool"] for s in trace["sessions"] for t in s["turns"] if t.get("tool", "").strip()]
    items = []
    for L in lengths:
        for d in depths:
            for j in range(per_cell):
                rng.shuffle(tools)
                # haystack: real tool outputs until we have more than L tokens, then cut to L tokens
                buf, n = [], 0
                for t in tools:
                    buf.append(t)
                    n += len(t) // 3  # rough; exact cut below
                    if n > L * 1.6:
                        break
                ids = tok.encode("\n\n".join(buf), add_special_tokens=False)
                if hasattr(ids, "keys"):
                    ids = ids["input_ids"]
                hay = tok.decode(list(ids)[:L])
                keys = rng.sample(KEY_WORDS, 4)
                vals = [str(rng.randrange(1_000_000, 9_999_999)) for _ in keys]
                target = 0
                # the asked-for needle at depth d, the three distractors at random depths
                spots = [d] + [rng.uniform(0.05, 0.95) for _ in range(3)]
                ins = sorted(zip(spots, range(4)), reverse=True)  # insert from the back so offsets stay valid
                for frac, k in ins:
                    pos = int(len(hay) * frac)
                    nl = hay.find("\n", pos)
                    pos = nl + 1 if nl != -1 else pos
                    hay = hay[:pos] + f"export CI_CACHE_KEY_{keys[k]}={vals[k]}\n" + hay[pos:]
                q = (f"Here is the output of commands an agent ran in a repository.\n\n{hay}\n\n"
                     f"What is the value of CI_CACHE_KEY_{keys[target]}? Reply with the number only.")
                prompt = chat_ids(tok, [{"role": "user", "content": q}], add_generation_prompt=True)
                if len(prompt) > max_model_len - 32:
                    # trim haystack proportionally and retry once
                    over = len(prompt) - (max_model_len - 32)
                    hay2 = hay[: max(0, len(hay) - int(over * 4.5))]
                    q = (f"Here is the output of commands an agent ran in a repository.\n\n{hay2}\n\n"
                         f"What is the value of CI_CACHE_KEY_{keys[target]}? Reply with the number only.")
                    prompt = chat_ids(tok, [{"role": "user", "content": q}], add_generation_prompt=True)
                    if len(prompt) > max_model_len - 32 or vals[target] not in q:
                        continue
                items.append({"length": L, "depth": d, "prompt": prompt, "answer": vals[target],
                              "distractors": [v for i, v in enumerate(vals) if i != target]})
    return items


# ---------------------------------------------------------------- replay at long contexts
def replay_long_items(trace, tok, n, seed, min_prompt, max_prompt):
    rng = random.Random(seed)
    sessions = list(trace["sessions"])
    rng.shuffle(sessions)
    items = []
    for s in sessions:
        if len(items) >= n:
            break
        turns = s["turns"]
        if len(turns) < 8:
            continue
        for _ in range(3):  # try a few late turns to land in the length window
            k = rng.randrange(len(turns) // 2, len(turns))
            msgs = [{"role": "system", "content": trace["system"]}, {"role": "user", "content": s["task"]}]
            for t in turns[:k]:
                msgs += [{"role": "assistant", "content": t["reply"]}, {"role": "user", "content": t["tool"]}]
            reply = turns[k]["reply"]
            if not reply.strip():
                continue
            prompt = chat_ids(tok, msgs, add_generation_prompt=True)
            if not (min_prompt <= len(prompt) <= max_prompt):
                continue
            target = list(tok.encode(reply, add_special_tokens=False))[:1024]
            if target and len(prompt) + len(target) < max_prompt + 1024:
                items.append((prompt, target, tok.decode(target)))
                break
    return items


# ---------------------------------------------------------------- humaneval
def load_humaneval():
    from huggingface_hub import hf_hub_download
    import pandas as pd
    p = hf_hub_download("openai/openai_humaneval", "openai_humaneval/test-00000-of-00001.parquet", repo_type="dataset")
    return pd.read_parquet(p).to_dict("records")


CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def build_program(prob, out):
    m = CODE_BLOCK.search(out)
    code = m.group(1) if m else out
    if f"def {prob['entry_point']}" not in code:
        code = prob["prompt"] + code  # completion-style answer
    imports = "\n".join(l for l in prob["prompt"].splitlines() if l.startswith(("import ", "from ")))
    return f"{imports}\n{code}\n\n{prob['test']}\n\ncheck({prob['entry_point']})\n"


def run_program(src, timeout=15):
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "prog.py")
        open(f, "w").write(src)
        try:
            r = subprocess.run([sys.executable, "-I", f], cwd=d, capture_output=True, timeout=timeout)
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False


# ---------------------------------------------------------------- cache check
async def reset_cache(session, url, engine):
    path = "/flush_cache" if engine == "sglang" else "/reset_prefix_cache"
    try:
        async with session.post(url + path, timeout=aiohttp.ClientTimeout(total=60)) as r:
            return r.status == 200
    except Exception:
        return False


async def gen_with_lp(session, url, engine, prompt, max_tokens):
    """Greedy generation; returns (text, list of generated-token logprobs, cached prompt tokens or None)."""
    if engine == "sglang":
        body = {"input_ids": prompt, "sampling_params": {"max_new_tokens": max_tokens, "temperature": 0.0},
                "return_logprob": True, "logprob_start_len": len(prompt)}
        async with session.post(url + "/generate", json=body, timeout=aiohttp.ClientTimeout(total=600)) as r:
            data = await r.json()
        if isinstance(data, list):
            data = data[0]
        meta = data.get("meta_info") or {}
        lps = [x[0] for x in (meta.get("output_token_logprobs") or []) if x]
        return data.get("text", ""), lps, meta.get("cached_tokens")
    body = {"model": "bench", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0, "logprobs": 0}
    async with session.post(url + "/v1/completions", json=body, timeout=aiohttp.ClientTimeout(total=600)) as r:
        data = await r.json()
    ch = data["choices"][0]
    lps = (ch.get("logprobs") or {}).get("token_logprobs") or []
    cached = ((data.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    return ch["text"], [x for x in lps if x is not None], cached


# ---------------------------------------------------------------- main
async def bench(a, tok, trace):
    url = f"http://127.0.0.1:{a.port}"
    stem = os.path.join(a.out_dir, f"{a.tag}_{a.variant}")
    summary = {"model": a.model, "tag": a.tag, "variant": a.variant, "engine": a.engine, "engine_args": a.engine_args}
    async with aiohttp.ClientSession() as s:
        if "needle" in a.tasks:
            lengths = [int(x) for x in a.needle_lengths.split(",")]
            depths = [float(x) for x in a.needle_depths.split(",")]
            items = needle_items(trace, tok, lengths, depths, a.needle_n, a.seed, a.max_model_len)
            t0 = time.time()
            outs = await run_all(lambda it: generate(s, url, a.engine, it["prompt"], 24), items, a.concurrency)
            rows, errs = [], 0
            for it, o in zip(items, outs):
                if isinstance(o, Exception):
                    errs += 1
                    o = ""
                digits = re.findall(r"\d{7}", o or "")
                rows.append({"length": it["length"], "depth": it["depth"], "prompt_tokens": len(it["prompt"]),
                             "answer": it["answer"], "output": (o or "").strip()[:60],
                             "correct": int(it["answer"] in digits),
                             "distractor": int(any(dv in digits for dv in it["distractors"]))})
            with open(stem + "_needle.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            by = {}
            for r in rows:
                by.setdefault(r["length"], []).append(r["correct"])
            summary["needle"] = {"n": len(rows), "errors": errs, "accuracy": round(statistics.mean(r["correct"] for r in rows), 4),
                                 "by_length": {str(k): round(statistics.mean(v), 4) for k, v in sorted(by.items())},
                                 "wall_s": round(time.time() - t0, 1)}
            print("[needle]", summary["needle"], flush=True)

        if "replaylong" in a.tasks:
            items = replay_long_items(trace, tok, a.replay_n, a.seed, a.min_prompt, a.max_model_len - 1100)
            if not items:
                print("[replaylong] no positions in the length window", flush=True)
            else:
                print(f"[replaylong] {len(items)} positions, prompt median {int(statistics.median(len(p) for p, _, _ in items))} tokens", flush=True)
                t0 = time.time()
                res = await run_all(lambda it: prompt_logprobs(s, url, a.engine, it[0], it[1]), items, min(a.concurrency, 4))  # long prompt logprobs are memory-heavy
                rows, nll, nchar, ntok, errs = [], 0.0, 0, 0, 0
                for i, ((p, t, txt), lp) in enumerate(zip(items, res)):
                    if isinstance(lp, Exception) or not lp or len(lp) != len(t):
                        errs += 1
                        continue
                    tot = -sum(lp)
                    nll += tot; nchar += len(txt); ntok += len(t)
                    rows.append({"item": i, "prompt_tokens": len(p), "reply_tokens": len(t), "reply_chars": len(txt), "nll": round(tot, 4)})
                with open(stem + "_replaylong.csv", "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["item", "prompt_tokens", "reply_tokens", "reply_chars", "nll"])
                    w.writeheader()
                    w.writerows(rows)
                summary["replaylong"] = {"pairs": len(rows), "errors": errs,
                                         "bits_per_char": round(nll / nchar / math.log(2), 5) if nchar else None,
                                         "nats_per_token": round(nll / ntok, 4) if ntok else None,
                                         "wall_s": round(time.time() - t0, 1)}
                print("[replaylong]", summary["replaylong"], flush=True)

        if "humaneval" in a.tasks:
            probs = load_humaneval()
            msgs = lambda p: [{"role": "user", "content": "Complete the following Python function. Reply with the "
                                                          "complete function in a ```python code block.\n\n```python\n" + p["prompt"] + "```"}]
            prompts = [chat_ids(tok, msgs(p), add_generation_prompt=True) for p in probs]
            t0 = time.time()
            outs = await run_all(lambda pr: generate(s, url, a.engine, pr, 768), prompts, a.concurrency)
            rows, errs = [], 0
            for p, o in zip(probs, outs):
                if isinstance(o, Exception):
                    errs += 1
                    o = ""
                ok = run_program(build_program(p, o or ""))
                rows.append({"task_id": p["task_id"], "passed": int(ok), "out_chars": len(o or "")})
            with open(stem + "_humaneval.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["task_id", "passed", "out_chars"])
                w.writeheader()
                w.writerows(rows)
            n = len(rows); k = sum(r["passed"] for r in rows)
            summary["humaneval"] = {"n": n, "passed": k, "errors": errs, "pass_at_1": round(k / n, 4),
                                    "stderr": round(math.sqrt(k / n * (1 - k / n) / n), 4), "wall_s": round(time.time() - t0, 1)}
            print("[humaneval]", summary["humaneval"], flush=True)

        if "cachecheck" in a.tasks:
            from accuracy_bench import replay_items
            items = replay_items(a.trace, tok, a.cache_n, a.seed + 7, 12000)
            t0 = time.time()
            rows, reset_ok = [], True
            for i, (p, _, _) in enumerate(items):
                reset_ok &= await reset_cache(s, url, a.engine)
                try:
                    ta, la, ca = await gen_with_lp(s, url, a.engine, p, a.cache_tokens)
                    tb, lb, cbk = await gen_with_lp(s, url, a.engine, p, a.cache_tokens)
                except Exception as e:
                    print("[cachecheck] error:", e, flush=True)
                    continue
                n = 0
                for x, y in zip(ta, tb):
                    if x != y:
                        break
                    n += 1
                m = min(len(la), len(lb))
                diffs = [abs(la[j] - lb[j]) for j in range(m)]
                # compare logprobs only up to the first differing token (after that the inputs differ)
                rows.append({"item": i, "prompt_tokens": len(p), "cached_cold": ca, "cached_warm": cbk,
                             "identical": int(ta == tb), "shared_prefix_frac": round(n / max(1, len(ta), len(tb)), 4),
                             "first_lp_diff": round(diffs[0], 6) if diffs else "",
                             "max_lp_diff_prefix": round(max(diffs[:max(1, n // 3)]), 6) if diffs else ""})
            with open(stem + "_cachecheck.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            fl = [r["first_lp_diff"] for r in rows if r["first_lp_diff"] != ""]
            cw = [r["cached_warm"] for r in rows if isinstance(r["cached_warm"], int)]
            cc = [r["cached_cold"] for r in rows if isinstance(r["cached_cold"], int)]
            summary["cachecheck"] = {"n": len(rows), "reset_ok": bool(reset_ok),
                                     "identical": round(statistics.mean(r["identical"] for r in rows), 4),
                                     "mean_shared_prefix": round(statistics.mean(r["shared_prefix_frac"] for r in rows), 4),
                                     "median_first_lp_diff": round(statistics.median(fl), 6) if fl else None,
                                     "max_first_lp_diff": round(max(fl), 6) if fl else None,
                                     "median_cached_cold": statistics.median(cc) if cc else None,
                                     "median_cached_warm": statistics.median(cw) if cw else None,
                                     "median_prompt": statistics.median(r["prompt_tokens"] for r in rows),
                                     "wall_s": round(time.time() - t0, 1)}
            print("[cachecheck]", summary["cachecheck"], flush=True)
    with open(stem + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["vllm", "sglang"], default="vllm")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--variant", default="default")
    ap.add_argument("--engine-args", default="{}")
    ap.add_argument("--tasks", default="needle,replaylong,humaneval,cachecheck")
    ap.add_argument("--trace", default="traces/swe_smith_1500.json")
    ap.add_argument("--needle-lengths", default="4000,8000,16000,31000")
    ap.add_argument("--needle-depths", default="0.1,0.5,0.9")
    ap.add_argument("--needle-n", type=int, default=15, help="prompts per (length, depth)")
    ap.add_argument("--replay-n", type=int, default=200)
    ap.add_argument("--min-prompt", type=int, default=12000)
    ap.add_argument("--cache-n", type=int, default=60)
    ap.add_argument("--cache-tokens", type=int, default=96)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-util", type=float, default=None)
    ap.add_argument("--startup-timeout", type=int, default=2400)
    ap.add_argument("--no-server", action="store_true")
    ap.add_argument("--out-dir", default="results/accuracy2")
    ap.add_argument("--selftest", action="store_true", help="build prompts with a local tokenizer, no server")
    a = ap.parse_args()
    a.tasks = set(a.tasks.split(","))
    cb.ENGINE = a.engine
    gpu_util = a.gpu_util or (0.8 if a.engine == "sglang" else 0.85)
    os.makedirs(a.out_dir, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    with open(a.trace) as f:
        trace = json.load(f)

    if a.selftest:
        it = needle_items(trace, tok, [4000, 31000], [0.5], 2, a.seed, a.max_model_len)
        print("needle prompts:", [(x["length"], len(x["prompt"])) for x in it])
        rl = replay_long_items(trace, tok, 5, a.seed, a.min_prompt, a.max_model_len - 1100)
        print("replaylong prompts:", [len(p) for p, _, _ in rl])
        return

    proc = None
    log_path = os.path.join(a.out_dir, f"{a.tag}_{a.variant}_server.log")
    try:
        if not a.no_server:
            proc = cb.start_server(a.model, a.port, a.max_model_len, gpu_util, json.loads(a.engine_args), log_path)
            try:
                asyncio.run(cb.wait_healthy(f"http://127.0.0.1:{a.port}", proc, timeout=a.startup_timeout))
            except Exception as e:
                print(f"[error] {e}", flush=True)
                try:
                    print("".join(open(log_path).readlines()[-40:]), flush=True)
                except OSError:
                    pass
                sys.exit(1)
        asyncio.run(bench(a, tok, trace))
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except Exception:
                proc.kill()


if __name__ == "__main__":
    main()
