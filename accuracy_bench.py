"""
Accuracy checks that go with the serving results.

Two questions:
  1. Does a serving setting we recommend (16-bit recurrent state in SGLang) change what the model computes?
  2. Are the hybrids as accurate as their dense counterparts?

Two measurements, run against one served model:

  replay  Real SWE-agent sessions (traces/swe_smith_*.json). For sampled (session, turn) pairs, the prompt is
          the chat so far (system prompt, task, earlier replies and tool outputs) and the target is Claude's
          actual next reply. We record the log-likelihood the model assigns to that reply. Reported as nats
          per character (comparable across tokenizers) and nats per token. Prefill only, so it is fast.
  gsm8k   GSM8K test questions, greedy decoding, exact match on the final number.

Usage (from the environment of the chosen engine):
  python accuracy_bench.py --engine vllm   --model ibm-granite/granite-4.0-h-micro --tag granite_hmicro --variant vllm
  python accuracy_bench.py --engine sglang --model ibm-granite/granite-4.0-h-micro --tag granite_hmicro \
      --variant sgl_bf16 --engine-args '{"mamba_ssm_dtype":"bfloat16"}'
Writes results/accuracy/<tag>_<variant>_{replay.csv,gsm8k.csv,summary.json}.
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
import sys
import time

import aiohttp

import concurrency_bench as cb


# ---------------------------------------------------------------- prompts
def chat_ids(tok, messages, add_generation_prompt):
    """Token ids of a chat. Renders the chat template to text, then tokenizes it, so the result is
    always a flat list of ints whatever apply_chat_template returns in this transformers version.
    Falls back to plain text if the tokenizer has no chat template."""
    text = None
    if getattr(tok, "chat_template", None):
        for kw in ({"enable_thinking": False}, {}):
            try:
                text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt, **kw)
                break
            except TypeError:
                continue
            except Exception:
                break
    if not isinstance(text, str):
        text = "".join(f"<{m['role']}>\n{m['content']}\n" for m in messages) + ("<assistant>\n" if add_generation_prompt else "")
    ids = tok.encode(text, add_special_tokens=False)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    ids = [int(x) for x in ids]
    return ids


def replay_items(trace_path, tok, n, seed, max_prompt):
    """(prompt ids, reply ids, reply text) for n random (session, turn) pairs whose prompt fits max_prompt."""
    with open(trace_path) as f:
        tr = json.load(f)
    rng = random.Random(seed)
    sessions = list(tr["sessions"])
    rng.shuffle(sessions)
    items = []
    for s in sessions:
        if len(items) >= n:
            break
        turns = s["turns"]
        if len(turns) < 2:
            continue
        k = rng.randrange(len(turns))
        msgs = [{"role": "system", "content": tr["system"]}, {"role": "user", "content": s["task"]}]
        for t in turns[:k]:
            msgs += [{"role": "assistant", "content": t["reply"]}, {"role": "user", "content": t["tool"]}]
        reply = turns[k]["reply"]
        if not reply.strip():
            continue
        prompt = chat_ids(tok, msgs, add_generation_prompt=True)
        if len(prompt) > max_prompt:
            continue
        target = list(tok.encode(reply, add_special_tokens=False))[:2048]
        if not target:
            continue
        items.append((prompt, target, tok.decode(target)))
    return items


def load_gsm8k(n, seed):
    from huggingface_hub import hf_hub_download
    import pandas as pd
    p = hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(p)
    rows = [(q, a.split("####")[-1].strip().replace(",", "")) for q, a in zip(df["question"], df["answer"])]
    random.Random(seed).shuffle(rows)
    return rows[:n]


NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_answer(text):
    m = re.search(r"answer is[^0-9\-]*(-?\d[\d,]*\.?\d*)", text, re.I)
    if m:
        s = m.group(1)
    else:
        nums = NUM.findall(text)
        if not nums:
            return None
        s = nums[-1]
    s = s.replace(",", "").rstrip(".")
    try:
        v = float(s)
        return str(int(v)) if v == int(v) else str(v)
    except ValueError:
        return None


# ---------------------------------------------------------------- requests
async def prompt_logprobs(session, url, engine, prompt, target):
    """Log-probabilities of the target tokens given the prompt (one value per target token)."""
    ids = prompt + target
    if engine == "sglang":
        body = {"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
                "return_logprob": True, "logprob_start_len": len(prompt) - 1, "top_logprobs_num": 0}
        async with session.post(url + "/generate", json=body, timeout=aiohttp.ClientTimeout(total=600)) as r:
            data = await r.json()
        if isinstance(data, list):
            data = data[0]
        lp = (data.get("meta_info") or {}).get("input_token_logprobs") or []
        vals = [x[0] for x in lp if x and x[0] is not None]
        return vals[-len(target):]
    body = {"model": "bench", "prompt": ids, "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 0}
    async with session.post(url + "/v1/completions", json=body, timeout=aiohttp.ClientTimeout(total=600)) as r:
        data = await r.json()
    pl = data["choices"][0].get("prompt_logprobs") or data.get("prompt_logprobs") or []
    vals = []
    for pos, entry in zip(range(len(ids)), pl):
        if pos < len(prompt) or not entry:
            continue
        e = entry.get(str(ids[pos])) or next(iter(entry.values()))
        vals.append(e["logprob"] if isinstance(e, dict) else float(e))
    return vals


async def generate(session, url, engine, prompt, max_tokens):
    if engine == "sglang":
        body = {"input_ids": prompt, "sampling_params": {"max_new_tokens": max_tokens, "temperature": 0.0}}
        async with session.post(url + "/generate", json=body, timeout=aiohttp.ClientTimeout(total=600)) as r:
            data = await r.json()
        if isinstance(data, list):
            data = data[0]
        return data.get("text", "")
    body = {"model": "bench", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0}
    async with session.post(url + "/v1/completions", json=body, timeout=aiohttp.ClientTimeout(total=600)) as r:
        data = await r.json()
    return data["choices"][0]["text"]


async def run_all(fn, items, concurrency):
    sem = asyncio.Semaphore(concurrency)
    out = [None] * len(items)

    async def one(i, it):
        async with sem:
            try:
                out[i] = await fn(it)
            except Exception as e:
                out[i] = e
    await asyncio.gather(*(one(i, it) for i, it in enumerate(items)))
    return out


# ---------------------------------------------------------------- main
async def bench(a, tok):
    url = f"http://127.0.0.1:{a.port}"
    os.makedirs(a.out_dir, exist_ok=True)
    stem = os.path.join(a.out_dir, f"{a.tag}_{a.variant}")
    summary = {"model": a.model, "tag": a.tag, "variant": a.variant, "engine": a.engine, "engine_args": a.engine_args}
    async with aiohttp.ClientSession() as s:
        if "replay" in a.tasks:
            items = replay_items(a.trace, tok, a.replay_n, a.seed, a.max_prompt)
            med = statistics.median(len(p) for p, _, _ in items)
            if med < 100:
                sys.exit(f"[error] replay prompts look broken (median {med} tokens); not writing results")
            print(f"[replay] {len(items)} (session, turn) pairs; prompt median "
                  f"{int(statistics.median(len(p) for p, _, _ in items))} tokens", flush=True)
            t0 = time.time()
            res = await run_all(lambda it: prompt_logprobs(s, url, a.engine, it[0], it[1]), items, a.concurrency)
            rows, nll, ntok, nchar, errs = [], 0.0, 0, 0, 0
            for i, ((p, t, txt), lp) in enumerate(zip(items, res)):
                if isinstance(lp, Exception) or not lp or len(lp) != len(t):
                    errs += 1
                    continue
                total = -sum(lp)
                nll += total
                ntok += len(t)
                nchar += len(txt)
                rows.append({"item": i, "prompt_tokens": len(p), "reply_tokens": len(t), "reply_chars": len(txt),
                             "nll": round(total, 4)})
            with open(stem + "_replay.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["item", "prompt_tokens", "reply_tokens", "reply_chars", "nll"])
                w.writeheader()
                w.writerows(rows)
            summary["replay"] = {"pairs": len(rows), "errors": errs, "reply_tokens": ntok, "reply_chars": nchar,
                                 "nats_per_token": round(nll / ntok, 4) if ntok else None,
                                 "nats_per_char": round(nll / nchar, 5) if nchar else None,
                                 "bits_per_char": round(nll / nchar / math.log(2), 5) if nchar else None,
                                 "wall_s": round(time.time() - t0, 1)}
            print("[replay]", summary["replay"], flush=True)
            if errs and not rows:
                print("[replay] first error:", next((r for r in res if isinstance(r, Exception)), res[0]), flush=True)

        if "gsm8k" in a.tasks:
            qs = load_gsm8k(a.gsm8k_n, a.seed)
            prompts = [chat_ids(tok, [{"role": "user", "content": q + "\n\nSolve step by step, then finish with "
                                       "'The answer is <number>.'"}], add_generation_prompt=True) for q, _ in qs]
            if statistics.median(len(p) for p in prompts) < 30:
                sys.exit("[error] GSM8K prompts look broken; not writing results")
            t0 = time.time()
            outs = await run_all(lambda p: generate(s, url, a.engine, p, a.gsm8k_max_tokens), prompts, a.concurrency)
            rows, correct, errs = [], 0, 0
            for i, ((q, gold), o) in enumerate(zip(qs, outs)):
                if isinstance(o, Exception):
                    errs += 1
                    o = ""
                pred = extract_answer(o)
                ok = pred is not None and pred == gold
                correct += ok
                rows.append({"item": i, "gold": gold, "pred": pred, "correct": int(ok), "out_chars": len(o)})
            with open(stem + "_gsm8k.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["item", "gold", "pred", "correct", "out_chars"])
                w.writeheader()
                w.writerows(rows)
            n = len(qs)
            se = math.sqrt(correct / n * (1 - correct / n) / n) if n else 0
            summary["gsm8k"] = {"n": n, "correct": correct, "errors": errs, "accuracy": round(correct / n, 4) if n else None,
                                "stderr": round(se, 4), "wall_s": round(time.time() - t0, 1)}
            print("[gsm8k]", summary["gsm8k"], flush=True)

        if "greedy" in a.tasks:
            # Greedy continuations of real agent prompts, saved for output-equality checks across settings
            # (e.g., 32- vs 16-bit recurrent state, and a 32-bit rerun as the nondeterminism baseline).
            items = replay_items(a.trace, tok, a.greedy_n, a.seed + 1, a.max_prompt)
            t0 = time.time()
            outs = await run_all(lambda it: generate(s, url, a.engine, it[0], a.greedy_tokens), items, a.concurrency)
            errs = 0
            with open(stem + "_greedy.jsonl", "w") as f:
                for i, o in enumerate(outs):
                    if isinstance(o, Exception):
                        errs += 1
                        o = None
                    f.write(json.dumps({"item": i, "prompt_tokens": len(items[i][0]), "text": o}) + "\n")
            summary["greedy"] = {"n": len(items), "errors": errs, "max_tokens": a.greedy_tokens,
                                 "wall_s": round(time.time() - t0, 1)}
            print("[greedy]", summary["greedy"], flush=True)
    with open(stem + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["vllm", "sglang"], default="vllm")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--variant", default="default")
    ap.add_argument("--engine-args", default="{}")
    ap.add_argument("--tasks", default="replay,gsm8k,greedy")
    ap.add_argument("--greedy-n", type=int, default=200)
    ap.add_argument("--greedy-tokens", type=int, default=256)
    ap.add_argument("--trace", default="traces/swe_smith_1500.json")
    ap.add_argument("--replay-n", type=int, default=300)
    ap.add_argument("--max-prompt", type=int, default=12000)
    ap.add_argument("--gsm8k-n", type=int, default=500)
    ap.add_argument("--gsm8k-max-tokens", type=int, default=768)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-util", type=float, default=None)
    ap.add_argument("--startup-timeout", type=int, default=2400)
    ap.add_argument("--no-server", action="store_true", help="use a server already running on --port")
    ap.add_argument("--out-dir", default="results/accuracy")
    a = ap.parse_args()
    a.tasks = set(a.tasks.split(","))
    cb.ENGINE = a.engine
    gpu_util = a.gpu_util or (0.8 if a.engine == "sglang" else 0.85)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)

    proc = None
    os.makedirs(a.out_dir, exist_ok=True)
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
        asyncio.run(bench(a, tok))
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except Exception:
                proc.kill()


if __name__ == "__main__":
    main()
