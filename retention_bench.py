#!/usr/bin/env python3
"""
Experiment 8: how many paused agent sessions can the prefix cache remember?

Each saved resume point costs memory: a dense model keeps the session's KV pages, a
hybrid model must also keep full copies of its recurrent state. This experiment measures
the effective memory cost of those resume points by filling the cache with N paused
sessions and checking how many of them can still resume from where they stopped.

For each N in --sessions-list, on a fresh (flushed) prefix cache:
  1. pause:  N sessions each run one turn: [shared system prompt][unique task] -> reply
  2. resume: in the same order (oldest first), each session sends
             [system][task][reply][tool output] and we read how many prompt tokens
             came from the cache.

Per session we record
  reusable   = tokens the engine could have reused (system + task + reply)
  cached     = tokens it actually reused
  kept_full  = cached >= reusable - 16          (resume point fully kept)
  kept_sys   = cached >= system length           (at least the shared prompt kept)
When N is small every session is kept; as N grows, resume points are evicted. The N at
which kept_full falls tells how many paused agents the GPU can hold resume points for, and
cache memory / that N is the effective memory per resume point.

Works with vLLM (default) or SGLang (--engine sglang, run inside the SGLang venv).

Usage:
  python retention_bench.py --model ibm-granite/granite-4.0-h-micro --variant default \
      --out results/ret_granite_hmicro_default.csv
  python retention_bench.py --engine sglang --model ibm-granite/granite-4.0-h-micro \
      --variant sgl_nobuf --engine-args '{"mamba_radix_cache_strategy":"no_buffer"}' \
      --out results/ret_granite_hmicro_sgl_nobuf.csv
  python retention_bench.py --dry-run --out /tmp/ret.csv      # no GPU, tests the pipeline
"""
import argparse
import csv
import json
import os
import random
import statistics
import sys
import time


class Tokens:
    def __init__(self, vocab, seed):
        self.rng = random.Random(seed)
        self.lo, self.hi = 1000, max(1001, min(vocab, 100000) - 1000)

    def take(self, n):
        return [self.rng.randrange(self.lo, self.hi) for _ in range(n)]


# ---------------------------------------------------------------- engines
class VllmEngine:
    name = "vllm"

    def __init__(self, model, max_len, gpu_util, engine_args):
        from vllm import LLM, SamplingParams
        try:
            from vllm.inputs import TokensPrompt
        except ImportError:
            TokensPrompt = dict
        self.TP, self.SP = TokensPrompt, SamplingParams
        args = dict(model=model, max_model_len=max_len, gpu_memory_utilization=gpu_util,
                    enable_prefix_caching=True, trust_remote_code=True)
        args.update(engine_args)
        self.llm = LLM(**args)
        self.tokenizer = self.llm.get_tokenizer()
        self.vocab = self.tokenizer.vocab_size

    def info(self):
        out = {}
        try:
            cc = self.llm.llm_engine.vllm_config.cache_config
            out = {"num_gpu_blocks": cc.num_gpu_blocks, "block_size": cc.block_size}
        except Exception:
            pass
        return out

    def generate(self, prompts, max_tokens):
        mts = max_tokens if isinstance(max_tokens, list) else [max_tokens] * len(prompts)
        sp = [self.SP(max_tokens=m, temperature=0.0, ignore_eos=True) for m in mts]
        outs = self.llm.generate([self.TP(prompt_token_ids=p) for p in prompts], sp, use_tqdm=False)
        return [(list(o.outputs[0].token_ids), getattr(o, "num_cached_tokens", None)) for o in outs]

    def flush(self):
        try:
            return bool(self.llm.reset_prefix_cache())
        except Exception as e:
            print(f"[warn] reset_prefix_cache failed: {e}", file=sys.stderr)
            return False

    def cache_snapshot(self):
        # Best effort: count cached blocks when the engine core runs in this process
        # (VLLM_ENABLE_V1_MULTIPROCESSING=0). Returns {} if the internals are not reachable.
        seen, stack = set(), [self.llm.llm_engine]
        while stack:
            o = stack.pop()
            if id(o) in seen or len(seen) > 4000:
                continue
            seen.add(id(o))
            if type(o).__name__ == "BlockPool":
                snap = {}
                try:
                    snap["free_blocks"] = o.get_num_free_blocks()
                except Exception:
                    pass
                for attr in ("cached_block_hash_to_block", "_cached_block_hash_to_block"):
                    m = getattr(o, attr, None)
                    if m is not None:
                        try:
                            snap["cached_block_entries"] = len(m)
                        except Exception:
                            pass
                return snap
            for k in ("engine_core", "scheduler", "kv_cache_manager", "coordinator", "block_pool",
                      "model_executor", "engine", "core"):
                v = getattr(o, k, None)
                if v is not None:
                    stack.append(v)
            for k in ("single_type_managers",):
                v = getattr(o, k, None)
                if isinstance(v, (list, tuple)):
                    stack.extend(v)
        return {}


class SglEngine:
    name = "sglang"

    def __init__(self, model, max_len, mem_frac, engine_args):
        import sglang as sgl
        args = dict(model_path=model, context_length=max_len, mem_fraction_static=mem_frac,
                    trust_remote_code=True, log_level="warning")
        args.update(engine_args)
        self.llm = sgl.Engine(**args)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.vocab = self.tokenizer.vocab_size

    def info(self):
        try:
            i = self.llm.get_server_info()
            keys = ("max_total_num_tokens", "mamba_radix_cache_strategy", "max_mamba_cache_size",
                    "mamba_full_memory_ratio", "page_size")
            out = {k: i.get(k) for k in keys if k in i}
            for k, v in i.items():  # some versions nest scheduler info
                if isinstance(v, dict):
                    out.update({kk: v[kk] for kk in keys if kk in v})
            return out
        except Exception:
            return {}

    def generate(self, prompts, max_tokens):
        mts = max_tokens if isinstance(max_tokens, list) else [max_tokens] * len(prompts)
        sp = [{"max_new_tokens": m, "temperature": 0.0, "ignore_eos": True} for m in mts]
        outs = self.llm.generate(input_ids=prompts, sampling_params=sp)
        if isinstance(outs, dict):
            outs = [outs]
        res = []
        for o, m in zip(outs, mts):
            meta = o.get("meta_info", {})
            gen = o.get("output_ids") or meta.get("output_ids") or []
            if len(gen) != m:
                gen = gen or [7] * m
            res.append((list(gen), meta.get("cached_tokens")))
        return res

    def flush(self):
        try:
            r = self.llm.flush_cache()
            return True if r is None else bool(getattr(r, "success", r))
        except Exception as e:
            print(f"[warn] flush_cache failed: {e}", file=sys.stderr)
            return False

    def cache_snapshot(self):
        return {}

    def close(self):
        try:
            self.llm.shutdown()
        except Exception:
            pass


class FakeEngine:
    """LRU cache of whole prefixes with a fixed number of resume slots, for pipeline tests."""
    name = "fake"

    def __init__(self, slots=100):
        from collections import OrderedDict
        self.vocab, self.slots, self.cache, self.tokenizer = 150000, slots, OrderedDict(), None

    def info(self):
        return {"slots": self.slots}

    def generate(self, prompts, max_tokens):
        res = []
        mts = max_tokens if isinstance(max_tokens, list) else [max_tokens] * len(prompts)
        for p, max_tokens in zip(prompts, mts):
            cached = 0
            for k in list(self.cache):
                if len(k) <= len(p) and tuple(p[:len(k)]) == k and len(k) > cached:
                    cached = len(k)
                    self.cache.move_to_end(k)
            full = tuple(p + [7] * max_tokens)
            self.cache[full] = True
            while len(self.cache) > self.slots:
                self.cache.popitem(last=False)
            res.append(([7] * max_tokens, cached))
        return res

    def flush(self):
        self.cache.clear()
        return True

    def cache_snapshot(self):
        return {"entries": len(self.cache)}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["vllm", "sglang"], default="vllm")
    ap.add_argument("--model")
    ap.add_argument("--variant", default="default")
    ap.add_argument("--engine-args", default="{}", help="JSON of extra engine arguments")
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-util", type=float, default=0.85, help="vLLM gpu_memory_utilization")
    ap.add_argument("--mem-frac", type=float, default=0.80, help="SGLang mem_fraction_static")
    ap.add_argument("--sessions-list", default="32,64,128,256,512,1024")
    ap.add_argument("--sys-len", type=int, default=4000)
    ap.add_argument("--task-len", type=int, default=2000)
    ap.add_argument("--gen-len", type=int, default=150)
    ap.add_argument("--tool-len", type=int, default=800)
    ap.add_argument("--pause-batch", type=int, default=64, help="sessions started together in phase 1")
    ap.add_argument("--resume-batch", type=int, default=16, help="sessions resumed together in phase 2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--trace", help="pause real agent sessions (prepare_traces.py output) at a random turn "
                                    "instead of synthetic ones; needs at least max(--sessions-list) sessions")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.dry_run:
        eng = FakeEngine()
        a.model = a.model or "fake"
    elif a.engine == "sglang":
        eng = SglEngine(a.model, a.max_model_len, a.mem_frac, json.loads(a.engine_args))
    else:
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # lets cache_snapshot see the block pool
        eng = VllmEngine(a.model, a.max_model_len, a.gpu_util, json.loads(a.engine_args))
    info = eng.info()
    print(f"[info] engine {eng.name}, cache {info}", flush=True)

    toks = Tokens(eng.vocab, a.seed)
    system = toks.take(a.sys_len)
    paused = None
    if a.trace:
        from traces import load_paused_prefixes
        nmax = max(int(x) for x in a.sessions_list.split(","))
        system, paused = load_paused_prefixes(a.trace, eng.tokenizer, nmax, a.seed, max_ctx=a.max_model_len)
        a.sys_len = len(system)
        lens = sorted(len(system) + len(p["prefix"]) for p in paused)
        print(f"[info] real sessions: {len(paused)} paused; system prompt {len(system)} tokens; paused context "
              f"median {lens[len(lens) // 2]}, p90 {lens[int(0.9 * (len(lens) - 1))]}", flush=True)
        if len(paused) < nmax:
            print(f"[warn] only {len(paused)} distinct sessions; larger N levels are skipped", file=sys.stderr)
    eng.generate([Tokens(eng.vocab, 999).take(256)], 4)  # warmup

    rows, summary = [], []
    for n in [int(x) for x in a.sessions_list.split(",")]:
        flushed = eng.flush()
        t0 = time.time()
        if paused is not None:
            if n > len(paused):
                continue
            prompts = [system + p["prefix"] for p in paused[:n]]
            gens = [p["gen_len"] for p in paused[:n]]
            tools = [p["tool"] for p in paused[:n]]
        else:
            tasks = [toks.take(a.task_len) for _ in range(n)]
            prompts = [system + t for t in tasks]
            gens = [a.gen_len] * n
            tools = None
        replies = []
        for i in range(0, n, a.pause_batch):                       # phase 1: pause N sessions
            replies += [g for g, _ in eng.generate(prompts[i:i + a.pause_batch], gens[i:i + a.pause_batch])]
        snap = eng.cache_snapshot()
        if tools is None:
            tools = [toks.take(a.tool_len) for _ in range(n)]
        res = []
        for i in range(0, n, a.resume_batch):                      # phase 2: resume, oldest first
            batch = [prompts[j] + replies[j] + tools[j] for j in range(i, min(n, i + a.resume_batch))]
            res += [c for _, c in eng.generate(batch, 1)]
        kept_full = kept_prompt = kept_sys = 0
        fracs, recomputed = [], []
        for j in range(n):
            reusable = len(prompts[j]) + len(replies[j])
            c = res[j]
            if c is None:
                continue
            c = int(c)
            kf, kp, ks = c >= reusable - 16, c >= len(prompts[j]) - 16, c >= a.sys_len - 16
            kept_full += kf; kept_prompt += kp; kept_sys += ks
            fracs.append(min(1.0, c / reusable))
            recomputed.append(reusable + len(tools[j]) - c)
            rows.append({"model": a.model, "variant": a.variant, "engine": eng.name, "n_sessions": n,
                         "session": j, "reusable": reusable, "cached": c, "kept_full": int(kf),
                         "kept_prompt": int(kp), "kept_sys": int(ks)})
        m = len(fracs) or 1
        s = {"model": a.model, "variant": a.variant, "engine": eng.name, "n_sessions": n,
             "flushed": flushed, "pct_kept_full": round(100 * kept_full / m, 1),
             "pct_kept_prompt": round(100 * kept_prompt / m, 1), "pct_kept_sys": round(100 * kept_sys / m, 1),
             "mean_reused_frac": round(statistics.mean(fracs), 3) if fracs else None,
             "mean_recomputed": round(statistics.mean(recomputed)) if recomputed else None,
             "wall_s": round(time.time() - t0, 1), **{f"cache_{k}": v for k, v in snap.items()}}
        summary.append(s)
        print(f"N={n:5d}  resume point fully kept {s['pct_kept_full']:5.1f}%  prompt kept "
              f"{s['pct_kept_prompt']:5.1f}%  system prompt kept {s['pct_kept_sys']:5.1f}%  "
              f"mean reused {s['mean_reused_frac']}  {snap}  ({s['wall_s']} s)", flush=True)
        if not fracs:
            print("[warn] engine reported no cached-token counts; results are empty", file=sys.stderr)

    if hasattr(eng, "close"):
        eng.close()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if rows:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    with open(a.out.replace(".csv", "_summary.json"), "w") as f:
        json.dump({"info": info, "args": vars(a), "levels": summary}, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
