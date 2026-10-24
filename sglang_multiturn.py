#!/usr/bin/env python3
"""
Same multi-turn agent replay as multiturn_bench.py, but on SGLang.

Uses SGLang's offline Engine and reads meta_info["cached_tokens"] for each
resumed turn. Workload, lengths and seeds are identical to the vLLM runs, so
the CSVs are directly comparable and analyze_multiturn.py reads them as-is.

Install SGLang in its OWN venv (it pins a different torch than vLLM 0.31):
  python -m venv /workspace/sgl && . /workspace/sgl/bin/activate
  pip install -U pip && pip install "sglang[all]"

Usage:
  python sglang_multiturn.py --model Qwen/Qwen3.5-4B --variant sgl_auto \
      --engine-args '{}' --out results/mt_qwen35_4b_sgl_auto.csv
  python sglang_multiturn.py --model Qwen/Qwen3.5-4B --variant sgl_nobuf \
      --engine-args '{"mamba_radix_cache_strategy":"no_buffer"}' --out results/mt_qwen35_4b_sgl_nobuf.csv
"""
import argparse
import csv
import json
import os
import statistics
import sys
import time

# Reuse the workload generator so both engines see identical sessions.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiturn_bench import Tokens, make_sessions  # noqa: E402


class SglEngine:
    def __init__(self, model, max_len, mem_frac, engine_args):
        import sglang as sgl
        args = dict(model_path=model, context_length=max_len, mem_fraction_static=mem_frac,
                    trust_remote_code=True, log_level="warning")
        args.update(engine_args)
        self.llm = sgl.Engine(**args)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.vocab = self.tokenizer.vocab_size
        try:
            info = self.llm.get_server_info()
            self.strategy = info.get("mamba_radix_cache_strategy", "?")
        except Exception:
            self.strategy = "?"

    def run(self, ids, max_tokens):
        sp = {"max_new_tokens": max_tokens, "temperature": 0.0, "ignore_eos": True}
        t0 = time.perf_counter()
        out = self.llm.generate(input_ids=ids, sampling_params=sp)
        ms = (time.perf_counter() - t0) * 1e3
        if isinstance(out, list):
            out = out[0]
        meta = out.get("meta_info", {})
        gen = out.get("output_ids") or meta.get("output_ids") or []
        if len(gen) != max_tokens:  # older versions return text only
            gen = [7] * max_tokens if not gen else gen
        return list(gen), meta.get("cached_tokens"), ms

    def close(self):
        try:
            self.llm.shutdown()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", default="sgl_auto")
    ap.add_argument("--engine-args", default="{}", help="JSON dict passed to sglang.Engine(...)")
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--mem-frac", type=float, default=0.80)
    ap.add_argument("--sessions", type=int, default=6)
    ap.add_argument("--turns", type=int, default=15)
    ap.add_argument("--sys-len", type=int, default=4000)
    ap.add_argument("--task-len", type=int, default=500)
    ap.add_argument("--gen-len", type=int, default=150)
    ap.add_argument("--tool-median", type=int, default=800)
    ap.add_argument("--tool-sigma", type=float, default=1.0)
    ap.add_argument("--tool-min", type=int, default=30)
    ap.add_argument("--tool-max", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trace", help="replay real agent sessions from prepare_traces.py")
    ap.add_argument("--trace-sessions", type=int, default=12)
    ap.add_argument("--trace-turns", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    eng = SglEngine(a.model, a.max_model_len, a.mem_frac, json.loads(a.engine_args))
    print(f"[info] SGLang mamba_radix_cache_strategy = {eng.strategy}", flush=True)

    toks = Tokens(eng.vocab, a.seed)
    if a.trace:
        from traces import load_trace, describe
        system, tsess = load_trace(a.trace, eng.tokenizer, a.max_model_len, a.trace_sessions, a.trace_turns)
        print("[info] " + describe(system, tsess), flush=True)
        sessions = [{"session": t["session"], "task_len": len(t["task"]), "tools": [len(x["tool"]) for x in t["turns"]],
                     "task_ids": t["task"], "tool_ids": [x["tool"] for x in t["turns"]],
                     "gen_lens": [x["gen_len"] for x in t["turns"]]} for t in tsess]
    else:
        sessions = make_sessions(a.sessions, a.turns, a.sys_len, a.task_len, a.gen_len, a.tool_median,
                                 a.tool_sigma, a.tool_min, a.tool_max, a.max_model_len, a.seed)
        system = toks.take(a.sys_len)
    eng.run(Tokens(eng.vocab, 999).take(256), 4)  # warmup

    rows = []
    for s in sessions:
        hist = system + (s["task_ids"] if "task_ids" in s else toks.take(s["task_len"]))
        new_tokens = s["task_len"]
        for turn in range(len(s["tools"]) + 1):
            _, cached, resume_ms = eng.run(hist, 1)
            row = {"model": a.model, "variant": a.variant, "session": s["session"], "turn": turn,
                   "prompt_len": len(hist), "new_tokens": new_tokens,
                   "cached_tokens": cached if cached is not None else "", "resume_ms": round(resume_ms, 2)}
            if cached is not None:
                rec = len(hist) - int(cached)
                row["recomputed"] = rec
                row["wasted"] = max(0, rec - new_tokens)
            rows.append(row)
            g = s["gen_lens"][turn] if "gen_lens" in s and turn < len(s["gen_lens"]) else a.gen_len
            gen, _, _ = eng.run(hist, g)
            hist = hist + gen
            if turn < len(s["tools"]):
                t = s["tools"][turn]
                hist = hist + (s["tool_ids"][turn] if "tool_ids" in s else toks.take(t))
                new_tokens = t
        w = [r["wasted"] for r in rows if r["session"] == s["session"] and r["turn"] > 0 and "wasted" in r]
        print(f"session {s['session']}: final ctx {len(hist):,}"
              + (f", mean wasted/turn {statistics.mean(w):.0f}" if w else ", no cached_tokens reported"), flush=True)

    eng.close()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    keys = ["model", "variant", "session", "turn", "prompt_len", "new_tokens", "cached_tokens",
            "recomputed", "wasted", "resume_ms"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
