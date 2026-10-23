#!/usr/bin/env python3
"""
Analytic per-session state size for dense and hybrid LLMs, from config.json.

Splits a paused session's GPU state into:
  - recurrent state (fixed size, from Mamba-2 / Gated DeltaNet layers)
  - KV cache (grows per token, from full-attention layers)
  - sliding-window KV (capped at the window, if any)

No GPU needed. Cross-check the result against vllm_capacity.py, which
measures what vLLM actually allocates.

Usage:
  python state_sizes.py --model Qwen/Qwen3.5-4B --out results/qwen35_4b_sizes.json
  python state_sizes.py --config path/to/config.json --out sizes.json
"""
import argparse
import json
import sys

DTYPE_BYTES = {"float32": 4, "fp32": 4, "bfloat16": 2, "bf16": 2,
               "float16": 2, "fp16": 2, "fp8": 1}


def load_config(model=None, config_path=None, revision=None):
    if config_path:
        with open(config_path) as f:
            cfg = json.load(f)
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(model, "config.json", revision=revision)
        with open(path) as f:
            cfg = json.load(f)
    # Multimodal checkpoints nest the language model config.
    for key in ("text_config", "llm_config", "language_config"):
        if isinstance(cfg.get(key), dict) and "num_hidden_layers" in cfg[key]:
            print(f"[info] using nested '{key}'", file=sys.stderr)
            return cfg[key]
    return cfg


def need(cfg, *names):
    for n in names:
        if cfg.get(n) is not None:
            return cfg[n]
    raise KeyError(f"config has none of {names}; add the right field name for this model")


def classify_layers(cfg):
    """Return a list of layer kinds: 'attn', 'swa', 'gdn', 'mamba', 'other'."""
    n = cfg.get("num_hidden_layers")
    if "layer_types" in cfg and cfg["layer_types"]:
        kinds = []
        for t in cfg["layer_types"]:
            t = str(t).lower()
            if t in ("full_attention", "attention", "global"):
                kinds.append("attn")
            elif "sliding" in t or t == "local":
                kinds.append("swa")
            elif "linear" in t or "delta" in t:
                kinds.append("gdn")
            elif "mamba" in t:
                kinds.append("mamba")
            else:
                kinds.append("other")
        return kinds, "layer_types"
    if cfg.get("hybrid_override_pattern"):
        # Nemotron-H style: M = Mamba-2, * = attention, - = MLP, E = MoE
        m = {"M": "mamba", "*": "attn"}
        return [m.get(c, "other") for c in cfg["hybrid_override_pattern"]], "hybrid_override_pattern"
    if cfg.get("full_attention_interval"):
        k = int(cfg["full_attention_interval"])
        return ["attn" if (i + 1) % k == 0 else "gdn" for i in range(n)], "full_attention_interval"
    if cfg.get("attn_layer_indices") is not None or cfg.get("attn_layer_period"):
        if cfg.get("attn_layer_indices") is not None:
            idx = set(cfg["attn_layer_indices"])
        else:
            p, off = cfg["attn_layer_period"], cfg.get("attn_layer_offset", 0)
            idx = {i for i in range(n) if i % p == off}
        return ["attn" if i in idx else "mamba" for i in range(n)], "attn_layer_indices/period"
    return ["attn"] * n, "dense (no hybrid fields found)"


def kv_bytes_per_token_per_layer(cfg, kv_dtype_bytes):
    n_heads = need(cfg, "num_attention_heads")
    kv_heads = cfg.get("num_key_value_heads") or n_heads
    head_dim = cfg.get("head_dim") or need(cfg, "hidden_size") // n_heads
    return 2 * kv_heads * head_dim * kv_dtype_bytes


def gdn_state_bytes_per_layer(cfg, ssm_b, conv_b):
    vh = need(cfg, "linear_num_value_heads")
    kh = need(cfg, "linear_num_key_heads")
    kd = need(cfg, "linear_key_head_dim")
    vd = need(cfg, "linear_value_head_dim")
    ck = need(cfg, "linear_conv_kernel_dim", "conv_kernel")
    recurrent = vh * kd * vd * ssm_b
    conv_dim = 2 * kh * kd + vh * vd
    conv = (ck - 1) * conv_dim * conv_b
    return recurrent, conv


def mamba_state_bytes_per_layer(cfg, ssm_b, conv_b):
    heads = need(cfg, "mamba_num_heads", "mamba_n_heads")
    hd = need(cfg, "mamba_head_dim", "mamba_d_head")
    ds = need(cfg, "ssm_state_size", "mamba_d_state", "state_size")
    groups = cfg.get("n_groups") or cfg.get("mamba_n_groups") or 1
    ck = need(cfg, "conv_kernel", "mamba_d_conv")
    recurrent = heads * hd * ds * ssm_b
    conv_dim = heads * hd + 2 * groups * ds
    conv = (ck - 1) * conv_dim * conv_b
    return recurrent, conv


def compute(cfg, kv_dtype="bf16", ssm_dtype="fp32", conv_dtype="bf16"):
    kinds, source = classify_layers(cfg)
    kvb, ssmb, convb = DTYPE_BYTES[kv_dtype], DTYPE_BYTES[ssm_dtype], DTYPE_BYTES[conv_dtype]
    counts = {k: kinds.count(k) for k in ("attn", "swa", "gdn", "mamba", "other")}

    per_layer_kv = kv_bytes_per_token_per_layer(cfg, kvb) if (counts["attn"] or counts["swa"]) else 0
    rec = conv = 0
    if counts["gdn"]:
        r, c = gdn_state_bytes_per_layer(cfg, ssmb, convb)
        rec += r * counts["gdn"]; conv += c * counts["gdn"]
    if counts["mamba"]:
        r, c = mamba_state_bytes_per_layer(cfg, ssmb, convb)
        rec += r * counts["mamba"]; conv += c * counts["mamba"]

    window = cfg.get("sliding_window") or 0
    out = {
        "layer_source": source,
        "layer_counts": counts,
        "dtypes": {"kv": kv_dtype, "ssm": ssm_dtype, "conv": conv_dtype},
        "kv_bytes_per_token": per_layer_kv * counts["attn"],
        "swa_bytes_per_token": per_layer_kv * counts["swa"],
        "swa_window_tokens": window,
        "recurrent_state_bytes": rec,
        "conv_state_bytes": conv,
        "fixed_state_bytes": rec + conv,  # SSM part of a paused session
    }
    kvpt = out["kv_bytes_per_token"]
    out["crossover_tokens"] = (out["fixed_state_bytes"] / kvpt) if kvpt else None
    return out


def table(sizes, contexts):
    rows = []
    for L in contexts:
        swa = sizes["swa_bytes_per_token"] * (min(L, sizes["swa_window_tokens"]) if sizes["swa_window_tokens"] else L)
        kv = sizes["kv_bytes_per_token"] * L + swa
        rows.append({"context": L,
                     "fixed_state_MiB": sizes["fixed_state_bytes"] / 2**20,
                     "kv_MiB": kv / 2**20,
                     "total_MiB": (sizes["fixed_state_bytes"] + kv) / 2**20})
    return rows


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model")
    g.add_argument("--config")
    ap.add_argument("--revision")
    ap.add_argument("--kv-dtype", default="bf16", choices=DTYPE_BYTES)
    ap.add_argument("--ssm-dtype", default="fp32", choices=DTYPE_BYTES,
                    help="dtype of the recurrent state as your engine stores it (check vLLM's mamba_ssm_cache_dtype)")
    ap.add_argument("--conv-dtype", default="bf16", choices=DTYPE_BYTES)
    ap.add_argument("--contexts", default="1000,2000,4000,8000,16000,32000,64000,128000")
    ap.add_argument("--out")
    a = ap.parse_args()

    cfg = load_config(a.model, a.config, a.revision)
    sizes = compute(cfg, a.kv_dtype, a.ssm_dtype, a.conv_dtype)
    sizes["model"] = a.model or a.config
    sizes["table"] = table(sizes, [int(x) for x in a.contexts.split(",")])

    print(f"\n{sizes['model']}  (layers from: {sizes['layer_source']})")
    print(f"  layer counts: {sizes['layer_counts']}")
    print(f"  fixed SSM state per session: {sizes['fixed_state_bytes']/2**20:.1f} MiB "
          f"(recurrent {sizes['recurrent_state_bytes']/2**20:.1f}, conv {sizes['conv_state_bytes']/2**20:.2f})")
    print(f"  full-attention KV per token: {sizes['kv_bytes_per_token']/1024:.1f} KiB")
    if sizes["crossover_tokens"]:
        print(f"  crossover (KV == fixed state): ~{sizes['crossover_tokens']:,.0f} tokens")
    print(f"\n  {'context':>8} {'state MiB':>10} {'KV MiB':>10} {'total MiB':>10}")
    for r in sizes["table"]:
        print(f"  {r['context']:>8} {r['fixed_state_MiB']:>10.1f} {r['kv_MiB']:>10.1f} {r['total_MiB']:>10.1f}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(sizes, f, indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
