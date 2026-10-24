"""
Load a trace written by prepare_traces.py and turn it into token IDs for one model.

Each session becomes {"session", "task": [ids], "turns": [{"gen_len": int, "tool": [ids]}]},
cut so that the whole session fits in max_ctx tokens. Reply lengths are the recorded
assistant replies, measured in this model's tokens and clipped to [reply_min, reply_max].
"""
import json
import statistics


class _HashTokenizer:
    """Stand-in tokenizer for dry runs: one ID per whitespace-separated word."""

    def encode(self, text, add_special_tokens=False):
        return [1000 + (hash(w) % 50000) for w in text.split()]


def encode(tok, text):
    tok = tok or _HashTokenizer()
    try:
        return list(tok.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tok.encode(text))


def load_trace(path, tokenizer, max_ctx, n_sessions=None, max_turns=None, reply_min=16, reply_max=2048,
               margin=256):
    with open(path) as f:
        tr = json.load(f)
    system = encode(tokenizer, tr["system"])
    out = []
    for i, s in enumerate(tr["sessions"][:n_sessions] if n_sessions else tr["sessions"]):
        task = encode(tokenizer, s["task"])
        ctx = len(system) + len(task)
        if ctx + reply_min + margin > max_ctx:
            continue
        turns = []
        for t in s["turns"][:max_turns] if max_turns else s["turns"]:
            g = min(reply_max, max(reply_min, len(encode(tokenizer, t["reply"]))))
            tool = encode(tokenizer, t["tool"])
            if ctx + g + len(tool) + reply_min + margin > max_ctx:
                break
            turns.append({"gen_len": g, "tool": tool})
            ctx += g + len(tool)
        out.append({"session": i, "traj_id": s.get("traj_id"), "task": task, "turns": turns})
    return system, out


def describe(system, sessions):
    t = [len(s["turns"]) for s in sessions]
    g = [x["gen_len"] for s in sessions for x in s["turns"]]
    o = [len(x["tool"]) for s in sessions for x in s["turns"]]
    f = [len(system) + len(s["task"]) + sum(x["gen_len"] + len(x["tool"]) for x in s["turns"]) for s in sessions]
    med = lambda xs: int(statistics.median(xs)) if xs else 0
    p90 = lambda xs: sorted(xs)[int(0.9 * (len(xs) - 1))] if xs else 0
    return (f"trace: {len(sessions)} sessions; system prompt {len(system)} tokens; turns median {med(t)} "
            f"(p90 {p90(t)}); reply tokens median {med(g)} (p90 {p90(g)}); tool output tokens median {med(o)} "
            f"(p90 {p90(o)}); final context median {med(f)} (p90 {p90(f)})")
