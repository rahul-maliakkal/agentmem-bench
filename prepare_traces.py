#!/usr/bin/env python3
"""
Turn real coding-agent trajectories into a replayable trace file.

Source: SWE-smith trajectories (SWE-agent + Claude 3.7 Sonnet on SWE-smith tasks, MIT license),
https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories . Each row's `messages`
column holds one SWE-agent run as a chat: a system prompt (instructions and tool
documentation, shared by all runs), the task (a GitHub issue), then alternating assistant
turns (the agent's reasoning and tool call) and tool/user turns (command output).

The trace keeps the real text of the system prompt, task and tool outputs, and the length of
each assistant reply. The benchmarks replay it: the model generates a reply of the recorded
length (its own tokens, so prefix caching sees exactly what it produced), then the recorded
tool output is appended.

Usage (on the pod; needs huggingface_hub and pyarrow):
  python prepare_traces.py --out traces/swe_smith.json
  python prepare_traces.py --local some.parquet --out traces/test.json     # no download
"""
import argparse
import json
import os
import random
import statistics
import sys


def text_of(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # [{"type": "text", "text": ...}, ...]
        return "".join(text_of(c.get("text") if isinstance(c, dict) else c) for c in content)
    if isinstance(content, dict):
        return text_of(content.get("text") or content.get("content"))
    return str(content)


def parse_messages(raw):
    msgs = json.loads(raw) if isinstance(raw, str) else list(raw)
    out = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or m.get("from") or "").lower()
        text = text_of(m.get("content", m.get("text", m.get("value"))))
        if m.get("tool_calls"):  # function-calling format: keep the call itself as part of the reply
            text += json.dumps(m["tool_calls"])
        out.append((role, text))
    return out


def to_session(msgs):
    """system, task, [(reply_text, tool_text), ...] or None if the run does not fit the pattern."""
    system = next((t for r, t in msgs if r == "system"), None)
    i = next((k for k, (r, _) in enumerate(msgs) if r in ("user", "human")), None)
    if system is None or i is None:
        return None
    task = msgs[i][1]
    turns, k = [], i + 1
    while k < len(msgs):
        r, t = msgs[k]
        if r in ("assistant", "ai", "gpt"):
            reply, tool, k = t, [], k + 1
            while k < len(msgs) and msgs[k][0] not in ("assistant", "ai", "gpt"):
                tool.append(msgs[k][1])
                k += 1
            turns.append((reply, "\n".join(tool)))
        else:
            k += 1
    return (system, task, turns) if turns else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="SWE-bench/SWE-smith-trajectories")
    ap.add_argument("--file", default="data/tool-00001-of-00008.parquet",
                    help="one parquet shard of the 'tool' split (function-calling SWE-agent runs)")
    ap.add_argument("--local", help="use a local parquet file instead of downloading")
    ap.add_argument("--n", type=int, default=256, help="number of sessions to keep")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    path = a.local
    if not path:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(a.repo, a.file, repo_type="dataset")
    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=["messages", "traj_id", "instance_id"])
    rows = table.to_pylist()
    print(f"[info] {len(rows)} trajectories in {os.path.basename(path)}")

    sessions, systems = [], {}
    for row in rows:
        try:
            s = to_session(parse_messages(row["messages"]))
        except Exception:
            s = None
        if s is None:
            continue
        system, task, turns = s
        systems[system] = systems.get(system, 0) + 1
        sessions.append({"traj_id": row.get("traj_id") or row.get("instance_id"), "system": system,
                         "task": task, "turns": [{"reply": r, "tool": t} for r, t in turns]})
    if not sessions:
        sys.exit("ERROR: no trajectory matched the expected chat format; check the 'messages' column")

    # Keep runs that share the most common system prompt, so the shared prefix is real.
    shared, count = max(systems.items(), key=lambda kv: kv[1])
    pool = [s for s in sessions if s["system"] == shared]
    random.Random(a.seed).shuffle(pool)
    pool = pool[:a.n]
    for s in pool:
        del s["system"]
    print(f"[info] parsed {len(sessions)} runs; {count} share the most common system prompt; keeping {len(pool)}")

    chars = lambda xs: (statistics.median(xs), sorted(xs)[int(0.9 * (len(xs) - 1))]) if xs else (0, 0)
    turns = [len(s["turns"]) for s in pool]
    rep = [len(t["reply"]) for s in pool for t in s["turns"]]
    tool = [len(t["tool"]) for s in pool for t in s["turns"]]
    print(f"[stats] system prompt {len(shared):,} chars; turns per run median/p90 {chars(turns)}; "
          f"reply chars {chars(rep)}; tool output chars {chars(tool)}  (about 4 chars per token)")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"source": f"{a.repo}/{a.file}" if not a.local else a.local, "system": shared,
                   "sessions": pool}, f)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
