"""E5 step 1-2: multi-turn tool-use conversations with real and corrupted tool outputs.

Runs When2Tool's own agent loop (`evaluate_batched`, record_mode="full", greedy) with
two thin hooks installed on its module:
  build_envs     - remembers which task each environment list belongs to
  route_tool_call- logs every tool result; in the corrupted condition replaces the
                   first successful tool result of each conversation with
                   corrupt.corrupt(...) (same format, similar length)
Greedy decoding makes the two conditions identical up to the first tool result, so
they differ only in what the tool returned (checked: `same_first_call`).

Conversations: single-hop tasks (stratified per tier, test + train; train feeds the
P3-specific probe) and multi-hop tasks, in `force_tool` (every task calls, incl. easy
ones) and the default `current` mode (voluntary calls).

Output: results/e5/conv_<split>_<mode>_<cond>.<hash>.jsonl, one item per task with
the full messages, per-round prompt texts (trace) and the tool-result log.

Usage:  python -m toolmech.conversations [--per_tier 150] [--modes force_tool,current]
"""
import argparse
import json
import os
import time

import numpy as np

from toolmech.config import load_config, load_tasks, task_env
from toolmech.corrupt import corrupt, payload


class ToolHooks:
    def __init__(self, utils_mod, condition, rng, pool):
        self.u = utils_mod
        self.cond, self.rng, self.pool = condition, rng, pool
        self.owner, self.log = {}, {}
        self._orig_build, self._orig_route = utils_mod.build_envs, utils_mod.route_tool_call

    def __enter__(self):
        u = self.u

        def build_envs(task, include_metadata=False):
            envs, tools = self._orig_build(task, include_metadata)
            self.owner[id(envs)] = (str(task["id"]), task_env(task))
            return envs, tools

        def route_tool_call(envs, tool_name, arguments):
            res = self._orig_route(envs, tool_name, arguments)
            tid, env = self.owner.get(id(envs), ("?", "?"))
            log = self.log.setdefault(tid, [])
            entry = {"hop": len(log), "tool": tool_name, "arguments": arguments, "real": res}
            if self.cond == "corrupt" and not any(e.get("info", {}).get("corrupted") for e in log):
                new, info = corrupt(res, f"{env}/{tool_name}", self.rng, self.pool)
                entry["info"] = info
                if info["corrupted"]:
                    entry["returned"] = new
                    res = new
            log.append(entry)
            return res

        u.build_envs, u.route_tool_call = build_envs, route_tool_call
        return self

    def __exit__(self, *a):
        self.u.build_envs, self.u.route_tool_call = self._orig_build, self._orig_route


def select(cfg, split, per_tier, seed=0):
    tasks = load_tasks(cfg, split)
    if not per_tier:
        return tasks
    rng = np.random.default_rng(seed)
    out = []
    for tier in ("easy", "medium", "hard"):
        g = [t for t in tasks if t.get("difficulty") == tier]
        by_env = {}
        for t in g:
            by_env.setdefault(task_env(t), []).append(t)
        for v in by_env.values():
            rng.shuffle(v)
        envs, picked, i = sorted(by_env), [], 0
        while len(picked) < min(per_tier, len(g)):
            e = envs[i % len(envs)]
            if by_env[e]:
                picked.append(by_env[e].pop())
            i += 1
        out += picked
    return out


def build_pool(items_log):
    pool = {}
    for tid, (env, log) in items_log.items():
        for e in log:
            k, v = payload(e["real"])
            if k is not None:
                pool.setdefault(f"{env}/{e['tool']}/{k}", []).append(v)
    return pool


def run_condition(cfg, model, tasks, mode, cond, pool, seed):
    import utils  # When2Tool
    rng = np.random.default_rng(seed)
    with ToolHooks(utils, cond, rng, pool) as H:
        items = utils.evaluate_batched(tasks, model, max_rounds=cfg["vllm"]["max_rounds"], record_mode="full",
                                       prompt_mode=mode, require_reasoning=False, tool_format="xml")
    out = []
    for t, it in zip(tasks, items):
        _, boxed, cleaned, correct = utils.item_final_eval(it)
        out.append({"id": str(t["id"]), "env": task_env(t), "tier": t.get("difficulty"), "mode": mode, "cond": cond,
                    "multi_step": bool(t.get("multi_step")), "gold": t["expected"]["answer"],
                    "final_answer": cleaned, "correct": bool(correct), "tool_calls": int(it.get("tool_calls", 0)),
                    "rounds": int(it.get("rounds", 0)), "messages": it.get("output"),
                    "trace": [{"prompt_text": tr.get("prompt_text"), "raw": tr.get("model_raw_output"),
                               "type": (tr.get("parsed_output") or {}).get("type"),
                               "tool_name": (tr.get("parsed_output") or {}).get("tool_name"),
                               "tool_result": tr.get("tool_result")} for tr in it.get("trace", [])],
                    "tool_log": H.log.get(str(t["id"]), [])})
    return out


def save(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, path)


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def run(cfg, splits, modes, per_tier, gpu_util, step="e5"):
    import copy
    from toolmech.config import Config
    from toolmech.labels import _Model
    mcfg = Config(copy.deepcopy(dict(cfg)))      # runtime-only memory setting; cfg (and its hash) unchanged
    if gpu_util:
        mcfg["vllm"]["gpu_memory_utilization"] = gpu_util
    model = _Model(mcfg, temperature=0.0, top_p=1.0, top_k=-1, max_new_tokens=cfg["sampling"]["max_new_tokens"])
    for split in splits:
        tasks = select(cfg, split, per_tier if (not split.startswith("multihop") or step != "e5") else 0)
        for mode in modes:
            paths = {c: cfg.path(step, f"conv_{split}_{mode}_{c}", "jsonl") for c in ("real", "corrupt")}
            if not os.path.exists(paths["real"]):
                t0 = time.time()
                save(paths["real"], run_condition(cfg, model, tasks, mode, "real", {}, cfg["seed"]))
                print(f"{split} {mode} real n={len(tasks)} {time.time() - t0:.0f}s", flush=True)
            real = load(paths["real"])
            if not os.path.exists(paths["corrupt"]):
                # pool of real payloads from every real run so far (same env/tool/field)
                logs = {}
                for p in os.listdir(os.path.dirname(paths["real"])):
                    if "_real." in p and p.endswith(".jsonl"):
                        for r in load(os.path.join(os.path.dirname(paths["real"]), p)):
                            logs[f"{p}:{r['id']}"] = (r["env"], r["tool_log"])
                pool = build_pool(logs)
                called = [t for t, r in zip(tasks, real) if r["tool_log"]]
                t0 = time.time()
                rows = run_condition(cfg, model, called, mode, "corrupt", pool, cfg["seed"] + 1)
                rmap = {r["id"]: r for r in real}
                for r in rows:
                    a, b = rmap[r["id"]]["tool_log"], r["tool_log"]
                    r["same_first_call"] = bool(a and b and a[0]["tool"] == b[0]["tool"]
                                                and a[0]["arguments"] == b[0]["arguments"])
                save(paths["corrupt"], rows)
                print(f"{split} {mode} corrupt n={len(called)} {time.time() - t0:.0f}s "
                      f"same_first_call={np.mean([r['same_first_call'] for r in rows]):.3f} "
                      f"corrupted={np.mean([any(e.get('info', {}).get('corrupted') for e in r['tool_log']) for r in rows]):.3f}",
                      flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="test,multihop_test,train,multihop_train")
    ap.add_argument("--modes", default="force_tool,current")
    ap.add_argument("--per_tier", type=int, default=150)
    ap.add_argument("--gpu_util", type=float, default=None)
    ap.add_argument("--step", default="e5", help="results subfolder (e5_smoke for tests)")
    a = ap.parse_args()
    cfg = load_config()
    run(cfg, a.splits.split(","), a.modes.split(","), a.per_tier, a.gpu_util, a.step)


if __name__ == "__main__":
    main()
