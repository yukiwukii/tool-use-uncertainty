"""E1: the label table (M2) for every When2Tool task.

Per task:  need, tier, env                      <- When2Tool E0 (hard_no_tool, one sample)
           fail_rate, SE, n_clusters             <- k no-tool samples (hard_no_tool loop, T=1)
           act (greedy), call_rate, late_share   <- n call samples per mode (cfg.modes: primary
                                                    sparse_tool, secondary current); suffixed _<mode>
           act_w2t                               <- mean tool use over E0's 3 run_eval runs (if present)

Sampling uses the *full* distribution (T=1, top_p=1, top_k=-1). When2Tool's vLLM
wrapper otherwise inherits Qwen's generation_config (top_k=20, top_p=0.8).

Usage:  python -m toolmech.labels [--splits train,test] [--stage notool|calls|merge|all]
"""
import argparse
import json
import math
import os
import re
from collections import Counter

import numpy as np
import pandas as pd

from toolmech.config import ROOT, W2T_ROOT, load_config, load_tasks, task_env
from toolmech.prompts import load_prompts

TOOL_TAG = "<tool_call>"


# ---------------------------------------------------------------------------
# Semantic entropy over short answers (no NLI: normalised exact / numeric match)
# ---------------------------------------------------------------------------
_NUM = re.compile(r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?")
_THOUSANDS = re.compile(r"[-+]?\d{1,3}(,\d{3})+(\.\d+)?")


def _canon(x):
    """Hashable canonical form: numbers as rounded floats, containers recursively."""
    if isinstance(x, bool):
        return x
    if isinstance(x, int):
        return x  # exact; big ints (hashes, products) overflow float
    if isinstance(x, float):
        return int(x) if x.is_integer() else round(x, 9)
    if isinstance(x, (list, tuple)):
        return tuple(_canon(v) for v in x)
    if isinstance(x, dict):
        return tuple(sorted((str(k), _canon(v)) for k, v in x.items()))
    if isinstance(x, (set, frozenset)):  # "{1, 2}" parses as a set; order-free
        return ("set",) + tuple(sorted((_canon(v) for v in x), key=repr))
    try:
        hash(x)
        return x
    except TypeError:  # any other unhashable literal: fall back to its repr
        return ("repr", repr(x))


def _num(t):
    """Numeric string -> canonical number (exact int when integral)."""
    try:
        return int(t)
    except ValueError:
        return _canon(float(t))


def answer_key(ans):
    """Cluster key for a boxed answer. Missing answers (no \\boxed{}) share one key."""
    from utils import normalize_scalar, parse_structured  # When2Tool, same as its grader
    if ans is None or not str(ans).strip():
        return ("<none>",)
    if _THOUSANDS.fullmatch(str(ans).strip()):  # "1,000" would parse as the tuple (1, 0)
        return ("n", _num(str(ans).strip().replace(",", "")))
    s = parse_structured(ans)
    if s is not None:
        c = _canon(s)
        return ("n", c) if isinstance(c, (int, float)) and not isinstance(c, bool) else ("s", c)
    t = normalize_scalar(ans)
    t_num = t.replace(",", "")
    if _NUM.fullmatch(t_num):
        return ("n", _num(t_num))
    return ("t", t)


def semantic_entropy(answers):
    """Discrete SE (nats) = entropy of cluster frequencies; returns (SE, n_clusters)."""
    counts = Counter(answer_key(a) for a in answers)
    n = sum(counts.values())
    if n == 0:
        return float("nan"), 0
    p = np.array(list(counts.values()), dtype=float) / n
    return float(-(p * np.log(p)).sum()), len(counts)


def call_position(text):
    """(called, late): a call anywhere in the first assistant output, and whether
    non-whitespace text precedes it (M1: TUL misses such calls)."""
    i = (text or "").find(TOOL_TAG)
    if i < 0:
        return False, False
    return True, bool(text[:i].strip())


# ---------------------------------------------------------------------------
# vLLM model: When2Tool's backend, plus memory fraction and seed
# ---------------------------------------------------------------------------
class _Model:
    """Duck-types When2Tool's AgentModel for `evaluate_batched`."""

    def __init__(self, cfg, temperature, top_p, top_k, max_new_tokens):
        from transformers import AutoTokenizer, GenerationConfig
        from vllm import LLM, SamplingParams
        from model import VLLMAgentBackend  # When2Tool
        from utils import get_system_prompt

        eng = VLLMAgentBackend.__new__(VLLMAgentBackend)
        eng.SamplingParams = SamplingParams
        eng.max_new_tokens = max_new_tokens
        eng.enable_thinking = False
        eng.tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
        gc = GenerationConfig.from_pretrained(cfg["model_id"])
        gc.temperature, gc.top_p, gc.top_k = float(temperature), float(top_p), int(top_k)
        gc.repetition_penalty = None
        eng.generation_config = gc
        v = cfg["vllm"]
        eng.llm = LLM(model=cfg["model_id"], tensor_parallel_size=1, dtype="bfloat16",
                      max_model_len=v["max_model_len"],
                      gpu_memory_utilization=v["gpu_memory_utilization"], seed=cfg["seed"])
        self.engine = eng
        self.llm = eng.llm
        self.tokenizer = eng.tokenizer
        self.system_prompt = get_system_prompt("xml")

    def generate_batch(self, messages_batch, tools_batch, prefills=None):
        return self.engine.generate_batch(messages_batch, tools_batch, prefills=prefills)

    def sampling(self, **kw):
        return self.engine.SamplingParams(**kw)


def _sampling_cfg(cfg):
    s = cfg["sampling"]
    return dict(temperature=s["temperature"], top_p=s["top_p"], top_k=s.get("top_k", -1))


def _subset(items, limit):
    """First `limit` items per (env, tier); all items if limit is 0."""
    if not limit:
        return items
    seen, out = Counter(), []
    for x in items:
        key = (task_env(x) if "environments" in x else x.get("env"), x.get("difficulty"))
        if seen[key] < limit:
            seen[key] += 1
            out.append(x)
    return out


# ---------------------------------------------------------------------------
# Stage 1: k no-tool samples per task (hard_no_tool loop) -> fail_rate, SE
# ---------------------------------------------------------------------------
def sample_notool(cfg, model, split, step="e1", limit=0):
    from utils import evaluate_batched, item_final_eval
    path = cfg.path(step, f"notool_samples_{split}", "parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    k = cfg["sampling"]["k_notool"]
    tasks = _subset(load_tasks(cfg, split), limit)
    cached = {r["id"]: r["prompt_text"] for r in load_prompts(cfg, split, "no_tool")}
    reps = [t for t in tasks for _ in range(k)]
    items = evaluate_batched(reps, model, max_rounds=cfg["vllm"]["max_rounds"], record_mode="lite",
                             prompt_mode="hard_no_tool", require_reasoning=False, tool_format="xml")
    rows = []
    for j, (t, it) in enumerate(zip(reps, items)):
        _, boxed, cleaned, correct = item_final_eval(it)
        first_prompt = (it.get("trace") or [{}])[0].get("prompt_text")
        rows.append({
            "id": t["id"], "sample": j % k, "answer": cleaned, "correct": bool(correct),
            "rounds": int(it.get("rounds", 0)), "rejected_calls": int(it.get("tool_calls", 0) or 0),
            "prompt_matches_cache": (first_prompt == cached.get(t["id"])) if first_prompt else None,
        })
    df = pd.DataFrame(rows)
    df.to_parquet(path)
    return df


# ---------------------------------------------------------------------------
# Stage 2: call samples on the cached `current` prompt -> act, call_rate, late_share
# ---------------------------------------------------------------------------
def call_modes(cfg):
    m = cfg.get("modes", {"primary": "sparse_tool", "secondary": "current"})
    return [m["primary"], m["secondary"]]


def sample_calls(cfg, model, split, step="e1", limit=0, mode="current"):
    path = cfg.path(step, f"call_samples_{split}_{mode}", "parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    recs = _subset(load_prompts(cfg, split, mode), limit)
    prompts = [r["prompt_text"] for r in recs]
    n = cfg["sampling"]["n_call_samples"]
    mt = cfg["sampling"].get("call_max_tokens", 512)
    samp = model.llm.generate(prompts, model.sampling(n=n, max_tokens=mt, seed=cfg["seed"],
                                                      **_sampling_cfg(cfg)))
    greedy = model.llm.generate(prompts, model.sampling(n=1, max_tokens=mt, temperature=0.0))
    rows = []
    for r, s, g in zip(recs, samp, greedy):
        for j, o in enumerate(s.outputs):
            c, late = call_position(o.text)
            rows.append({"id": r["id"], "sample": j, "called": c, "late": late, "greedy": False,
                         "call_char": o.text.find(TOOL_TAG), "text": o.text[:200]})
        c, late = call_position(g.outputs[0].text)
        rows.append({"id": r["id"], "sample": -1, "called": c, "late": late, "greedy": True,
                     "call_char": g.outputs[0].text.find(TOOL_TAG), "text": g.outputs[0].text[:200]})
    df = pd.DataFrame(rows)
    df.to_parquet(path)
    return df


# ---------------------------------------------------------------------------
# Stage 3: merge into labels.parquet
# ---------------------------------------------------------------------------
def _w2t_need(split):
    p = os.path.join(W2T_ROOT, "probe_data", "qwen3-4b-instruct", f"{split}_labels_no_reasoning.json")
    meta = json.load(open(p))["task_meta"]
    return {str(m["id"]): int(m["tool_necessary"]) for m in meta}


def _w2t_act():
    """Mean tool use over E0's run_eval runs (outputs/<model>/current.json: {run_i: [items]})."""
    from utils import item_tool_calls
    f = os.path.join(W2T_ROOT, "outputs", "qwen3-4b-instruct", "current.json")
    if not os.path.exists(f):
        return {}
    acc = {}
    for items in json.load(open(f)).values():
        for it in items:
            acc.setdefault(str(it.get("id")), []).append(int(item_tool_calls(it) > 0))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def merge(cfg, splits, step="e1"):
    out = []
    act_w2t = _w2t_act()
    for split in splits:
        tasks = {str(t["id"]): t for t in load_tasks(cfg, split)}
        need = _w2t_need(split) if not split.startswith("multihop") else {}
        nt = pd.read_parquet(cfg.path(step, f"notool_samples_{split}", "parquet"))
        cs_by_mode = {m: pd.read_parquet(cfg.path(step, f"call_samples_{split}_{m}", "parquet"))
                      for m in call_modes(cfg)}
        for tid, g in nt.groupby(nt["id"].astype(str)):
            se, ncl = semantic_entropy(g["answer"].tolist())
            t = tasks[tid]
            row = {
                "id": tid, "split": split, "env": task_env(t), "tier": t.get("difficulty"),
                "need": need.get(tid, np.nan),
                "fail_rate": 1.0 - g["correct"].mean(), "SE": se, "n_clusters": ncl,
                "act_w2t": act_w2t.get(tid, np.nan),  # E0, `current` mode, Qwen default sampling
            }
            for i, (m, cs) in enumerate(cs_by_mode.items()):
                c = cs[cs["id"].astype(str) == tid]
                samp, gr = c[~c["greedy"]], c[c["greedy"]]
                vals = {
                    "act": int(gr["called"].iloc[0]) if len(gr) else np.nan,
                    "call_rate": samp["called"].mean(),
                    "call_rate_first": (samp["called"] & ~samp["late"]).mean(),
                    "late_share": samp["late"].sum() / max(1, samp["called"].sum()),
                }
                row.update({f"{k}_{m}": v for k, v in vals.items()})
                if i == 0:  # primary mode also under the plain names used downstream
                    row.update(vals)
            out.append(row)
    df = pd.DataFrame(out)
    path = cfg.path(step, "labels", "parquet")
    df.to_parquet(path)
    return df, path


def report(df):
    """E1 pass criteria: no missing labels; SE > 0 for >= 30% of medium tasks."""
    core = ["fail_rate", "SE", "act", "call_rate", "tier"]
    sh = df[df["split"].isin(["train", "test"])]
    missing = int(sh[core + ["need"]].isna().sum().sum())
    med = sh[sh["tier"] == "medium"]
    se_pos = float((med["SE"] > 0).mean()) if len(med) else float("nan")
    print(f"tasks={len(df)}  missing core labels={missing}  medium SE>0 share={se_pos:.3f}")
    print(sh.groupby("tier")[["need", "fail_rate", "SE", "act", "call_rate", "late_share"]].mean().round(3))
    ok = missing == 0 and se_pos >= 0.30
    print("E1 PASS" if ok else "E1 FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test: N tasks per (env, tier), written to results/e1_smoke/")
    ap.add_argument("--stage", default="all", choices=["notool", "calls", "merge", "all"])
    a = ap.parse_args()
    cfg = load_config()
    splits = a.splits.split(",")
    step = "e1_smoke" if a.limit else "e1"
    if a.stage in ("notool", "calls", "all"):
        model = _Model(cfg, max_new_tokens=cfg["sampling"]["max_new_tokens"], **_sampling_cfg(cfg))
        for split in splits:
            if a.stage in ("notool", "all"):
                sample_notool(cfg, model, split, step, a.limit)
            if a.stage in ("calls", "all"):
                for mode in call_modes(cfg):
                    sample_calls(cfg, model, split, step, a.limit, mode)
    if a.stage in ("merge", "all"):
        df, path = merge(cfg, splits, step)
        print(path)
        report(df)


if __name__ == "__main__":
    main()
