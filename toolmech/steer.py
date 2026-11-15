"""M5 / E4: steering sweeps with readouts from real replies (PLAN.md M5, E4 steps 1-5).

For each (direction, layer, k) on a fixed stratified test subset:
  manip  - k_notool sampled answers on the `no_tool` prompt -> SE, fail rate,
           unparseable share (manipulation check; M5 "required").
  calls  - n_call_samples sampled first replies (T=1) + 1 greedy reply on the
           <mode> prompt -> call rate, greedy decision, late-call share,
           direct-answer accuracy (share of samples answering directly and correctly),
           abstain rate.
The steered vector is alpha = k * std(H_l . d) * d, with std over the train-split
residuals of the primary mode; the random control uses the alpha of SE^dm
("random direction of equal norm").

Directions come from the primary mode (sparse_tool) for every prompt mode, so the
two modes see the same push and differ only in the response-policy sentence.

Generation is in vLLM (toolmech.vsteer), bf16 like E1, so k = 0 reproduces E1.
Each condition is one parquet in results/e4/steer/ and is skipped if present
(the queue is resumable). Conditions run in priority order (see `queue`).

Usage:  python -m toolmech.steer [--phases A,B,C,D,E,F] [--per_tier 100]
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from toolmech.config import load_config, load_tasks, task_env
from toolmech.prompts import load_prompts

TOOL_TAG = "<tool_call>"


# ---------------------------------------------------------------------------
# Task subset, directions, scales
# ---------------------------------------------------------------------------
def task_subset(cfg, per_tier, split="test", seed=0):
    """`per_tier` tasks per tier, spread evenly over environments (fixed seed)."""
    rng = np.random.default_rng(seed)
    tasks = load_tasks(cfg, split)
    df = pd.DataFrame({"id": [str(t["id"]) for t in tasks], "tier": [t["difficulty"] for t in tasks],
                       "env": [task_env(t) for t in tasks]})
    out = []
    for tier, g in df.groupby("tier"):
        envs = sorted(g.env.unique())
        per_env = {e: list(rng.permutation(g[g.env == e].id.to_numpy())) for e in envs}
        picked, i = [], 0
        while len(picked) < min(per_tier, len(g)):
            e = envs[i % len(envs)]
            if per_env[e]:
                picked.append(per_env[e].pop())
            i += 1
        out += picked
    return out


def load_directions(cfg, mode):
    D = torch.load(cfg.path("e3", f"directions_{mode}", "pt"), weights_only=False)
    return D


def best_layers(cfg, mode, n):
    dec = pd.read_parquet(cfg.path("e3", f"decoding_{mode}", "parquet"))
    d = dec[(dec.label == "SE") & (dec.method == "ridge") & (dec.scope == "all") & (dec.layer >= 0)]
    return [int(l) for l in d.sort_values("value", ascending=False).layer.head(n)]


def random_dir(d, seed):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(d, generator=g)
    return v / v.norm()


def steering_vectors(cfg, mode, layer, names):
    """{name: (unit vector [d], std of projection on train residuals)}."""
    D = load_directions(cfg, mode)
    X = torch.load(cfg.path("e2", f"resid_train_{mode}", "pt"), weights_only=False)["resid"][:, layer].float()
    out = {}
    for n in names:
        if n == "random":
            continue
        v = D[n][layer].float()
        if not torch.isfinite(v).all() or v.norm() == 0:
            continue
        v = v / v.norm()
        out[n] = (v, float((X @ v).std()))
    if "random" in names:
        out["random"] = (random_dir(X.shape[1], 1234 + layer), out["SE^dm"][1])
    return out


# ---------------------------------------------------------------------------
# Readouts
# ---------------------------------------------------------------------------
def _grade(text, gold):
    from utils import clean, compare_values, extract_boxed  # When2Tool
    box = extract_boxed(text or "")
    if not box:
        return None, False
    ans = clean(box)
    return ans, bool(compare_values(ans, gold))


def manip_rows(outs, recs, gold, env):
    from toolmech.labels import is_abstain, semantic_entropy
    rows = []
    for r, o in zip(recs, outs):
        answers, correct, unparse, called = [], [], 0, 0
        for c in o.outputs:
            called += TOOL_TAG in c.text
            a, ok = _grade(c.text, gold[r["id"]])
            if a is None:
                unparse += 1
                continue
            answers.append(a)
            correct.append(ok)
        n = len(o.outputs)
        se, ncl = semantic_entropy(answers, env[r["id"]]) if answers else (float("nan"), 0)
        rows.append({"id": r["id"], "SE": se, "n_clusters": ncl,
                     "fail_rate": 1 - (sum(correct) / n),            # unparseable counts as wrong
                     "unparse_rate": unparse / n, "call_attempt_rate": called / n,
                     "abstain_rate": float(np.mean([is_abstain(a, env[r["id"]]) for a in answers])) if answers else np.nan,
                     "mean_len": float(np.mean([len(c.token_ids) for c in o.outputs]))})
    return rows


def call_rows(samp, greedy, recs, gold, env):
    from toolmech.labels import call_position, is_abstain
    rows = []
    for r, s, g in zip(recs, samp, greedy):
        called, late, direct_ok, unparse, abst = [], [], [], 0, 0
        for c in s.outputs:
            ca, la = call_position(c.text)
            called.append(ca)
            late.append(la)
            if not ca:
                a, ok = _grade(c.text, gold[r["id"]])
                direct_ok.append(ok)
                unparse += a is None
                abst += bool(a) and is_abstain(a, env[r["id"]])
            else:
                direct_ok.append(False)
        gt = g.outputs[0].text
        gc, gl = call_position(gt)
        ga, gok = _grade(gt, gold[r["id"]]) if not gc else (None, False)
        n = len(s.outputs)
        rows.append({"id": r["id"], "call_rate": float(np.mean(called)),
                     "late_share": float(np.sum(late) / max(1, np.sum(called))),
                     "direct_acc": float(np.mean(direct_ok)), "unparse_rate": unparse / n,
                     "abstain_rate": abst / n, "greedy_call": bool(gc), "greedy_late": bool(gl),
                     "greedy_correct": bool(gok), "greedy_text": gt[:300],
                     "mean_len": float(np.mean([len(c.token_ids) for c in s.outputs]))})
    return rows


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def queue(dirs_order, layers, ks, modes, ks_short=(-4, -2, 2, 4)):
    """Priority order (rev. after timing: ~5-10 min per condition on the A40).
    A: SE^dm at the best layer, all k (manip, then calls in the primary mode).
    B: positive control act^dm and random, best layer, all k.
    C: the other directions at the best layer, k in ks_short.
    D: secondary mode at the best layer for SE^dm, act^dm, random, k in ks_short.
    E: the other two layers for SE^dm, act^dm, random, k in ks_short.
    F: the rest of the full grid (all k everywhere), if time allows.
    k = 0 (unsteered) runs once per (kind, mode)."""
    L1, rest = layers[0], layers[1:]
    prim, sec = modes[0], modes[1:]
    core = [d for d in ("SE^dm", "act^dm", "random") if d in dirs_order]
    others = [d for d in dirs_order if d not in core]
    kshort = [0] + [k for k in ks_short if k in ks]
    q = []

    def block(layer, dirs, kinds_modes, phase, kk):
        for d in dirs:
            for kind, mode in kinds_modes:
                for k in kk:
                    q.append((phase, kind, mode, d, layer, k))

    pm = [("manip", "no_tool"), ("calls", prim)]
    sm = [("calls", m) for m in sec]
    block(L1, core[:1], pm, "A", ks)
    block(L1, core[1:], pm, "B", ks)
    block(L1, others, pm, "C", kshort)
    block(L1, core, sm, "D", kshort)
    for l in rest:
        block(l, core, pm, "E", kshort)
    block(L1, dirs_order, pm + sm, "F", ks)
    for l in rest:
        block(l, dirs_order, pm + sm, "F", ks)
    return q


def run(cfg, phases, per_tier, dirs_order, ks, gpu_util):
    from toolmech.vsteer import SteerableLLM
    S = cfg["steering"]
    m = cfg.get("modes", {"primary": "sparse_tool", "secondary": "current"})
    modes = [m["primary"], m["secondary"]]
    layers = best_layers(cfg, modes[0], S["n_best_layers"])
    ids = task_subset(cfg, per_tier)
    tasks = {str(t["id"]): t for t in load_tasks(cfg, "test")}
    gold = {i: tasks[i]["expected"]["answer"] for i in ids}
    env = {i: task_env(tasks[i]) for i in ids}
    outdir = os.path.join(os.path.dirname(cfg.path("e4", "x", "x")), "steer")
    os.makedirs(outdir, exist_ok=True)
    json.dump({"layers": layers, "ids": ids, "ks": ks, "dirs": dirs_order, "per_tier": per_tier,
               "modes": modes, "hash": cfg.hash},
              open(os.path.join(outdir, f"meta_n{len(ids)}.{cfg.hash}.json"), "w"), indent=1)
    print(f"layers={layers} tasks={len(ids)}", flush=True)

    prompts = {}
    for pm in ["no_tool"] + modes:
        recs = {str(r["id"]): r for r in load_prompts(cfg, "test", pm)}
        prompts[pm] = [dict(recs[i], id=i) for i in ids]

    vecs = {l: steering_vectors(cfg, modes[0], l, dirs_order) for l in layers}
    json.dump({str(l): {n: s for n, (_, s) in v.items()} for l, v in vecs.items()},
              open(os.path.join(outdir, f"scales.{cfg.hash}.json"), "w"), indent=1)
    llm = SteerableLLM(cfg, gpu_memory_utilization=gpu_util)
    samp = cfg["sampling"]
    nt = S.get("k_notool", 10)
    nc = S.get("n_call_samples", 20)

    for phase, kind, mode, d, layer, k in queue(dirs_order, layers, ks, modes):
        if phase not in phases or d not in vecs[layer]:
            continue
        tag = "base" if k == 0 else f"{d}_L{layer}_k{k:+d}"
        path = os.path.join(outdir, f"{kind}_{mode}_n{len(ids)}_{tag}.{cfg.hash}.parquet")
        if os.path.exists(path):
            continue
        v, sd = vecs[layer][d]
        vec = None if k == 0 else (k * sd * v)
        recs = prompts[mode]
        texts = [r["prompt_text"] for r in recs]
        t0 = time.time()
        if kind == "manip":
            outs = llm.generate(texts, layer, vec, n=nt, max_tokens=samp["max_new_tokens"],
                                temperature=samp["temperature"], top_p=samp["top_p"],
                                top_k=samp.get("top_k", -1), seed=cfg["seed"])
            rows = manip_rows(outs, recs, gold, env)
        else:
            st = dict(max_tokens=samp.get("call_max_tokens", 512), stop=["</tool_call>"])
            s = llm.generate(texts, layer, vec, n=nc, temperature=samp["temperature"], top_p=samp["top_p"],
                             top_k=samp.get("top_k", -1), seed=cfg["seed"], **st)
            g = llm.generate(texts, layer, vec, n=1, temperature=0.0, **st)
            rows = call_rows(s, g, recs, gold, env)
        df = pd.DataFrame(rows)
        df["tier"] = [tasks[i]["difficulty"] for i in df.id]
        df["env"] = [env[i] for i in df.id]
        df = df.assign(kind=kind, mode=mode, direction=d if k else "none", layer=layer if k else -1, k=k,
                       alpha=0.0 if k == 0 else k * sd)
        df.to_parquet(path)
        summ = df[[c for c in ("SE", "fail_rate", "call_rate", "greedy_call", "direct_acc", "unparse_rate")
                   if c in df]].mean().round(3).to_dict()
        print(f"[{phase}] {kind:5s} {mode:11s} {d:7s} L{layer} k={k:+d} {time.time() - t0:6.0f}s {summ}",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default="A,B,C,D,E,F")
    ap.add_argument("--per_tier", type=int, default=None)
    ap.add_argument("--dirs", default="SE^dm,act^dm,random,SE,need,diff,act,SE|fail,SE^dm_fm")
    ap.add_argument("--ks", default=None)
    ap.add_argument("--gpu_util", type=float, default=0.55)
    a = ap.parse_args()
    cfg = load_config()
    ks = [int(x) for x in a.ks.split(",")] if a.ks else cfg["steering"]["ks"]
    run(cfg, set(a.phases.split(",")), a.per_tier or cfg["steering"]["manip_tasks_per_tier"],
        a.dirs.split(","), ks, a.gpu_util)


if __name__ == "__main__":
    main()
