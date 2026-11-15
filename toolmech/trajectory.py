"""E5 steps 3-5: uncertainty at P1 / P2 / P3 along tool-use conversations (PLAN.md M6).

Positions (from When2Tool's per-round prompt texts, never fixed offsets):
  P1      last token of the first prompt (before the first assistant turn)
  P2_h    last token of the h-th tool call (`</tool_call>`), before its result
  P3_h    last token of the prompt after the h-th tool result (next assistant turn)

Stages:
  resid   HF fp32 pass: pre-norm residual at every position, all layers (stored fp16)
  sepost  vLLM: k_post answers sampled at T=1 from every P3 prompt -> SE_post,
          accuracy, share of samples that call again
  analyze probe readouts: the P1 ridge SE probe (fit on E2 train states, primary mode)
          applied at every position, and a P3-specific ridge probe fit on train
          conversations' P3 states against SE_post and tested on test conversations;
          contrasts by need x condition (real / corrupt) and per hop.

Outputs in results/e5/: positions_<tag>.parquet, resid_<tag>.pt, sepost_<tag>.parquet,
trajectory_summary.json

Usage:  python -m toolmech.trajectory --stage resid|sepost|analyze
"""
import argparse
import glob
import json
import os
import re
import time

import numpy as np
import pandas as pd
import torch

from toolmech.config import load_config

CALL_END = "</tool_call>"


def conv_files(cfg):
    d = os.path.dirname(cfg.path("e5", "x", "x"))
    return sorted(glob.glob(os.path.join(d, f"conv_*.{cfg.hash}.jsonl")))


def tag_of(path):
    return os.path.basename(path).split(".")[0][len("conv_"):]


def _rejected(tr):
    res = tr.get("tool_result") or {}
    return isinstance(res, dict) and "rejected" in str(res.get("message", "")).lower()


def positions(item):
    """[(pos, hop, text)] for one conversation."""
    tr = item.get("trace") or []
    if not tr:
        return []
    out = [("P1", 0, tr[0]["prompt_text"])]
    hop = 0
    for r, t in enumerate(tr):
        if t.get("type") != "tool" or t.get("tool_result") is None or _rejected(t):
            continue
        hop += 1
        raw = t.get("raw") or ""
        j = raw.find(CALL_END)
        if j >= 0:
            out.append(("P2", hop, t["prompt_text"] + raw[: j + len(CALL_END)]))
        if r + 1 < len(tr):
            out.append(("P3", hop, tr[r + 1]["prompt_text"]))
    return out


# ---------------------------------------------------------------------------
# Stage resid
# ---------------------------------------------------------------------------
@torch.no_grad()
def resid_stage(cfg, token_budget=12000):
    from toolmech import hf
    from toolmech.conversations import load
    tok, model = hf.load_hf(cfg)
    L = hf.n_layers(model)
    for f in conv_files(cfg):
        tag = tag_of(f)
        rp = cfg.path("e5", f"resid_{tag}", "pt")
        if os.path.exists(rp):
            continue
        rows, texts = [], []
        for it in load(f):
            for pos, hop, text in positions(it):
                rows.append({"id": it["id"], "pos": pos, "hop": hop, "env": it["env"], "tier": it["tier"],
                             "mode": it["mode"], "cond": it["cond"], "multi_step": it["multi_step"],
                             "correct": it["correct"], "tool_calls": it["tool_calls"]})
                texts.append(text)
        lens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
        order = np.argsort(lens)
        R = torch.zeros(len(texts), L, model.config.hidden_size, dtype=torch.float16)
        t0, b = time.time(), 0
        while b < len(order):
            n = 1                    # ascending lengths: the last row is the longest
            while b + n < len(order) and n < 32 and lens[order[b + n]] * (n + 1) <= token_budget:
                n += 1
            idx = order[b:b + n]
            ids, mask, _ = hf.encode(tok, [texts[i] for i in idx])
            gather = torch.full((ids.shape[0], 1), ids.shape[1] - 1, device="cuda")
            with hf.capture(model, gather) as store:
                model.model(input_ids=ids, attention_mask=mask)
            R[torch.as_tensor(idx)] = torch.stack([store[l][:, 0] for l in range(L)], 1).half()
            b += n
        df = pd.DataFrame(rows)
        df["n_tokens"] = lens
        df.to_parquet(cfg.path("e5", f"positions_{tag}", "parquet"))
        torch.save({"resid": R, "layers": list(range(L))}, rp)
        print(f"[resid] {tag}: {len(texts)} positions in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# Stage sepost
# ---------------------------------------------------------------------------
def sepost_stage(cfg, gpu_util=0.85):
    from vllm import LLM, SamplingParams
    from utils import clean, compare_values, extract_boxed  # When2Tool
    from toolmech.conversations import load
    from toolmech.labels import is_abstain, semantic_entropy
    k = cfg["conversations"]["k_post"]
    s = cfg["sampling"]
    llm = None
    for f in conv_files(cfg):
        tag = tag_of(f)
        sp = cfg.path("e5", f"sepost_{tag}", "parquet")
        if os.path.exists(sp):
            continue
        if llm is None:
            llm = LLM(model=cfg["model_id"], dtype="bfloat16", max_model_len=16384,
                      gpu_memory_utilization=gpu_util, seed=cfg["seed"])
        meta, texts = [], []
        for it in load(f):
            for pos, hop, text in positions(it):
                if pos == "P3":
                    meta.append((it["id"], hop, it["gold"], it["env"]))
                    texts.append(text)
        t0 = time.time()
        outs = llm.generate(texts, SamplingParams(n=k, temperature=s["temperature"], top_p=s["top_p"],
                                                  top_k=s.get("top_k", -1), max_tokens=s["max_new_tokens"],
                                                  seed=cfg["seed"]), use_tqdm=False)
        rows = []
        for (tid, hop, gold, env), o in zip(meta, outs):
            ans, ok, calls = [], [], 0
            for c in o.outputs:
                if "<tool_call>" in c.text:
                    calls += 1
                    continue
                b = extract_boxed(c.text)
                if b:
                    a = clean(b)
                    ans.append(a)
                    ok.append(bool(compare_values(a, clean(gold))))
            se, ncl = semantic_entropy(ans, env) if ans else (np.nan, 0)
            rows.append({"id": tid, "hop": hop, "SE_post": se, "n_clusters": ncl, "n_answers": len(ans),
                         "acc_post": float(np.mean(ok)) if ok else np.nan, "call_again_rate": calls / k,
                         "abstain_post": float(np.mean([is_abstain(a, env) for a in ans])) if ans else np.nan})
        pd.DataFrame(rows).to_parquet(sp)
        print(f"[sepost] {tag}: {len(texts)} P3 prompts in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# Stage analyze
# ---------------------------------------------------------------------------
def _ridge_fit(X, y, alpha):
    from sklearn.linear_model import Ridge
    m = ~np.isnan(y)
    mu, sd = X[m].mean(0), X[m].std(0) + 1e-6
    r = Ridge(alpha=alpha).fit((X[m] - mu) / sd, y[m])
    return lambda Z: r.predict((Z - mu) / sd)


def analyze(cfg, layer=None):
    from scipy.stats import spearmanr
    from toolmech.config import Config  # noqa: F401
    m = cfg.get("modes", {"primary": "sparse_tool"})["primary"]
    if layer is None:
        dec = pd.read_parquet(cfg.path("e3", f"decoding_{m}", "parquet"))
        d = dec[(dec.label == "SE") & (dec.method == "ridge") & (dec.scope == "all") & (dec.layer >= 0)]
        layer = int(d.sort_values("value", ascending=False).layer.iloc[0])
    alpha = cfg["probes"]["ridge_alpha"]
    labels = pd.read_parquet(cfg.path("e1", "labels", "parquet")).set_index("id")
    blob = torch.load(cfg.path("e2", f"resid_train_{m}", "pt"), weights_only=False)
    p1 = _ridge_fit(blob["resid"][:, layer].float().numpy(),
                    labels.loc[[str(i) for i in blob["ids"]], "SE"].to_numpy(float), alpha)

    frames = []
    for f in conv_files(cfg):
        tag = tag_of(f)
        pp, rp, sp = (cfg.path("e5", f"{n}_{tag}", e) for n, e in
                      (("positions", "parquet"), ("resid", "pt"), ("sepost", "parquet")))
        if not (os.path.exists(pp) and os.path.exists(rp)):
            continue
        pos = pd.read_parquet(pp)
        X = torch.load(rp, weights_only=False)["resid"][:, layer].float().numpy()
        pos["probe_P1"] = p1(X)
        pos["split"] = "train" if re.match(r"(multihop_)?train_", tag) else "test"
        pos["_row"] = np.arange(len(pos))
        pos["_tag"] = tag
        if os.path.exists(sp):
            pos = pos.merge(pd.read_parquet(sp), on=["id", "hop"], how="left")
            pos.loc[pos.pos != "P3", ["SE_post", "acc_post", "call_again_rate"]] = np.nan
        frames.append((pos, X))
    if not frames:
        print("nothing to analyze")
        return
    df = pd.concat([f for f, _ in frames], ignore_index=True)
    Xall = np.concatenate([x[f["_row"].to_numpy()] for f, x in frames])
    df["need"] = labels.reindex(df.id)["need"].to_numpy()
    df["SE_P1"] = labels.reindex(df.id)["SE"].to_numpy()

    # P3-specific probe: fit on train P3 states vs SE_post, test on test P3
    out = {"layer": layer}
    if "SE_post" in df:
        tr = (df.pos == "P3") & (df.split == "train") & df.SE_post.notna()
        te = (df.pos == "P3") & (df.split == "test") & df.SE_post.notna()
        if tr.sum() > 50 and te.sum() > 20:
            p3 = _ridge_fit(Xall[tr.to_numpy()], df.loc[tr, "SE_post"].to_numpy(float), alpha)
            df["probe_P3"] = p3(Xall)
            out["P3_probe_rho_test"] = float(spearmanr(df.loc[te, "probe_P3"], df.loc[te, "SE_post"])[0])
            out["P1_probe_at_P3_rho_test"] = float(spearmanr(df.loc[te, "probe_P1"], df.loc[te, "SE_post"])[0])
            out["n_P3_train"], out["n_P3_test"] = int(tr.sum()), int(te.sum())
    test = df[df.split == "test"]
    keys = ["mode", "multi_step", "cond", "need", "pos", "hop"]
    cols = [c for c in ("probe_P1", "probe_P3", "SE_post", "acc_post", "call_again_rate", "SE_P1") if c in test]
    tab = test.groupby(keys)[cols].mean().round(3).reset_index()
    tab["n"] = test.groupby(keys).size().to_numpy()
    tab.to_csv(cfg.path("e5", "trajectory_table", "csv"), index=False)
    # Delta SE = SE_post(first P3) - SE(P1), by need x condition
    first = test[(test.pos == "P3") & (test.hop == 1)].copy()
    first["dSE"] = first.SE_post - first.SE_P1
    out["dSE"] = first.groupby(["mode", "multi_step", "cond", "need"]).agg(
        dSE=("dSE", "mean"), SE_post=("SE_post", "mean"), SE_P1=("SE_P1", "mean"),
        acc_post=("acc_post", "mean"), n=("dSE", "size")).round(3).reset_index().to_dict("records")
    df.drop(columns=["_row"]).to_parquet(cfg.path("e5", "trajectory_rows", "parquet"))
    json.dump(out, open(cfg.path("e5", "trajectory_summary", "json"), "w"), indent=1, default=str)
    print(json.dumps(out, indent=1, default=str)[:4000])
    print(tab.to_string(index=False)[:6000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["resid", "sepost", "analyze"])
    ap.add_argument("--gpu_util", type=float, default=0.85)
    ap.add_argument("--layer", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config()
    if a.stage == "resid":
        resid_stage(cfg)
    elif a.stage == "sepost":
        sepost_stage(cfg, a.gpu_util)
    else:
        analyze(cfg, a.layer)


if __name__ == "__main__":
    main()
