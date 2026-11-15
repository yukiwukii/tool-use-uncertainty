"""M3 / E3: per-layer directions and their decoding curves.

Inputs:  results/e2/resid_<split>_<mode>.<hash>.pt   (last-prompt-token, pre-norm, all layers)
         results/e1/labels.<hash>.parquet
Outputs: results/e3/directions_<mode>.<hash>.pt      {name: [L, d] unit vectors, "meta": ...}
         results/e3/decoding_<mode>.<hash>.parquet   one row per (layer, label, method, scope)

Directions are fit on the train split and evaluated on test. Geometry directions
(PLAN.md M3/M4): difference in means for binary labels, ridge for continuous labels,
and the stratified differences in means d_SE^dm and d_act^dm (rev. Nov 14; d_act^dm in
sparse_tool only). Tool use is measured from replies (act, call_rate), not TUL.
Logistic probes are fit for decoding only.

Usage:  python -m toolmech.directions [--modes sparse_tool,current]
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

from toolmech.config import load_config, load_tasks

TIER_NUM = {"easy": 0, "medium": 1, "hard": 2}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_split(cfg, split, mode, labels):
    """(X [n, L, d] float32 numpy, labels aligned to X rows)."""
    blob = torch.load(cfg.path("e2", f"resid_{split}_{mode}", "pt"), weights_only=False)
    X = blob["resid"].float().numpy()
    lab = labels.set_index("id").loc[[str(i) for i in blob["ids"]]].reset_index()
    return X, lab


def add_derived_labels(lab, mode):
    """Mode-specific act/call columns and tier codes (labels come from replies, not TUL)."""
    lab = lab.copy()
    lab["act"] = lab[f"act_{mode}"]
    lab["call_rate"] = lab[f"call_rate_{mode}"]
    lab["tier_num"] = lab["tier"].map(TIER_NUM)
    lab["diff"] = lab["tier"].map({"easy": 0.0, "hard": 1.0})          # medium -> NaN (held out)
    return lab


def residualise(train, test, y, on):
    """y with a linear fit on `on` (fit on train) removed, for train and test.
    `on` may be a numeric column or 'tier' (one-hot means)."""
    if on == "tier":
        means = train.groupby("tier")[y].mean()
        return train[y] - train["tier"].map(means), test[y] - test["tier"].map(means)
    A = np.c_[np.ones(len(train)), train[on].to_numpy()]
    coef, *_ = np.linalg.lstsq(A, train[y].to_numpy(), rcond=None)
    f = lambda d: d[y].to_numpy() - np.c_[np.ones(len(d)), d[on].to_numpy()] @ coef
    return f(train), f(test)


# ---------------------------------------------------------------------------
# Direction estimators (one layer: X is [n, d])
# ---------------------------------------------------------------------------
def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def diff_in_means(X, y):
    y = np.asarray(y, dtype=float)
    m = ~np.isnan(y)
    if (y[m] == 1).sum() == 0 or (y[m] == 0).sum() == 0:      # one class only: undefined
        return np.full(X.shape[1], np.nan)
    return unit(X[m][y[m] == 1].mean(0) - X[m][y[m] == 0].mean(0))


def ridge_dir(X, y, alpha):
    """Ridge on standardised features, mapped back to raw coordinates (w / sigma)."""
    y = np.asarray(y, dtype=float)
    m = ~np.isnan(y)
    mu, sd = X[m].mean(0), X[m].std(0) + 1e-6
    r = Ridge(alpha=alpha).fit((X[m] - mu) / sd, y[m])
    return unit(r.coef_ / sd), (r, mu, sd)


def logistic_dir(X, y, C):
    y = np.asarray(y, dtype=float)
    m = ~np.isnan(y)
    if np.unique(y[m]).size < 2:                              # one class only: undefined
        return np.full(X.shape[1], np.nan), None
    mu, sd = X[m].mean(0), X[m].std(0) + 1e-6
    r = LogisticRegression(C=C, max_iter=2000).fit((X[m] - mu) / sd, y[m].astype(int))
    return unit(r.coef_[0] / sd), (r, mu, sd)


def stratified_dm(X, lab, score, cells=("env", "tier"), top=1 / 3, min_n=3, keep=None, binary=False):
    """Difference in means computed inside each cell, averaged (weighted by cell size).

    score:  column used to split each cell. binary=True: score==1 vs score==0;
            otherwise top `top` fraction minus bottom `top` fraction of the cell.
    keep:   optional boolean mask of usable rows (e.g. majority-correct tasks).
    Returns (unit vector, info dict with contributing cells and tasks)."""
    use = np.ones(len(lab), bool) if keep is None else np.array(keep, dtype=bool, copy=True)
    use &= lab[score].notna().to_numpy()
    acc, wsum, n_cells, n_tasks = 0.0, 0.0, 0, 0
    for _, idx in lab[use].groupby(list(cells)).groups.items():
        idx = np.asarray(idx)
        s = lab.loc[idx, score].to_numpy(dtype=float)
        if binary:
            hi, lo = idx[s == 1], idx[s == 0]
        else:
            if np.unique(s).size < 2:
                continue
            q_lo, q_hi = np.quantile(s, [top, 1 - top])
            hi, lo = idx[s >= q_hi], idx[s <= q_lo]
            if q_hi == q_lo:  # ties: strict split around the median instead
                med = np.median(s)
                hi, lo = idx[s > med], idx[s <= med]
        if len(hi) < min_n or len(lo) < min_n:
            continue
        w = len(hi) + len(lo)
        acc = acc + w * (X[hi].mean(0) - X[lo].mean(0))
        wsum += w
        n_cells += 1
        n_tasks += w
    if wsum == 0:
        return np.full(X.shape[1], np.nan), {"cells": 0, "tasks": 0}
    return unit(acc / wsum), {"cells": n_cells, "tasks": n_tasks}


def stratified_dm_matched(X, lab, score="SE", match="fail_rate", cells=("env", "tier"), top=1 / 3,
                          tol=0.1, min_n=3):
    """PLAN.md M3 step 2, second option: inside each cell, pair every top-third-`score`
    task with an unused bottom-third task whose `match` differs by at most `tol`
    (greedy, nearest first); difference of the matched means, weighted by pairs.
    Uses all tasks rather than only majority-correct ones."""
    use = lab[score].notna().to_numpy() & lab[match].notna().to_numpy()
    acc, wsum, n_cells, n_tasks = 0.0, 0.0, 0, 0
    for _, idx in lab[use].groupby(list(cells)).groups.items():
        idx = np.asarray(idx)
        s = lab.loc[idx, score].to_numpy(dtype=float)
        f = lab.loc[idx, match].to_numpy(dtype=float)
        if np.unique(s).size < 2:
            continue
        q_lo, q_hi = np.quantile(s, [top, 1 - top])
        hi_m, lo_m = s >= q_hi, s <= q_lo
        if q_hi == q_lo:
            med = np.median(s)
            hi_m, lo_m = s > med, s <= med
        lo_free = list(np.where(lo_m)[0])
        H, L = [], []
        for i in np.where(hi_m)[0][np.argsort(-s[hi_m])]:
            if not lo_free:
                break
            d = np.abs(f[lo_free] - f[i])
            j = int(np.argmin(d))
            if d[j] <= tol and s[lo_free[j]] < s[i]:
                H.append(idx[i]); L.append(idx[lo_free.pop(j)])
        if len(H) < min_n:
            continue
        acc = acc + len(H) * (X[H].mean(0) - X[L].mean(0))
        wsum += len(H)
        n_cells += 1
        n_tasks += 2 * len(H)
    if wsum == 0:
        return np.full(X.shape[1], np.nan), {"cells": 0, "tasks": 0}
    return unit(acc / wsum), {"cells": n_cells, "tasks": n_tasks}


# Geometry directions: name -> fn(X_layer, lab, cfg) -> (vector, info). Shared with geometry.py
# so bootstrap and null refits use exactly the same estimator.
def _ridge(col):
    return lambda X, lab, cfg: (ridge_dir(X, lab[col], cfg["probes"]["ridge_alpha"])[0], {})


def _dm(col):
    return lambda X, lab, cfg: (diff_in_means(X, lab[col].to_numpy()), {})


def _se_dm(X, lab, cfg):
    return stratified_dm(X, lab, "SE", keep=(lab["fail_rate"] < 0.5).to_numpy())


def _se_dm_fm(X, lab, cfg):
    return stratified_dm_matched(X, lab, "SE", "fail_rate")


def _act_dm(X, lab, cfg):
    if lab["mode"].iloc[0] != "sparse_tool":   # rev. Nov 14: in `current` 91% call -> no contrast in cells
        return np.full(X.shape[1], np.nan), {"cells": 0, "tasks": 0, "skipped": "not built outside sparse_tool"}
    return stratified_dm(X, lab, "act", binary=True)


GEOMETRY_SPECS = {
    "SE": _ridge("SE"),
    "SE|fail": _ridge("SE_res_fail"),
    "SE|diff": _ridge("SE_res_diff"),
    "fail": _ridge("fail_rate"),
    "need": _dm("need"),
    "act": _dm("act"),
    "diff": _dm("diff"),
    "SE^dm": _se_dm,
    "SE^dm_fm": _se_dm_fm,      # fail-matched variant (M3 step 2, second option)
    "act^dm": _act_dm,
}


# ---------------------------------------------------------------------------
# Decoding metrics
# ---------------------------------------------------------------------------
def _metric(y, s, binary):
    y, s = np.asarray(y, float), np.asarray(s, float)
    m = ~(np.isnan(y) | np.isnan(s))
    y, s = y[m], s[m]
    if len(y) < 5:
        return np.nan
    if binary:
        return roc_auc_score(y, s) if 0 < y.mean() < 1 else np.nan
    if np.std(y) == 0 or np.std(s) == 0:
        return np.nan
    return spearmanr(y, s).statistic


def scoped_metrics(lab, y_col, s, binary):
    """Metric over all test tasks, mean within tier, mean within env x tier cell."""
    out = {"all": _metric(lab[y_col], s, binary)}
    for scope, keys in (("within_tier", ["tier"]), ("within_cell", ["env", "tier"])):
        vals = [_metric(g[y_col], s[g.index], binary) for _, g in lab.groupby(keys)]
        vals = [v for v in vals if not np.isnan(v)]
        out[scope] = float(np.mean(vals)) if vals else np.nan
        out[scope + "_n"] = len(vals)
    return out


def surface_features(cfg, lab):
    """Prompt-only baseline features: token count, digit count, max operand size, env one-hot."""
    from toolmech.prompts import load_prompts
    ntok = {}
    for split in ("train", "test"):
        for r in load_prompts(cfg, split, "current"):
            ntok[str(r["id"])] = r["n_tokens"]
    instr = {}
    for split in ("train", "test"):
        for t in load_tasks(cfg, split):
            instr[str(t["id"])] = t["instruction"]
    feats = pd.DataFrame({"id": lab["id"]})
    feats["n_tokens"] = feats["id"].map(ntok)
    feats["n_digits"] = feats["id"].map(lambda i: sum(c.isdigit() for c in instr[i]))
    feats["log_max_num"] = feats["id"].map(
        lambda i: np.log10(1 + max([float(x) for x in re.findall(r"\d+", instr[i])[:50]] or [0])))
    env = pd.get_dummies(lab["env"], prefix="env").astype(float)
    return pd.concat([feats.drop(columns="id"), env.reset_index(drop=True)], axis=1).to_numpy(float)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
BINARY = {"need": True, "act": True, "diff": True}
CONTINUOUS = ("SE", "fail_rate", "call_rate", "SE_res_fail", "SE_res_diff", "SE_abstain")


def run_mode(cfg, mode, surface=True):
    labels = pd.read_parquet(cfg.path("e1", "labels", "parquet"))
    Xtr, ltr = load_split(cfg, "train", mode, labels)
    Xte, lte = load_split(cfg, "test", mode, labels)
    ltr = add_derived_labels(ltr, mode)
    lte = add_derived_labels(lte, mode)
    ltr["mode"] = lte["mode"] = mode
    for name, on in (("SE_res_fail", "fail_rate"), ("SE_res_diff", "tier")):
        ltr[name], lte[name] = residualise(ltr, lte, "SE", on)
    L = Xtr.shape[1]
    P = cfg["probes"]

    dirs = {k: np.zeros((L, Xtr.shape[2]), np.float32) for k in GEOMETRY_SPECS}
    dirs.update({f"logit_{k}": np.zeros((L, Xtr.shape[2]), np.float32) for k in BINARY})
    info, rows = {}, []
    for l in range(L):
        Xl, Xt = Xtr[:, l], Xte[:, l]
        for name, fn in GEOMETRY_SPECS.items():
            v, inf = fn(Xl, ltr, cfg)
            dirs[name][l] = v
            if inf:
                info.setdefault(name, []).append(inf)
        # decoding: logistic for binary labels, ridge for continuous, plus dm projections
        for y, binary in BINARY.items():
            v, fit = logistic_dir(Xl, ltr[y], P["logistic_C"])
            dirs[f"logit_{y}"][l] = v
            if fit is not None:
                r, mu, sd = fit
                s = pd.Series(r.decision_function((Xt - mu) / sd), index=lte.index)
                rows += _rows(mode, l, y, "logistic", scoped_metrics(lte, y, s, True))
            s = pd.Series(Xt @ dirs[y][l], index=lte.index)
            rows += _rows(mode, l, y, "dm_projection", scoped_metrics(lte, y, s, True))
        for y in CONTINUOUS:
            _, (r, mu, sd) = ridge_dir(Xl, ltr[y], P["ridge_alpha"])
            s = pd.Series(r.predict((Xt - mu) / sd), index=lte.index)
            rows += _rows(mode, l, y, "ridge", scoped_metrics(lte, y, s, False))
        for name, y in (("SE^dm", "SE"), ("SE^dm_fm", "SE"), ("act^dm", "act")):
            s = pd.Series(Xt @ dirs[name][l], index=lte.index)
            rows += _rows(mode, l, y, f"{name}_projection", scoped_metrics(lte, y, s, y == "act"))
        # norm channel (Stolfo et al.): does ||h|| alone predict SE / need?
        s = pd.Series(np.linalg.norm(Xt, axis=1), index=lte.index)
        rows += _rows(mode, l, "SE", "norm_only", scoped_metrics(lte, "SE", s, False))
        rows += _rows(mode, l, "need", "norm_only", scoped_metrics(lte, "need", s, True))
        print(f"[{mode}] layer {l + 1}/{L} done", flush=True)

    # surface-feature baselines (layer = -1)
    Ftr, Fte = (surface_features(cfg, ltr), surface_features(cfg, lte)) if surface else (None, None)
    for y, binary in (list(BINARY.items()) + [(c, False) for c in CONTINUOUS]) if surface else []:
        if binary:
            _, fit = logistic_dir(Ftr, ltr[y], 1.0)
            if fit is None:
                continue
            r, mu, sd = fit
            s = r.decision_function((Fte - mu) / sd)
        else:
            _, (r, mu, sd) = ridge_dir(Ftr, ltr[y], 1.0)
            s = r.predict((Fte - mu) / sd)
        rows += _rows(mode, -1, y, "surface_features", scoped_metrics(lte, y, pd.Series(s, index=lte.index), binary))

    meta = {"mode": mode, "n_train": len(ltr), "n_test": len(lte), "layers": L,
            "stratified_info": {k: v[0] for k, v in info.items()}}  # same cells at every layer
    torch.save({**{k: torch.from_numpy(v) for k, v in dirs.items()}, "meta": meta},
               cfg.path("e3", f"directions_{mode}", "pt"))
    dec = pd.DataFrame(rows)
    dec.to_parquet(cfg.path("e3", f"decoding_{mode}", "parquet"))
    return dirs, dec, meta


def _rows(mode, layer, label, method, m):
    return [{"mode": mode, "layer": layer, "label": label, "method": method, "scope": sc,
             "value": m[sc], "n_groups": m.get(sc + "_n", np.nan)}
            for sc in ("all", "within_tier", "within_cell")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default=None)
    a = ap.parse_args()
    cfg = load_config()
    m = cfg.get("modes", {"primary": "sparse_tool", "secondary": "current"})
    for mode in (a.modes.split(",") if a.modes else [m["primary"], m["secondary"]]):
        _, dec, meta = run_mode(cfg, mode)
        best = (dec[(dec.label == "SE") & (dec.method == "ridge") & (dec.scope == "all")]
                .sort_values("value", ascending=False).head(3))
        print(f"[{mode}] best SE layers: {best[['layer', 'value']].to_dict('records')}")
        print(f"[{mode}] stratified cells: {meta['stratified_info']}")


if __name__ == "__main__":
    main()
