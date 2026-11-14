"""M4 / E3: geometry of the directions, and the behaviour tables of the E3 details.

Inputs:  results/e3/directions_<mode>.<hash>.pt, results/e3/decoding_<mode>.<hash>.parquet,
         results/e2/resid_train_<mode>.<hash>.pt (bootstrap / null refits, whitening),
         results/e1/labels.<hash>.parquet
Outputs: results/e3/cosines_<mode>.<hash>.parquet   every layer: raw cosine for each pair;
                                                     best layers: + bootstrap CI, null 95%, whitened
         results/e3/behaviour_<mode>.<hash>.json     quadrant table, within-tier / within-cell
                                                     Spearman of SE with the call rate

Usage:  python -m toolmech.geometry [--modes sparse_tool,current] [--n_boot 100] [--n_null 100]
"""
import argparse
import itertools
import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf

from toolmech.config import load_config
from toolmech.directions import (GEOMETRY_SPECS, add_derived_labels, load_split, residualise)


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0 or np.isnan(na) or np.isnan(nb):
        return np.nan
    return float(a @ b / (na * nb))


def whitener(X):
    """C^(-1/2) from a Ledoit-Wolf covariance (n < d, so the sample covariance is singular)."""
    C = LedoitWolf().fit(X).covariance_
    w, V = np.linalg.eigh(C)
    return (V / np.sqrt(np.clip(w, 1e-8, None))) @ V.T


def best_layers(dec, k=3):
    s = dec[(dec.label == "SE") & (dec.method == "ridge") & (dec.scope == "all") & (dec.layer >= 0)]
    return s.sort_values("value", ascending=False).head(k)["layer"].astype(int).tolist()


def refit_all(Xl, lab, cfg, names):
    return {n: GEOMETRY_SPECS[n](Xl, lab, cfg)[0] for n in names}


def geometry_mode(cfg, mode, n_boot, n_null, seed=0):
    rng = np.random.default_rng(seed)
    D = torch.load(cfg.path("e3", f"directions_{mode}", "pt"), weights_only=False)
    names = list(GEOMETRY_SPECS)
    L = D[names[0]].shape[0]
    rows = []
    for l in range(L):
        for a, b in itertools.combinations(names, 2):
            rows.append({"mode": mode, "layer": l, "a": a, "b": b,
                         "cos": cos(D[a][l].numpy(), D[b][l].numpy())})
    cosdf = pd.DataFrame(rows)

    dec = pd.read_parquet(cfg.path("e3", f"decoding_{mode}", "parquet"))
    layers = best_layers(dec)
    labels = pd.read_parquet(cfg.path("e1", "labels", "parquet"))
    Xtr, ltr = load_split(cfg, "train", mode, labels)
    ltr = add_derived_labels(ltr, mode)
    ltr["mode"] = mode
    ltr["SE_res_fail"], _ = residualise(ltr, ltr, "SE", "fail_rate")
    ltr["SE_res_diff"], _ = residualise(ltr, ltr, "SE", "tier")
    label_cols = ["SE", "SE_res_fail", "SE_res_diff", "fail_rate", "need", "act", "diff"]

    extra = []
    for l in layers:
        Xl = Xtr[:, l]
        W = whitener(Xl)
        base = {n: D[n][l].numpy() for n in names}
        boots = []
        for _ in range(n_boot):                       # stability: resample tasks, refit
            i = rng.integers(0, len(ltr), len(ltr))
            boots.append(refit_all(Xl[i], ltr.iloc[i].reset_index(drop=True), cfg, names))
        nulls = []
        for _ in range(n_null):                       # chance: shuffle labels jointly, refit
            p = rng.permutation(len(ltr))
            sh = ltr.copy()
            for c in label_cols:
                if c in sh:
                    sh[c] = ltr[c].to_numpy()[p]
            nulls.append(refit_all(Xl, sh, cfg, names))
        for a, b in itertools.combinations(names, 2):
            bc = np.array([cos(x[a], x[b]) for x in boots])
            nc = np.abs([cos(x[a], x[b]) for x in nulls])
            extra.append({"mode": mode, "layer": l, "a": a, "b": b,
                          "boot_lo": np.nanpercentile(bc, 2.5), "boot_hi": np.nanpercentile(bc, 97.5),
                          "null_abs95": np.nanpercentile(nc, 95),
                          "cos_whitened": cos(W @ base[a], W @ base[b])})
        print(f"[{mode}] geometry layer {l}: {n_boot} boots, {n_null} nulls", flush=True)
    cosdf = cosdf.merge(pd.DataFrame(extra), on=["mode", "layer", "a", "b"], how="left")
    cosdf.to_parquet(cfg.path("e3", f"cosines_{mode}", "parquet"))
    return cosdf, layers


def behaviour_tables(cfg, mode):
    """E3 details 2-3: within-tier/cell Spearman of SE with the call rate, and the quadrants."""
    lab = pd.read_parquet(cfg.path("e1", "labels", "parquet"))
    t = lab[lab.split == "test"].copy()
    t["call"] = t[f"call_rate_{mode}"]
    t["abstain"] = t[f"abstain_rate_{mode}"]

    def rho(g, x, y):
        if g[x].std() == 0 or g[y].std() == 0 or g[[x, y]].isna().any(axis=None):
            return np.nan
        return float(spearmanr(g[x], g[y]).statistic)

    out = {"mode": mode}
    for y in ("call",):
        out[f"rho_SE_{y}_all"] = rho(t, "SE", y)
        out[f"rho_SE_{y}_by_tier"] = {k: rho(g, "SE", y) for k, g in t.groupby("tier")}
        cells = [rho(g, "SE", y) for _, g in t.groupby(["env", "tier"])]
        cells = [c for c in cells if not np.isnan(c)]
        out[f"rho_SE_{y}_within_cell"] = {"median": float(np.median(cells)) if cells else None,
                                          "share_positive": float(np.mean(np.array(cells) > 0)) if cells else None,
                                          "n_cells": len(cells)}
        out[f"rho_fail_{y}_all"] = rho(t, "fail_rate", y)
    q = []
    for tier in ("easy", "hard"):
        g = t[t.tier == tier]
        med = g["SE"].median()
        for name, part in (("low_SE", g[g.SE <= med]), ("high_SE", g[g.SE > med])):
            q.append({"tier": tier, "SE_half": name, "n": int(len(part)),
                      "call_rate": float(part["call"].mean()), "abstain_rate": float(part["abstain"].mean()),
                      "fail_rate": float(part["fail_rate"].mean()), "need": float(part["need"].mean()),
                      "mean_SE": float(part["SE"].mean())})
    out["quadrants"] = q
    with open(cfg.path("e3", f"behaviour_{mode}", "json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default=None)
    ap.add_argument("--n_boot", type=int, default=None)
    ap.add_argument("--n_null", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config()
    m = cfg.get("modes", {"primary": "sparse_tool", "secondary": "current"})
    nb = a.n_boot or cfg["probes"]["n_bootstrap"]
    nn = a.n_null or cfg["probes"]["n_null"]
    for mode in (a.modes.split(",") if a.modes else [m["primary"], m["secondary"]]):
        cosdf, layers = geometry_mode(cfg, mode, nb, nn)
        beh = behaviour_tables(cfg, mode)
        top = cosdf[cosdf.layer == layers[0]].dropna(subset=["boot_lo"])
        print(f"[{mode}] best layers {layers}; key cosines at layer {layers[0]}:")
        for a_, b_ in (("SE^dm", "act^dm"), ("SE", "act"), ("SE", "need"), ("SE", "fail"), ("act^dm", "diff"),
                       ("SE|diff", "act"), ("SE|fail", "need")):
            r = top[((top.a == a_) & (top.b == b_)) | ((top.a == b_) & (top.b == a_))]
            if len(r):
                r = r.iloc[0]
                print(f"  {a_:7s} vs {b_:7s} cos {r.cos:+.2f} [{r.boot_lo:+.2f}, {r.boot_hi:+.2f}] "
                      f"null95 {r.null_abs95:.2f} whitened {r.cos_whitened:+.2f}")
        print(f"[{mode}] quadrants: {beh['quadrants']}")


if __name__ == "__main__":
    main()
