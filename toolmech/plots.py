"""Summaries and figures for E3-E5 (REPORT.md figures 1-6).

Usage:  python -m toolmech.plots [--what e3,steer,patch,e5]
Writes CSV/JSON summaries and PNGs to results/figures/.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from toolmech.config import ROOT, load_config

FIG = os.path.join(ROOT, "results", "figures")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ---------------------------------------------------------------------------
# E3
# ---------------------------------------------------------------------------
def e3(cfg):
    plt = _plt()
    for mode in ("sparse_tool", "current"):
        p = cfg.path("e3", f"decoding_{mode}", "parquet")
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        for ax, sc in zip(axes, ("all", "within_tier", "within_cell")):
            for (lab, meth) in (("SE", "ridge"), ("fail_rate", "ridge"), ("call_rate", "ridge"),
                                ("need", "logistic"), ("act", "logistic"), ("diff", "logistic")):
                s = d[(d.label == lab) & (d.method == meth) & (d.scope == sc) & (d.layer >= 0)].sort_values("layer")
                if len(s) and s.value.notna().any():
                    ax.plot(s.layer, s.value, label=f"{lab} ({'AUROC' if meth == 'logistic' else 'Spearman'})")
                b = d[(d.label == lab) & (d.method == "surface_features") & (d.scope == sc)]
                if len(b) and b.value.notna().any() and lab in ("SE", "need"):
                    ax.axhline(float(b.value.iloc[0]), ls=":", lw=1, color="gray")
            ax.set_title(f"{mode}: {sc}")
            ax.set_xlabel("layer")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"e3_decoding_{mode}.png"), dpi=120)
        plt.close(fig)
        cp = cfg.path("e3", f"cosines_{mode}", "parquet")
        if os.path.exists(cp):
            c = pd.read_parquet(cp)
            c = c[c.boot_lo.notna()]
            c.to_csv(os.path.join(FIG, f"e3_cosines_{mode}.csv"), index=False)


# ---------------------------------------------------------------------------
# E4 steering
# ---------------------------------------------------------------------------
def load_steer(cfg, n=300):
    d = os.path.join(ROOT, "results", "e4", "steer")
    fs = glob.glob(os.path.join(d, f"*_n{n}_*.{cfg.hash}.parquet"))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def steer(cfg, n=300):
    df = load_steer(cfg, n)
    if df.empty:
        return
    man = df[df.kind == "manip"]
    cal = df[df.kind == "calls"]
    base_m = man[man.k == 0][["id", "SE", "fail_rate", "unparse_rate"]].set_index("id")
    rows = []
    for (d, l), g in man[man.k != 0].groupby(["direction", "layer"]):
        for k, gk in g.groupby("k"):
            b = base_m.loc[gk.id]
            rows.append({"direction": d, "layer": l, "k": k, "SE": gk.SE.mean(), "dSE": (gk.SE.values - b.SE.values).mean(),
                         "fail_rate": gk.fail_rate.mean(), "unparse": gk.unparse_rate.mean(),
                         "d_unparse": gk.unparse_rate.mean() - b.unparse_rate.mean()})
    M = pd.DataFrame(rows)
    out = {}
    for mode, gm in cal.groupby("mode"):
        base_c = gm[gm.k == 0].set_index("id")
        r2 = []
        for (d, l), g in gm[gm.k != 0].groupby(["direction", "layer"]):
            for k, gk in g.groupby("k"):
                b = base_c.loc[gk.id]
                rec = {"mode": mode, "direction": d, "layer": l, "k": k,
                       "call_rate": gk.call_rate.mean(), "d_call": (gk.call_rate.values - b.call_rate.values).mean(),
                       "greedy_call": gk.greedy_call.mean(), "direct_acc": gk.direct_acc.mean(),
                       "unparse_calls": gk.unparse_rate.mean()}
                for t in ("easy", "medium", "hard"):
                    m = (gk.tier == t).to_numpy()
                    rec[f"d_call_{t}"] = (gk.call_rate.values[m] - b.call_rate.values[m]).mean()
                    rec[f"direct_acc_{t}"] = gk.direct_acc.values[m].mean()
                r2.append(rec)
        C = pd.DataFrame(r2)
        if not M.empty:
            C = C.merge(M, on=["direction", "layer", "k"], how="left")
        # drop strengths whose no-tool answers become unparseable (>10 points above baseline)
        C["valid"] = C.d_unparse.fillna(0) <= 0.10
        C.to_csv(os.path.join(FIG, f"e4_steer_{mode}.csv"), index=False)
        # slope of d_call vs measured dSE (through the origin) per direction/layer, valid strengths only
        sl = []
        for (d, l), g in C[C.valid & C.dSE.notna()].groupby(["direction", "layer"]):
            x, y = g.dSE.to_numpy(), g.d_call.to_numpy()
            sl.append({"direction": d, "layer": l, "slope_dcall_per_dSE": float((x * y).sum() / max((x * x).sum(), 1e-9)),
                       "max_abs_dSE": float(np.abs(x).max()), "max_abs_dcall": float(np.abs(y).max()), "n_k": len(g)})
        out[mode] = {"base_call_rate": float(base_c.call_rate.mean()), "slopes": sl}
        _steer_fig(C, mode)
    if not M.empty:
        M.to_csv(os.path.join(FIG, "e4_manip.csv"), index=False)
        out["base_SE"] = float(base_m.SE.mean())
        out["base_unparse"] = float(base_m.unparse_rate.mean())
    json.dump(out, open(os.path.join(FIG, "e4_steer_summary.json"), "w"), indent=1, default=float)
    return out


def _steer_fig(C, mode):
    plt = _plt()
    for l, gl in C.groupby("layer"):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for d, g in gl.groupby("direction"):
            g = g.sort_values("k")
            if g.SE.notna().any():
                axes[0].plot(g.k, g.dSE, marker="o", label=d)
            axes[1].plot(g.k, g.d_call, marker="o", label=d)
            v = g[g.valid & g.dSE.notna()]
            if len(v):
                axes[2].scatter(v.dSE, v.d_call, label=d, s=15)
        axes[0].set(title=f"manipulation check (L{l})", xlabel="k", ylabel="ΔSE vs unsteered")
        axes[1].set(title=f"call rate, {mode} (L{l})", xlabel="k", ylabel="Δ call rate")
        axes[2].set(title="Δ call rate vs measured ΔSE", xlabel="ΔSE", ylabel="Δ call rate")
        for a in axes:
            a.axhline(0, color="gray", lw=0.5)
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"e4_steer_{mode}_L{l}.png"), dpi=120)
        plt.close(fig)


# ---------------------------------------------------------------------------
# E4 patching
# ---------------------------------------------------------------------------
def boot_ci(x, n=1000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def patch(cfg, mode="sparse_tool", n=300):
    d = os.path.join(ROOT, "results", "e4", "patch")
    out = {}
    for stage in ("sweep", "dims", "pos"):
        p = os.path.join(d, f"{stage}_{mode}_n{n}.{cfg.hash}.parquet")
        if not os.path.exists(p):
            continue
        df = pd.read_parquet(p)
        rows = []
        for (l, pt, di), g in df.groupby(["layer", "patch", "direction"]):
            per_pair = g.groupby("pair").effect.mean()
            lo, hi = boot_ci(per_pair)
            rows.append({"layer": l, "patch": pt, "direction": di, "effect": per_pair.mean(), "ci_lo": lo, "ci_hi": hi,
                         "norm_effect_median": g.norm_effect.median(), "n": len(per_pair)})
        T = pd.DataFrame(rows)
        T.to_csv(os.path.join(FIG, f"e4_patch_{stage}_{mode}.csv"), index=False)
        out[stage] = T
    if "sweep" in out:
        plt = _plt()
        T = out["sweep"]
        fig, ax = plt.subplots(figsize=(8, 4))
        for di, g in T.groupby("direction"):
            g = g.sort_values("layer")
            ax.plot(g.layer, g.effect, marker=".", label=di)
            ax.fill_between(g.layer, g.ci_lo, g.ci_hi, alpha=0.2)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set(xlabel="layer", ylabel="ΔTUL (patched − recipient), nats", title=f"suffix patching, {mode}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"e4_patch_sweep_{mode}.png"), dpi=120)
        plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="e3,steer,patch")
    a = ap.parse_args()
    cfg = load_config()
    os.makedirs(FIG, exist_ok=True)
    w = a.what.split(",")
    if "e3" in w:
        e3(cfg)
    if "steer" in w:
        print(json.dumps(steer(cfg), indent=1, default=float)[:3000])
    if "patch" in w:
        for k, v in (patch(cfg) or {}).items():
            print(k)
            print(v.round(3).to_string(index=False)[:3000])


if __name__ == "__main__":
    main()
