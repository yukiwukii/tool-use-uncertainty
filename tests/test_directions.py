"""Synthetic checks for directions.py / geometry.py: known planted directions."""
import numpy as np
import pandas as pd
import pytest
import torch

from toolmech.config import load_config
from toolmech import directions as D
from toolmech import geometry as G

rng = np.random.default_rng(0)


def _unit(v):
    return v / np.linalg.norm(v)


def _confounded(n_per_cell=30, d=64, envs=6):
    """SE depends on env (confound) and, within env, on a true direction u."""
    u = _unit(rng.normal(size=d))
    offs = {e: rng.normal(size=d) * 3 for e in range(envs)}
    rows, X = [], []
    for e in range(envs):
        for t in ("easy", "hard"):
            base_se = e / envs * 2                      # env-level SE differences
            for _ in range(n_per_cell):
                z = rng.normal()
                se = max(0.0, base_se + 0.3 * z)
                X.append(offs[e] + 2.0 * z * u + rng.normal(size=d) * 0.5)
                rows.append({"env": f"E{e}", "tier": t, "SE": se, "fail_rate": 0.2})
    return np.array(X), pd.DataFrame(rows), u, offs


def test_stratified_dm_ignores_env_confound():
    X, lab, u, _ = _confounded()
    plain = D.diff_in_means(X, (lab.SE > lab.SE.median()).astype(int).to_numpy())
    strat, info = D.stratified_dm(X, lab, "SE")
    assert info["cells"] == 12
    assert abs(plain @ u) < 0.5          # plain difference is dominated by env offsets
    assert strat @ u > 0.9               # stratified one recovers the true direction


def test_stratified_dm_binary_and_min_n():
    X, lab, u, _ = _confounded()
    lab["act"] = (lab.SE > lab.groupby(["env", "tier"]).SE.transform("median")).astype(float)
    v, info = D.stratified_dm(X, lab, "act", binary=True)
    assert v @ u > 0.9 and info["cells"] == 12
    lab.loc[lab.env != "E0", "act"] = 0.0   # only E0 cells have both sides
    _, info = D.stratified_dm(X, lab, "act", binary=True)
    assert info["cells"] == 2


def test_ridge_and_residualise():
    X, lab, u, _ = _confounded()
    v, _ = D.ridge_dir(X, lab.SE, alpha=10.0)
    assert v @ u > 0.3                   # ridge mixes in env offsets but has the u component
    lab["x"] = rng.normal(size=len(lab))
    lab["y"] = 3 * lab.x + 1 + rng.normal(size=len(lab)) * 0.01
    r_tr, r_te = D.residualise(lab, lab, "y", "x")
    assert np.abs(np.corrcoef(r_tr, lab.x)[0, 1]) < 0.05
    r_tr, _ = D.residualise(lab, lab, "SE", "tier")
    assert abs(pd.Series(r_tr).groupby(lab.tier).mean()).max() < 1e-9


def test_scoped_metrics_within_cell():
    _, lab, _, _ = _confounded()
    s = pd.Series(lab.SE.to_numpy() + rng.normal(size=len(lab)) * 0.01)
    m = D.scoped_metrics(lab, "SE", s, binary=False)
    assert m["all"] > 0.99 and m["within_cell"] > 0.9 and m["within_cell_n"] == 12


def test_whitened_cos_identity():
    X = rng.normal(size=(200, 16))
    W = G.whitener(X)
    a = rng.normal(size=16)
    assert G.cos(W @ a, W @ a) == pytest.approx(1.0)
    assert np.isnan(G.cos(np.zeros(3), np.ones(3)))


def test_end_to_end_on_synthetic_files(tmp_path):
    """run_mode + geometry_mode + behaviour_tables on fake E1/E2 artefacts."""
    cfg = load_config()
    cfg["results_dir"] = str(tmp_path)
    L, d = 3, 32
    u, v = _unit(rng.normal(size=d)), _unit(rng.normal(size=d))
    rows, ids = [], {"train": [], "test": []}
    states = {"train": [], "test": []}
    k = 0
    for split, n in (("train", 8), ("test", 10)):
        for e in range(4):
            for tier in ("easy", "medium", "hard"):
                for _ in range(n):
                    se, act = abs(rng.normal()), float(rng.random() < 0.4)
                    h = np.stack([rng.normal(size=d) * 0.3 + l * (se * u + act * v) for l in range(L)])
                    tid = f"t{k}"; k += 1
                    ids[split].append(tid); states[split].append(h)
                    rows.append({"id": tid, "split": split, "env": f"E{e}", "tier": tier,
                                 "need": float(se > 0.8), "fail_rate": min(1, se / 2), "SE": se,
                                 "SE_abstain": se, "act_sparse_tool": act, "call_rate_sparse_tool": act,
                                 "abstain_rate_sparse_tool": 0.0})
    pd.DataFrame(rows).to_parquet(cfg.path("e1", "labels", "parquet"))
    for split in ids:
        torch.save({"ids": ids[split], "layers": list(range(L)),
                    "resid": torch.tensor(np.array(states[split]), dtype=torch.float32)},
                   cfg.path("e2", f"resid_{split}_sparse_tool", "pt"))
    dirs, dec, meta = D.run_mode(cfg, "sparse_tool", surface=False)
    assert dirs["SE^dm"][2] @ u > 0.8 and dirs["act^dm"][2] @ v > 0.8
    se_l2 = dec[(dec.label == "SE") & (dec.method == "ridge") & (dec.scope == "all") & (dec.layer == 2)].value.item()
    assert se_l2 > 0.8
    cosdf, layers = G.geometry_mode(cfg, "sparse_tool", n_boot=5, n_null=5)
    assert set(layers) <= {0, 1, 2} and cosdf.boot_lo.notna().any()
    beh = G.behaviour_tables(cfg, "sparse_tool")
    assert len(beh["quadrants"]) == 4


def test_end_to_end_current_mode_uses_tul(tmp_path):
    cfg = load_config()
    cfg["results_dir"] = str(tmp_path)
    L, d = 2, 32
    u, w = _unit(rng.normal(size=d)), _unit(rng.normal(size=d))
    rows, ids, states, tuls = [], {"train": [], "test": []}, {"train": [], "test": []}, {"train": [], "test": []}
    k = 0
    for split, n in (("train", 9), ("test", 9)):
        for e in range(3):
            for tier in ("easy", "medium", "hard"):
                for _ in range(n):
                    se, tul = abs(rng.normal()), rng.normal() * 3 + 5
                    h = np.stack([rng.normal(size=d) * 0.3 + se * u + 0.5 * tul * w for _ in range(L)])
                    tid = f"c{k}"; k += 1
                    ids[split].append(tid); states[split].append(h)
                    tuls[split].append({"id": tid, "tul_1": tul, "tul_8": tul})
                    rows.append({"id": tid, "split": split, "env": f"E{e}", "tier": tier, "need": float(se > 0.8),
                                 "fail_rate": 0.1, "SE": se, "SE_abstain": se, "act_current": 1.0,
                                 "call_rate_current": 1.0, "abstain_rate_current": 0.0})
    pd.DataFrame(rows).to_parquet(cfg.path("e1", "labels", "parquet"))
    for split in ids:
        torch.save({"ids": ids[split], "layers": list(range(L)),
                    "resid": torch.tensor(np.array(states[split]), dtype=torch.float32)},
                   cfg.path("e2", f"resid_{split}_current", "pt"))
        pd.DataFrame(tuls[split]).to_parquet(cfg.path("e2", f"tul_{split}_current", "parquet"))
    dirs, dec, meta = D.run_mode(cfg, "current", surface=False)
    assert dirs["act^dm"][1] @ w > 0.8                     # built from TUL thirds, not the constant act
    assert meta["stratified_info"]["act^dm"]["cells"] == 9
    cosdf, _ = G.geometry_mode(cfg, "current", n_boot=3, n_null=3)
    beh = G.behaviour_tables(cfg, "current")
    assert beh["rho_SE_tul_all"] is not None
