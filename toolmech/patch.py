"""M5b / E4: activation patching between matched high- and low-SE tasks.

Every prompt in a mode ends with the same response-policy suffix and assistant
header, so the residual at those positions can be swapped between tasks without
aligning lengths (the shared suffix length S is computed from the cached prompts).

Pairs (test split): within an environment x tier cell, a top-third-SE task and a
bottom-third-SE task, greedily matched on fail_rate (|diff| <= 0.1 first, then
nearest). Each pair is patched both ways (high -> low, low -> high).

Readout (PLAN.md M5b, M1): TUL along a *fixed reply*. For each recipient we take its
own unpatched HF greedy reply (up to `reply_tokens`, cut at the first tool call or
end of turn) and teacher-force it under the patch:
    TUL_fix = logit(1 - prod_{t in reply} (1 - p_t(<tool_call>)))
so a call that would come late in the reply still counts. The donor's reference is
its own TUL_fix on its own reply. Effects are reported raw (patched - recipient)
and normalised by (donor - recipient) for pairs with |donor - recipient| >= 1 nat.

Patches at a layer l (all S suffix positions at once unless noted):
  full     - recipient residual replaced by the donor's
  1d:<dir> - only the component along unit d is swapped: h + ((h_d - h) . d) d
             for d in SE^dm, act^dm and a random unit vector
  pos      - full patch at one suffix position at a time (best layers only)

Outputs: results/e4/patch/{pairs,replies,sweep,dims,pos}.<hash>.parquet

Usage:  python -m toolmech.patch [--stages pairs,replies,sweep,dims,pos] [--n_pairs 300]
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

from toolmech import hf
from toolmech.config import load_config
from toolmech.prompts import load_prompts


def _log1mexp64(x):
    """log(1 - exp(x)) in float64 for x < 0, without the -1e-7 clamp of tul._log1mexp
    (that clamp floors TUL at about -16.1 nats for recipients that almost never call)."""
    x = x.double().clamp(max=-1e-300)
    return torch.where(x > -0.6931, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


def outpath(cfg, name, ext="parquet"):
    p = cfg.path("e4", f"patch/{name}", ext)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Pairs
# ---------------------------------------------------------------------------
def make_pairs(cfg, n_pairs, seed=0, tol=0.1):
    lab = pd.read_parquet(cfg.path("e1", "labels", "parquet"))
    lab = lab[lab.split == "test"] if "split" in lab else lab
    rng = np.random.default_rng(seed)
    rows = []
    for (env, tier), g in lab.groupby(["env", "tier"]):
        g = g.dropna(subset=["SE"])
        if len(g) < 6 or g.SE.nunique() < 2:
            continue
        lo_q, hi_q = g.SE.quantile(1 / 3), g.SE.quantile(2 / 3)
        hi = g[g.SE > hi_q] if (g.SE > hi_q).sum() >= 1 else g[g.SE >= hi_q]
        lo = g[g.SE <= lo_q]
        hi = hi.sample(frac=1, random_state=int(rng.integers(1 << 30)))
        lo_pool = lo.copy()
        for _, h in hi.iterrows():
            if lo_pool.empty:
                break
            d = (lo_pool.fail_rate - h.fail_rate).abs()
            j = d.idxmin()
            l = lo_pool.loc[j]
            if l.SE >= h.SE:
                continue
            rows.append({"env": env, "tier": tier, "hi": str(h.id), "lo": str(l.id), "SE_hi": h.SE, "SE_lo": l.SE,
                         "fail_hi": h.fail_rate, "fail_lo": l.fail_rate, "fail_matched": bool(d[j] <= tol)})
            lo_pool = lo_pool.drop(j)
    df = pd.DataFrame(rows)
    # prefer fail-matched pairs, then spread over cells
    df = df.sort_values("fail_matched", ascending=False, kind="stable")
    if len(df) > n_pairs:
        df = (df.assign(r=df.groupby(["env", "tier"]).cumcount())
              .sort_values(["fail_matched", "r"], ascending=[False, True]).head(n_pairs).drop(columns="r"))
    return df.reset_index(drop=True)


def shared_suffix_len(tok, texts):
    ids = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    n = 0
    while all(len(x) > n for x in ids) and len({x[-1 - n] for x in ids}) == 1:
        n += 1
    return n


# ---------------------------------------------------------------------------
# Fixed replies and teacher-forced TUL
# ---------------------------------------------------------------------------
@torch.no_grad()
def greedy_replies(model, tok, texts, max_new, bs=8):
    stop = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    call = hf.tool_token_ids(tok)[0]
    out = []
    for b in range(0, len(texts), bs):
        ids, mask, _ = hf.encode(tok, texts[b:b + bs])
        g = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new, do_sample=False,
                           pad_token_id=tok.pad_token_id, temperature=None, top_p=None, top_k=None)
        for row in g[:, ids.shape[1]:].tolist():
            r = []
            for t in row:
                if t == call or t in stop:     # reply is read up to (not including) the call / end
                    r.append(t)
                    break
                r.append(t)
            out.append(r)
    return out


class Patcher:
    """Teacher-forced TUL_fix with optional residual patches at one layer."""

    def __init__(self, model, tok, S):
        self.model, self.tok, self.S = model, tok, S
        self.call = hf.tool_token_ids(tok)[0]

    def batch(self, prompts, replies):
        """Left-padded prompt+reply ids; returns ids, mask, prompt_end [B], reply_len [B]."""
        seqs, pe, rl = [], [], []
        for p, r in zip(prompts, replies):
            pi = self.tok(p, add_special_tokens=False)["input_ids"]
            seqs.append(pi + list(r))
            pe.append(len(pi))
            rl.append(len(r))
        T = max(len(s) for s in seqs)
        pad = [T - len(s) for s in seqs]
        ids = torch.tensor([[self.tok.pad_token_id] * p + s for p, s in zip(pad, seqs)], device="cuda")
        mask = torch.tensor([[0] * p + [1] * len(s) for p, s in zip(pad, seqs)], device="cuda")
        pe = torch.tensor([p + e for p, e in zip(pad, pe)], device="cuda")     # padded prompt end
        return ids, mask, pe, torch.tensor(rl, device="cuda")

    def suffix_idx(self, pe):
        return pe.unsqueeze(1) - self.S + torch.arange(self.S, device=pe.device).unsqueeze(0)   # [B, S]

    @torch.no_grad()
    def capture(self, prompts, layers):
        """Residuals at the S suffix positions: {layer: [B, S, d]} (float32, on GPU)."""
        ids, mask, _, _ = self.batch(prompts, [[] for _ in prompts])
        pe = mask.sum(1) * 0 + ids.shape[1]
        idx = self.suffix_idx(pe)
        store, hs = {}, []
        for l in layers:
            def hook(_m, _i, out, l=l):
                h = out[0] if isinstance(out, tuple) else out
                store[l] = h.gather(1, idx.unsqueeze(-1).expand(-1, -1, h.shape[-1])).clone()
            hs.append(hf.blocks(self.model)[l].register_forward_hook(hook))
        try:
            self.model.model(input_ids=ids, attention_mask=mask)
        finally:
            for h in hs:
                h.remove()
        return store

    @torch.no_grad()
    def tul_fix(self, prompts, replies, layer=None, donor=None, dirs=None, pos=None):
        """TUL_fix [B]. donor: [B, S, d] states to write at `layer`; dirs: optional unit
        vector [d] -> 1-D patch along it; pos: optional single suffix offset to patch."""
        ids, mask, pe, rl = self.batch(prompts, replies)
        handle = None
        if layer is not None:
            idx = self.suffix_idx(pe)

            def hook(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                if h.shape[1] == 1:
                    return out
                ix = idx if pos is None else idx[:, pos:pos + 1]
                dn = donor if pos is None else donor[:, pos:pos + 1]
                gi = ix.unsqueeze(-1).expand(-1, -1, h.shape[-1])
                cur = h.gather(1, gi)
                if dirs is None:
                    new = dn.to(h.dtype)
                else:
                    d = dirs.to(h.device, h.dtype)
                    new = cur + ((dn.to(h.dtype) - cur) @ d).unsqueeze(-1) * d
                h = h.scatter(1, gi, new)
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            handle = hf.blocks(self.model)[layer].register_forward_hook(hook)
        try:
            hid = self.model.model(input_ids=ids, attention_mask=mask).last_hidden_state
        finally:
            if handle:
                handle.remove()
        # logits at positions pe-1 ... pe+rl-2 predict reply tokens 0..rl-1
        B = ids.shape[0]
        R = int(rl.max())
        pidx = (pe - 1).unsqueeze(1) + torch.arange(R, device=ids.device).unsqueeze(0)
        valid = torch.arange(R, device=ids.device).unsqueeze(0) < rl.unsqueeze(1)
        pidx = pidx.clamp(max=ids.shape[1] - 1)
        hsel = hid.gather(1, pidx.unsqueeze(-1).expand(-1, -1, hid.shape[-1]))
        lp = self.model.lm_head(hsel).float().log_softmax(-1)[..., self.call]       # [B, R]
        s = (_log1mexp64(lp) * valid.double()).sum(1)                              # log P(no call)
        return (_log1mexp64(s) - s).float().cpu()


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def directions(cfg, mode, layer):
    D = torch.load(cfg.path("e3", f"directions_{mode}", "pt"), weights_only=False)
    out = {}
    for n in ("SE^dm", "act^dm"):
        v = D[n][layer].float()
        if torch.isfinite(v).all() and v.norm() > 0:
            out[n] = (v / v.norm()).cuda()
    g = torch.Generator().manual_seed(4321 + layer)
    r = torch.randn(D["SE^dm"].shape[1], generator=g)
    out["random"] = (r / r.norm()).cuda()
    return out


def run(cfg, stages, n_pairs, mode, reply_tokens, bs, layers_pos, pos_pairs=100):
    tok, model = hf.load_hf(cfg)
    recs = {str(r["id"]): r for r in load_prompts(cfg, "test", mode)}
    S = shared_suffix_len(tok, [r["prompt_text"] for r in list(recs.values())[:400]])
    print(f"[{mode}] shared suffix tokens S={S}", flush=True)
    P = Patcher(model, tok, S)
    L = hf.n_layers(model)

    ppath = outpath(cfg, f"pairs_{mode}_n{n_pairs}")
    if not os.path.exists(ppath):
        make_pairs(cfg, n_pairs).to_parquet(ppath)
    pairs = pd.read_parquet(ppath)
    print(f"pairs={len(pairs)} fail-matched={pairs.fail_matched.mean():.2f} cells={pairs.groupby(['env','tier']).ngroups}",
          flush=True)
    tids = sorted(set(pairs.hi) | set(pairs.lo))

    rpath = outpath(cfg, f"replies_{mode}_n{n_pairs}")
    if not os.path.exists(rpath):
        t0 = time.time()
        reps = greedy_replies(model, tok, [recs[i]["prompt_text"] for i in tids], reply_tokens, bs)
        base = []
        for b in range(0, len(tids), bs):
            base += P.tul_fix([recs[i]["prompt_text"] for i in tids[b:b + bs]], reps[b:b + bs]).tolist()
        pd.DataFrame({"id": tids, "reply": reps, "tul_fix": base,
                      "called": [r[-1] == P.call if r else False for r in reps]}).to_parquet(rpath)
        print(f"replies {len(tids)} in {time.time() - t0:.0f}s", flush=True)
    rep = pd.read_parquet(rpath).set_index("id")
    reply = {i: list(rep.loc[i, "reply"]) for i in tids}
    base = rep.tul_fix.to_dict()

    # both directions: (donor, recipient)
    jobs = [(r.hi, r.lo, "hi->lo", k) for k, r in pairs.iterrows()] + \
           [(r.lo, r.hi, "lo->hi", k) for k, r in pairs.iterrows()]

    def sweep(name, layers, variants, max_pairs=None):
        path = outpath(cfg, f"{name}_{mode}_n{n_pairs}")
        jb_all = [j for j in jobs if max_pairs is None or j[3] < max_pairs]
        if os.path.exists(path):
            return pd.read_parquet(path)
        rows = []
        t0 = time.time()
        for b in range(0, len(jb_all), bs):
            jb = jb_all[b:b + bs]
            don_p = [recs[d]["prompt_text"] for d, _, _, _ in jb]
            rec_p = [recs[r]["prompt_text"] for _, r, _, _ in jb]
            rec_r = [reply[r] for _, r, _, _ in jb]
            states = P.capture(don_p, layers)
            for l in layers:
                for vname, kw in variants(l):
                    t = P.tul_fix(rec_p, rec_r, layer=l, donor=states[l], **kw)
                    for (d, r, direc, k), v in zip(jb, t.tolist()):
                        rows.append({"pair": k, "direction": direc, "donor": d, "recipient": r, "layer": l,
                                     "patch": vname, "tul_patched": v, "tul_rec": base[r], "tul_don": base[d]})
            if b == 0:
                print(f"[{name}] first batch {time.time() - t0:.0f}s; est total "
                      f"{(time.time() - t0) * len(jb_all) / len(jb) / 60:.0f} min", flush=True)
        df = pd.DataFrame(rows)
        df["effect"] = df.tul_patched - df.tul_rec
        gap = df.tul_don - df.tul_rec
        df["norm_effect"] = np.where(gap.abs() >= 1, df.effect / gap, np.nan)
        df.to_parquet(path)
        print(f"[{name}] done in {time.time() - t0:.0f}s", flush=True)
        return df

    best = None
    if "sweep" in stages:
        sw = sweep("sweep", list(range(L)), lambda l: [("full", {})])
        m = sw.groupby(["layer", "direction"]).effect.mean().unstack()
        print(m.round(3).to_string(), flush=True)
        # best layers: largest mean |effect| in the hi->lo direction
        best = [int(x) for x in m["hi->lo"].abs().sort_values(ascending=False).index[:3]]
        print(f"best patch layers {best}", flush=True)
    if best is None:
        best = layers_pos
    if "dims" in stages:
        dims_layers = sorted(set(best) | set(layers_pos or []))
        dl = {l: directions(cfg, "sparse_tool", l) for l in dims_layers}
        sweep("dims", dims_layers, lambda l: [("full", {})] + [(f"1d:{n}", {"dirs": v}) for n, v in dl[l].items()])
    if "pos" in stages:
        sweep("pos", best[:2], lambda l: [(f"pos{p - S}", {"pos": p}) for p in range(S)], max_pairs=pos_pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="sweep,dims,pos")
    ap.add_argument("--n_pairs", type=int, default=300)
    ap.add_argument("--mode", default="sparse_tool")
    ap.add_argument("--reply_tokens", type=int, default=256)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--pos_pairs", type=int, default=100, help="pairs used in the per-position stage")
    ap.add_argument("--layers", default="", help="extra layers for dims (e.g. best SE layers)")
    a = ap.parse_args()
    cfg = load_config()
    run(cfg, set(a.stages.split(",")), a.n_pairs, a.mode, a.reply_tokens, a.bs,
        [int(x) for x in a.layers.split(",") if x], a.pos_pairs)


if __name__ == "__main__":
    main()
