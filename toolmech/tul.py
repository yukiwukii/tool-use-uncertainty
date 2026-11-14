"""M1 / E2: tool-use log-odds (TUL), first-token and windowed.

First-token TUL (plan M1):  logit p(<tool_call> | prompt).

Windowed TUL (deviation, needed for `sparse_tool` where ~58% of calls start after
some text): follow the model's greedy reply for up to N tokens and take the
probability that a <tool_call> is emitted at *some* position on that path,

    TUL_N = logit( 1 - prod_{t<N'} (1 - p_t(tool)) ),

where N' = min(N, position where the greedy path calls a tool or ends). N=1
recovers first-token TUL. All windows come from one `generate` pass with
`output_logits=True`, so hooks (steering, patching) see exactly the forwards that
produced the path: steered prefill, unsteered decode (see hf.steer).

Usage:  python -m toolmech.tul [--modes sparse_tool,current] [--splits train,test]
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from toolmech import hf
from toolmech.config import load_config
from toolmech.prompts import load_prompts

DEFAULT_WINDOWS = (1, 8, 32, 64)


def _log1mexp(x):
    """log(1 - exp(x)) for x <= 0, numerically stable."""
    x = x.clamp(max=-1e-7)
    return torch.where(x > -0.6931, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


def windowed_tul(step_logits, gen_ids, tool_ids, eos_ids, windows=DEFAULT_WINDOWS):
    """TUL for each window from per-step logits.

    step_logits: [B, T, V] logits for each generated position (fp32)
    gen_ids:     [B, T]    greedy tokens actually generated
    Returns dict window -> [B] TUL, plus 'call_pos' [B] (first tool-call step, -1 if none)
    and 'path_len' [B] (steps up to and including the first call/EOS).
    """
    lp = step_logits.float().log_softmax(-1)
    lp_tool = torch.logsumexp(lp[..., tool_ids], dim=-1)            # [B, T]
    log_no_tool = _log1mexp(lp_tool)                                # log(1 - p_t)

    B, T = gen_ids.shape
    stop = torch.isin(gen_ids, torch.tensor(list(tool_ids) + list(eos_ids), device=gen_ids.device))
    # path_len: steps through the first stop token (inclusive); T if no stop
    first_stop = torch.where(stop.any(1), stop.float().argmax(1), torch.full((B,), T, device=gen_ids.device))
    path_len = torch.clamp(first_stop + 1, max=T)
    is_call = torch.isin(gen_ids, torch.tensor(list(tool_ids), device=gen_ids.device))
    call_pos = torch.where(is_call.any(1), is_call.float().argmax(1), torch.full((B,), -1, device=gen_ids.device))

    steps = torch.arange(T, device=gen_ids.device).unsqueeze(0)
    out = {}
    for n in windows:
        mask = (steps < torch.clamp(path_len, max=n).unsqueeze(1)).float()
        s = (log_no_tool * mask).sum(1)                             # log P(no call in window)
        out[n] = _log1mexp(s) - s                                   # logit(1 - e^s)
    out["call_pos"] = call_pos
    out["path_len"] = path_len
    return out


@torch.no_grad()
def tul_batch(model, tok, texts, windows=DEFAULT_WINDOWS, device="cuda"):
    """Greedy-generate max(windows) tokens and return windowed TULs for a batch."""
    ids, mask, _ = hf.encode(tok, texts, device)
    n_max = max(windows)
    g = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=n_max, do_sample=False,
                       return_dict_in_generate=True, output_logits=True,
                       pad_token_id=tok.pad_token_id, temperature=None, top_p=None, top_k=None)
    step_logits = torch.stack(g.logits, dim=1)                      # [B, T, V]
    gen = g.sequences[:, ids.shape[1]:]
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    return windowed_tul(step_logits, gen, hf.tool_token_ids(tok), eos, windows), gen


def run(cfg, modes, splits, batch_size=8):
    tok, model = hf.load_hf(cfg)
    windows = tuple(cfg.get("tul", {}).get("windows", DEFAULT_WINDOWS))
    for mode in modes:
        for split in splits:
            path = cfg.path("e2", f"tul_{split}_{mode}", "parquet")
            if os.path.exists(path):
                continue
            recs = load_prompts(cfg, split, mode)
            order = np.argsort([r["n_tokens"] for r in recs])          # length-sorted batches
            rows = []
            for b in range(0, len(order), batch_size):
                chunk = [recs[i] for i in order[b:b + batch_size]]
                res, gen = tul_batch(model, tok, [r["prompt_text"] for r in chunk], windows)
                for j, r in enumerate(chunk):
                    row = {"id": str(r["id"]), "split": split, "mode": mode,
                           "call_pos": int(res["call_pos"][j]), "path_len": int(res["path_len"][j]),
                           "greedy_text": tok.decode(gen[j][: int(res["path_len"][j])])[:200]}
                    row.update({f"tul_{n}": float(res[n][j]) for n in windows})
                    rows.append(row)
            pd.DataFrame(rows).to_parquet(path)
            print(path, len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default=None, help="default: cfg.modes.primary,secondary")
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--batch_size", type=int, default=8)
    a = ap.parse_args()
    cfg = load_config()
    m = cfg.get("modes", {"primary": "sparse_tool", "secondary": "current"})
    modes = a.modes.split(",") if a.modes else [m["primary"], m["secondary"]]
    run(cfg, modes, a.splits.split(","), a.batch_size)


if __name__ == "__main__":
    main()
