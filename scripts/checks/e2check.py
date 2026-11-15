import pandas as pd, numpy as np
from transformers import AutoTokenizer
from toolmech.config import load_config
cfg = load_config(); tok = AutoTokenizer.from_pretrained(cfg["model_id"])
W = 64
for split in ("train", "test"):
    for mode in ("sparse_tool", "current"):
        t = pd.read_parquet(cfg.path("e2", f"tul_{split}_{mode}", "parquet")).set_index("id")
        c = pd.read_parquet(cfg.path("e1", f"call_samples_{split}_{mode}", "parquet"))
        g = c[c.greedy].set_index("id")
        g.index = g.index.astype(str)
        df = t.join(g[["called", "call_char", "text"]], how="inner")
        hf_call = df.call_pos >= 0
        def vtok(r):
            if not r.called: return np.inf
            if r.call_char <= len(r.text) and r.call_char <= 200:
                return len(tok(r.text[:r.call_char], add_special_tokens=False)["input_ids"])
            n200 = len(tok(r.text[:200], add_special_tokens=False)["input_ids"])
            return n200 * r.call_char / 200          # extrapolated
        vpos = df.apply(vtok, axis=1)
        v_in = vpos < W
        agree = (hf_call == v_in)                    # same-window decision
        full_agree = (hf_call == df.called)
        late = (~hf_call) & df.called & ~v_in
        n = len(df)
        print(f"{split:5s} {mode:11s} n={n} greedy agree={full_agree.mean():.3f}  same-window agree={agree.mean():.4f} "
              f"late-call disagreements={int(late.sum())} same-window disagreements={int((~agree).sum())} "
              f"(HF-only calls {int((hf_call & ~v_in).sum())}, vLLM-only in-window {int((~hf_call & v_in).sum())})")
