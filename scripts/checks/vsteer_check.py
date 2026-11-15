"""Compare vLLM-steered vs HF-steered first-token distributions. Usage: vsteer_check.py vllm|hf OUT"""
import sys, json, torch, numpy as np, pandas as pd
from toolmech.config import load_config
from toolmech.prompts import load_prompts
cfg = load_config()
which, out = sys.argv[1], sys.argv[2]
L = 18
blob = torch.load(cfg.path("e2", "resid_train_sparse_tool", "pt"), weights_only=False)
lab = pd.read_parquet(cfg.path("e1", "labels", "parquet")).set_index("id").loc[[str(i) for i in blob["ids"]]]
X = blob["resid"][:, L].float()
a = torch.tensor(lab["act_sparse_tool"].to_numpy().astype(bool))
d = X[a].mean(0) - X[~a].mean(0); d = d / d.norm(); sd = float((X @ d).std())
recs = load_prompts(cfg, "test", "sparse_tool")[::150][:15]
texts = [r["prompt_text"] for r in recs]
res = {}
for k in (0, 4, -4):
    vec = k * sd * d
    if which == "vllm":
        from toolmech.vsteer import SteerableLLM
        if "llm" not in globals():
            llm = SteerableLLM(cfg, gpu_memory_utilization=0.3)
        o = llm.generate(texts, L, None if k == 0 else vec, n=1, max_tokens=1, temperature=0.0, logprobs=20)
        res[k] = [{str(t): float(lp.logprob) for t, lp in x.outputs[0].logprobs[0].items()} for x in o]
    else:
        from toolmech import hf
        if "model" not in globals():
            tok, model = hf.load_hf(cfg)
        res[k] = []
        for r in recs:
            ids, mask, pad = hf.encode(tok, [r["prompt_text"]])
            st = torch.tensor([hf.char_to_token(tok, r["prompt_text"], r["user_start_char"])], device="cuda")
            with torch.no_grad(), hf.steer(model, L, vec, st):
                lp = model(input_ids=ids, attention_mask=mask).logits[0, -1].float().log_softmax(-1)
            top = lp.topk(20)
            res[k].append({str(int(i)): float(v) for v, i in zip(top.values, top.indices)})
json.dump({"sd": sd, "res": {str(k): v for k, v in res.items()}}, open(out, "w"))
print("ok", sd)
