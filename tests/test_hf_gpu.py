"""GPU checks for toolmech/hf.py against plain HF numerics.

Run:  HF_HOME=$PWD/.hf_cache HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests/test_hf_gpu.py -q -s
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

from toolmech import hf  # noqa: E402
from toolmech.config import load_config  # noqa: E402
from toolmech.prompts import load_prompts  # noqa: E402

N = 8  # prompts per check


@pytest.fixture(scope="module")
def setup():
    cfg = load_config()
    tok, model = hf.load_hf(cfg)
    recs = {m: load_prompts(cfg, "test", m)[:: 2250 // N][:N] for m in ("current", "force_tool", "no_tool")}
    return cfg, tok, model, recs


@torch.no_grad()
def test_resid_matches_hidden_states(setup):
    _, tok, model, recs = setup
    text = recs["current"][0]["prompt_text"]
    ids, mask, _ = hf.encode(tok, [text])
    gather = torch.tensor([[ids.shape[1] - 1]], device=ids.device)
    with hf.capture(model, gather) as store:
        out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    L = hf.n_layers(model)
    assert len(store) == L
    for l in range(L - 1):
        ref = out.hidden_states[l + 1][0, -1].float().cpu()
        torch.testing.assert_close(store[l][0, 0], ref, rtol=0, atol=0)
    # last block: our hook is pre-norm, hidden_states[-1] is post-norm
    assert not torch.allclose(store[L - 1][0, 0], out.hidden_states[L][0, -1].float().cpu())
    normed = model.model.norm(store[L - 1][0, 0].to(model.device, model.dtype)).float().cpu()
    torch.testing.assert_close(normed, out.hidden_states[L][0, -1].float().cpu(), rtol=1e-2, atol=1e-2)


@torch.no_grad()
def test_left_padding_matches_single(setup):
    _, tok, model, recs = setup
    texts = [r["prompt_text"] for r in recs["current"]]
    batch_logits, batch_resid = hf.last_logits_and_resid(model, tok, texts)
    tool_ids = hf.tool_token_ids(tok)
    max_dtul = 0.0
    for i, t in enumerate(texts):
        lg, rs = hf.last_logits_and_resid(model, tok, [t])
        dtul = (hf.tul_from_logits(lg, tool_ids) - hf.tul_from_logits(batch_logits[i : i + 1], tool_ids)).abs().item()
        max_dtul = max(max_dtul, dtul)
        assert lg.argmax(-1).item() == batch_logits[i].argmax(-1).item()
    print(f"\nmax |TUL(single) - TUL(batched)| = {max_dtul:.4f} nats")
    assert max_dtul < 1e-2  # fp32: padding must not move TUL (bf16 gave up to 0.9)


@torch.no_grad()
def test_steer_zero_and_nonzero(setup):
    _, tok, model, recs = setup
    r = recs["current"][0]
    ids, mask, pad = hf.encode(tok, [r["prompt_text"]])
    start = pad + hf.char_to_token(tok, r["prompt_text"], r["user_start_char"])
    base = model(input_ids=ids, attention_mask=mask).logits[:, -1].float()
    d = model.config.hidden_size
    with hf.steer(model, 18, torch.zeros(d), start):
        z = model(input_ids=ids, attention_mask=mask).logits[:, -1].float()
    torch.testing.assert_close(z, base, rtol=0, atol=0)
    v = torch.randn(d)
    v = 50 * v / v.norm()
    with hf.steer(model, 18, v, start):
        s = model(input_ids=ids, attention_mask=mask).logits[:, -1].float()
    assert (s - base).abs().max() > 0.5


def test_user_start_token(setup):
    _, tok, _, recs = setup
    for r in recs["current"]:
        i = hf.char_to_token(tok, r["prompt_text"], r["user_start_char"])
        ids = tok(r["prompt_text"], add_special_tokens=False)["input_ids"]
        assert tok.convert_ids_to_tokens(ids[i]) == "<|im_start|>"
        assert tok.decode(ids[i : i + 2]).startswith("<|im_start|>user")


@torch.no_grad()
def test_tool_token_and_tul_sign(setup):
    _, tok, model, recs = setup
    tool_ids = hf.tool_token_ids(tok)
    assert len(tok("<tool_call>", add_special_tokens=False)["input_ids"]) == 1
    rows = []
    for mode in ("current", "force_tool", "no_tool"):
        lg, _ = hf.last_logits_and_resid(model, tok, [r["prompt_text"] for r in recs[mode]])
        tul = hf.tul_from_logits(lg, tool_ids)
        greedy = lg.argmax(-1)
        for t, g in zip(tul.tolist(), greedy.tolist()):
            is_tool = g in tool_ids
            rows.append((mode, round(t, 2), tok.convert_ids_to_tokens(g)))
            # greedy picks <tool_call> => p(tool) is the max; TUL>0 => greedy is <tool_call>
            if t > 0:
                assert is_tool
    for row in rows:
        print(row)
    by_mode = {m: [t for mm, t, _ in rows if mm == m] for m in ("current", "force_tool", "no_tool")}
    mean = {m: sum(v) / len(v) for m, v in by_mode.items()}
    print("mean TUL by mode:", mean)
    assert mean["force_tool"] > mean["current"] > mean["no_tool"]


@torch.no_grad()
def test_windowed_tul_consistency(setup):
    from toolmech.tul import tul_batch
    cfg, tok, model, recs = setup
    texts = [r["prompt_text"] for r in recs["current"][:4]]
    res, _ = tul_batch(model, tok, texts, windows=(1, 8))
    lg, _ = hf.last_logits_and_resid(model, tok, texts)
    first = hf.tul_from_logits(lg, hf.tool_token_ids(tok))
    torch.testing.assert_close(res[1].cpu(), first.cpu(), rtol=0, atol=1e-3)  # window 1 == M1 TUL
    assert (res[8] >= res[1] - 1e-4).all()                                   # windows only add mass
    single = torch.cat([tul_batch(model, tok, [t], windows=(8,))[0][8] for t in texts])
    torch.testing.assert_close(res[8].cpu(), single.cpu(), rtol=0, atol=1e-2)


@torch.no_grad()
def test_tul_pass_captures_same_resid(setup):
    from toolmech.tul import tul_batch
    cfg, tok, model, recs = setup
    texts = [r["prompt_text"] for r in recs["current"][:3]]
    L = hf.n_layers(model)
    res, _ = tul_batch(model, tok, texts, windows=(1, 8), capture_layers=range(L))
    _, ref = hf.last_logits_and_resid(model, tok, texts)
    assert res["resid"].shape == ref.shape == (3, L, model.config.hidden_size)
    # generate's prefill uses KV-cache kernels: fp32 rounding up to ~4e-4 in the last layer (values ~1e2)
    torch.testing.assert_close(res["resid"], ref, rtol=1e-3, atol=1e-3)
    # a wrong position (e.g. a decode step) would be far off: rows must not match each other
    assert (res["resid"][0] - res["resid"][1]).abs().max() > 1.0
