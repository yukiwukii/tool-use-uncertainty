"""HF model access: residual-stream reads, steering, and the tool-call token.

Residual convention: `resid[l]` is the output of decoder block l (0-indexed),
i.e. the raw residual stream *before* the final RMSNorm. This is read with
forward hooks on `model.model.layers[l]`, never from `output_hidden_states`
(whose last entry is post-norm).

Plan deviation (logged in REPORT.md): plain HF forward hooks replace
TransformerLens/nnsight. They run the exact HF numerics, so the
"TL logits differ from HF" pitfall cannot arise.
"""
from contextlib import contextmanager

import torch

TOOL_CALL_TOKEN = "<tool_call>"


def load_hf(cfg, device="cuda", dtype=torch.float32):
    """fp32 by default. In bf16, TUL for prompts near the decision boundary
    moves by up to ~0.9 nats with batch composition and is biased by ~0.5 nats
    vs fp32 (tests/test_hf_gpu.py); in fp32 batched == single to 1e-4."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], torch_dtype=dtype)
    model.to(device).eval()
    return tok, model


def blocks(model):
    return model.model.layers


def n_layers(model):
    return len(blocks(model))


def tool_token_ids(tok):
    """Ids of tokens that begin a tool call. For Qwen3 with When2Tool's xml
    format this is the single special token `<tool_call>`."""
    tid = tok.convert_tokens_to_ids(TOOL_CALL_TOKEN)
    assert tid is not None and tid != tok.unk_token_id, "missing <tool_call> token"
    return [tid]


def tul_from_logits(logits, tool_ids):
    """M1: log p(tool) - log(1 - p(tool)) from last-position logits [B, V]."""
    lp = logits.float().log_softmax(-1)
    lp_tool = torch.logsumexp(lp[:, tool_ids], dim=-1)
    return lp_tool - torch.log1p(-lp_tool.exp().clamp(max=1 - 1e-7))


def encode(tok, texts, device="cuda"):
    """Left-padded batch. Returns input_ids, attention_mask and `pad` (number of
    left pad tokens per row) so per-row token indices can be shifted."""
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    pad = (enc["attention_mask"] == 0).sum(1)
    return enc["input_ids"].to(device), enc["attention_mask"].to(device), pad.to(device)


def char_to_token(tok, text, char_idx):
    """Unpadded token index of the token covering `char_idx` in `text`."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    for i, (s, e) in enumerate(enc["offset_mapping"]):
        if s <= char_idx < e or s >= char_idx:
            return i
    return len(enc["input_ids"]) - 1


def _out_tensor(out):
    return out[0] if isinstance(out, tuple) else out


@contextmanager
def capture(model, gather_idx, layer_ids=None):
    """Collect residual states at chosen positions.

    gather_idx: LongTensor [B, P] of (padded) token positions per row.
    Yields a dict that, after the forward pass, maps layer -> [B, P, d] (float32, cpu).
    Only the first forward (prefill) is captured.
    """
    layer_ids = list(range(n_layers(model))) if layer_ids is None else list(layer_ids)
    store, handles = {}, []

    def make(l):
        def hook(_m, _inp, out):
            if l in store:
                return
            h = _out_tensor(out)
            idx = gather_idx.to(h.device).unsqueeze(-1).expand(-1, -1, h.shape[-1])
            store[l] = h.gather(1, idx).float().cpu()
        return hook

    for l in layer_ids:
        handles.append(blocks(model)[l].register_forward_hook(make(l)))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


@contextmanager
def steer(model, layer, vec, start_idx):
    """M5: add `vec` [d] to the residual output of block `layer` at every prompt
    position >= start_idx (per-row padded index, LongTensor [B]).

    Applied only during prefill (the forward whose length covers the prompt);
    decode steps with a KV cache are left unsteered, so generated tokens see the
    steered prompt through attention only.
    """
    def hook(_m, _inp, out):
        h = _out_tensor(out)
        if h.shape[1] == 1:  # decode step
            return out
        T = h.shape[1]
        pos = torch.arange(T, device=h.device).unsqueeze(0)
        mask = (pos >= start_idx.to(h.device).unsqueeze(1)).unsqueeze(-1).to(h.dtype)
        h = h + mask * vec.to(h.device, h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    handle = blocks(model)[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def last_logits_and_resid(model, tok, texts, layer_ids=None, device="cuda"):
    """One forward over prompts: TUL-ready last-token logits [B, V] (float32)
    and last-token residuals [B, L, d] (float32, cpu)."""
    ids, mask, _ = encode(tok, texts, device)
    gather = torch.full((ids.shape[0], 1), ids.shape[1] - 1, device=device)
    with capture(model, gather, layer_ids) as store:
        logits = model(input_ids=ids, attention_mask=mask).logits[:, -1].float()
    layers = sorted(store)
    resid = torch.stack([store[l][:, 0] for l in layers], dim=1)
    return logits, resid
