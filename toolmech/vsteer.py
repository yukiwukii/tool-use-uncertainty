"""Residual-stream steering inside vLLM (V0 engine), for the E4 sampling readouts.

Why: E4 needs ~20 sampled replies per task for every (direction, layer, k, mode);
HF fp32 generation is far too slow for that. vLLM is the engine E1's labels came
from, so k = 0 reproduces the E1 call rates (a built-in check).

Semantics match `hf.steer` (PLAN.md M5): add `vec` to the output of decoder block
`layer` at every *prompt* position from the start of the last user turn
(`<|im_start|>user`) onward, in the prefill forward only. Decode steps are not
steered; generated tokens see the steered prompt through the KV cache.

vLLM specifics:
- V0 engine (VLLM_USE_V1=0) so the model lives in this process and takes hooks.
- No chunked prefill and no prefix caching, so a prefill forward holds whole prompts
  and every prompt's KV is computed under the current steering vector.
- Qwen3 decoder layers return (hidden, residual); the residual stream after block l
  is hidden + residual, so adding `vec` to `hidden` adds it to the stream.
- The mask is built in a pre-hook on embed_tokens (Qwen3Model.__call__ bypasses hooks).
- Decode steps are CUDA-graph replays (hooks do not fire) or eager forwards with
  num_prefills == 0 (the hook returns early).
"""
import os

os.environ.setdefault("VLLM_USE_V1", "0")

import torch  # noqa: E402

USER_MARK = ("<|im_start|>", "user")


class SteerableLLM:
    def __init__(self, cfg, gpu_memory_utilization=None, max_model_len=None, dtype="bfloat16", max_num_seqs=1024):
        from vllm import LLM, SamplingParams
        v = cfg["vllm"]
        self.SamplingParams = SamplingParams
        self.llm = LLM(model=cfg["model_id"], tensor_parallel_size=1, dtype=dtype,
                       max_model_len=max_model_len or 8192,
                       gpu_memory_utilization=gpu_memory_utilization or v["gpu_memory_utilization"],
                       seed=cfg["seed"], enable_prefix_caching=False, enable_chunked_prefill=False,
                       max_num_seqs=max_num_seqs)
        self.tok = self.llm.get_tokenizer()
        self.model = self._find_model()
        self.layers = self.model.model.layers
        self.im_start = self.tok.convert_tokens_to_ids(USER_MARK[0])
        ut = self.tok.encode(USER_MARK[1], add_special_tokens=False)
        assert len(ut) == 1, ut
        self.user_tok = ut[0]
        self._vec, self._layer, self._mask = None, None, None
        # Qwen3Model's __call__ is replaced by vLLM's compile wrapper, which calls forward()
        # directly and skips module hooks; the embedding sees the same input_ids.
        self.model.model.embed_tokens.register_forward_pre_hook(self._pre_hook, with_kwargs=True)
        self._handle = None

    def _find_model(self):
        ex = self.llm.llm_engine.model_executor
        w = ex.driver_worker
        w = getattr(w, "worker", w)
        return w.model_runner.model

    # ---- hooks -------------------------------------------------------------
    def _pre_hook(self, _m, args, kwargs):
        """Before each forward: build the steering mask over flattened prefill tokens."""
        self._mask = None
        if self._vec is None:
            return
        from vllm.forward_context import get_forward_context
        md = get_forward_context().attn_metadata
        n_pref = getattr(md, "num_prefills", 0) if md is not None else 0
        if not n_pref:
            return
        ids = args[0] if args else kwargs["input_"]
        n_tok = md.num_prefill_tokens
        qsl = md.query_start_loc[: n_pref + 1].tolist() if md.query_start_loc is not None \
            else md.seq_start_loc[: n_pref + 1].tolist()
        hit = torch.zeros_like(ids, dtype=torch.bool)
        hit[:-1] = (ids[:-1] == self.im_start) & (ids[1:] == self.user_tok)
        mask = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
        hit_idx = hit[:n_tok].nonzero().flatten().tolist()
        for s, e in zip(qsl[:-1], qsl[1:]):
            starts = [i for i in hit_idx if s <= i < e]
            mask[(starts[-1] if starts else e):e] = True
        self._mask = mask

    def _layer_hook(self, _m, _inp, out):
        if self._mask is None:
            return out
        hidden, resid = out
        m = self._mask[: hidden.shape[0]].unsqueeze(-1).to(hidden.dtype)
        return hidden + m * self._vec.to(hidden.device, hidden.dtype), resid

    def set_steering(self, layer=None, vec=None):
        """Steer at block `layer` with `vec` [d] (already scaled), or clear with None."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._vec, self._layer = None, None
        if vec is not None:
            self._vec = torch.as_tensor(vec, dtype=torch.float32).cuda()
            self._layer = layer
            self._handle = self.layers[layer].register_forward_hook(self._layer_hook)

    def generate(self, prompts, layer=None, vec=None, **sp):
        self.set_steering(layer, vec)
        try:
            return self.llm.generate(prompts, self.SamplingParams(**sp), use_tqdm=False)
        finally:
            self.set_steering(None)
