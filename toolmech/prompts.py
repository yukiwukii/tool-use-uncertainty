"""Build and cache the exact prompt strings used by both vLLM and HF.

Prompts are rendered from When2Tool's own `init_state` (system prompt, the
ListManipulation contract, the per-mode user message) and the model's chat
template with `tools=`. Every step reads these cached strings, so vLLM sampling
and HF hidden-state reads always see byte-identical prompts.

Modes: 'current', 'sparse_tool' (primary analysis mode), 'force_tool', 'no_tool' (also the hard_no_tool text).

Usage:  python -m toolmech.prompts            # builds all splits x modes
"""
import json
import os

from toolmech.config import load_config, load_tasks, task_env

MODES = ("current", "sparse_tool", "force_tool", "no_tool")
SPLITS = ("train", "test", "multihop_train", "multihop_test")
USER_TURN_MARKER = "<|im_start|>user"


def _tokenizer(cfg):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(cfg["model_id"])


def messages_and_tools(task, mode):
    """When2Tool's initial conversation state for a task (no_reasoning variant)."""
    from utils import init_state, get_system_prompt
    st = init_state(task, get_system_prompt("xml"), record_mode="off",
                    prompt_mode=mode, require_reasoning=False, tool_format="xml")
    return st["messages"], st["tools"]


def render(tok, messages, tools, add_generation_prompt=True):
    kw = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
    if tools:
        kw["tools"] = tools
    return tok.apply_chat_template(messages, **kw)


def build(cfg, split, mode, tok=None):
    tok = tok or _tokenizer(cfg)
    out = []
    for t in load_tasks(cfg, split):
        messages, tools = messages_and_tools(t, mode)
        text = render(tok, messages, tools)
        out.append({
            "id": t["id"],
            "split": split,
            "mode": mode,
            "difficulty": t.get("difficulty", "unknown"),
            "env": task_env(t),
            "multi_step": bool(t.get("multi_step")),
            "messages": messages,
            "tools": tools,
            "prompt_text": text,
            # char offset of the (last) user turn; steering starts here (M5)
            "user_start_char": text.rfind(USER_TURN_MARKER),
            "n_tokens": len(tok(text, add_special_tokens=False)["input_ids"]),
        })
    return out


def load_prompts(cfg, split, mode):
    """Cached prompt records for (split, mode); built on first use."""
    path = cfg.path("prompts", f"{split}_{mode}", "jsonl")
    if not os.path.exists(path):
        recs = build(cfg, split, mode)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    with open(path) as f:
        return [json.loads(l) for l in f]


if __name__ == "__main__":
    cfg = load_config()
    for split in SPLITS:
        for mode in MODES:
            recs = load_prompts(cfg, split, mode)
            print(f"{split:15s} {mode:11s} n={len(recs)} "
                  f"max_tokens={max(r['n_tokens'] for r in recs)}")
