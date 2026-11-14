# Handoff: implementing PLAN.md (toolmech)

Status as of 2026-11-14 10:35 UTC. Read `PLAN.md` first, especially its **revision log**: the prompt mode, the TUL definition, the steering direction and E4 changed after E0. This file covers the environment, what has run, what is running, the traps found, and what to do next.

## TL;DR

- **Machine:** local box at `/tool-use-uncertainty` with one **NVIDIA A40 (48 GB)**, 96 CPUs, 503 GB RAM, 100 GB disk. No Merlin, devbox or HDFS any more; the user handles any leftover Merlin jobs from the previous handoff.
- **E0 passed:** all-layer probe test AUROC **0.9495** (paper 0.948).
- **E1 no-tool sampling is running** (train done; test in progress). The **E1 call sampling** has not started yet; see "Next steps" for the config edit that must come first.
- **Agreed changes** (in PLAN.md's revision log): `sparse_tool` is the primary prompt mode with `current` as secondary; windowed TUL; steering uses a stratified difference in means (`d_SE^dm`); activation patching added to E4; fp32 for all HF passes.
- Nothing is committed yet. Commit only when the user asks.

## Environment

- Project root `/tool-use-uncertainty` (git repo, branch `main`). `when2tool/`, `.venv/`, `.hf_cache/` and `results/` are untracked (see `.gitignore`; add `results/` before committing).
- `when2tool/`: github.com/Trustworthy-ML-Lab/when2tool at `ef0cf1e`. Data dumped with When2Tool's own HF loader to `when2tool/data/tasks_v1_{train,test,multihop_train,multihop_test}.json` (900 / 2250 / 180 / 450).
- `.venv/`: Python 3.11 built with `uv` from `toolmech/requirements.txt` (torch 2.6.0+cu124, vllm 0.8.5, transformers 4.55.2). `setuptools` had to be added: vLLM's engine fails to start without it.
- `.hf_cache/`: `Qwen/Qwen3-4B-Instruct-2507` at snapshot `cdbee75f…`. I wrote `refs/main` pointing at that snapshot by hand; without it, offline loads by model name fail.
- **Environment variables for every command:**
  ```bash
  export PYTHONPATH=/tool-use-uncertainty HF_HOME=/tool-use-uncertainty/.hf_cache HF_HUB_OFFLINE=1 HF_HUB_ENABLE_HF_TRANSFER=0
  ```
  The machine sets `HF_HUB_ENABLE_HF_TRANSFER=1` globally without `hf_transfer` installed, which breaks downloads.
- Use `.venv/bin/python`. Tests: `.venv/bin/python -m pytest tests -q -p no:cacheprovider` (the GPU tests skip without CUDA).

## What has run

| Step | Result | Where |
|---|---|---|
| Prompt caches | 12 caches (4 splits × `current`/`force_tool`/`no_tool`), `user_start_char` ≥ 0 everywhere, max 2,323 tokens. `sparse_tool` caches build on first use. | `results/prompts/*.jsonl` |
| `hf.py` GPU checks | 6/6 pass (`tests/test_hf_gpu.py`): hooks equal `hidden_states[l+1]`, last layer pre-norm, padding-invariant in fp32, steering identity at zero, user-turn index, TUL sign, windowed TUL consistency | — |
| **E0** (reduced) | Step 2 extract features 22 min; step 3 probe 38 s: test AUROC **0.9495**, within-tier 0.92–0.94; step 1 eval `current`/`no_reasoning` × 3 runs only, 18 min: accuracy **0.888** | `when2tool/probe_data/qwen3-4b-instruct/`, `when2tool/outputs/qwen3-4b-instruct/current.json`, log `results/e0/e0.log`, script `results/e0/run_e0.sh` |
| E1 smoke (`--limit 2`, 90 test tasks) | Passed: no missing labels, medium SE > 0 share 0.57; prompts byte-identical to cache (900/900); `need`~`fail_rate` r = 0.87; SE~fail_rate Spearman 0.65 | `results/e1_smoke/`, `results/e1_smoke.log` |
| Prompt-mode comparison (same 90 tasks, 20 samples) | call rate / mixed-task share / late-call share: `current` 0.92 / 7% / ~1%; `necessary_tool` 0.81 / 11% / ~1%; `sparse_tool` 0.17 / 33% / ~58% | scratch only; rerun with `toolmech.labels` call stage |

## What is running

- **E1 no-tool stage:** `python -m toolmech.labels --splits train,test --stage notool`, PID 12743, log `results/e1_notool.log`. Train is done (`results/e1/notool_samples_train.f8759166.parquet`, about 31 min). Test (22,500 samples) was in its follow-up rounds at 10:34; expected to finish around 10:50 UTC.
- Check it with: `grep -E "^round|Traceback" results/e1_notool.log | tail -3; ls results/e1/`.

## Code state (`toolmech/`)

| File | State |
|---|---|
| `config.py` | Done. Artefacts are `results/<step>/<name>.<cfg_hash>.<ext>`; the hash covers the **whole** config. |
| `prompts.py` | Done and run. `MODES` now includes `sparse_tool`. |
| `hf.py` | Done, GPU-tested. **`load_hf` defaults to fp32** (bf16 moved TUL by up to 0.9 nats near the decision boundary). |
| `labels.py` | Done, smoke-tested. Stages `notool` / `calls` / `merge`; `--limit N` writes to `results/e1_smoke/`. Call sampling runs for both `cfg.modes` (default `sparse_tool`, `current`); columns are suffixed `_<mode>` and the primary mode also fills the plain names. SE clustering uses When2Tool's normalisers and keeps integers exact. |
| `tul.py` | Done, tested (CPU and GPU), **not yet run on the full data**. Windowed TUL for N ∈ {1, 8, 32, 64} from one `generate(output_logits=True)` pass. |
| `directions`, `geometry`, `steer`, `patch`, `conversations`, `corrupt`, `trajectory`, `plots`, `REPORT.md` | Not written. `patch.py` is new in the revised plan (M5b). |
| `merlin/`, `gpu.sh` | Obsolete on this machine. |

Tests: `tests/test_labels.py` (4), `tests/test_tul.py` (3), `tests/test_hf_gpu.py` (6, GPU).

## Findings so far (for REPORT.md)

- **Over-calling:** in `current`, the model calls on 91% of test tasks; only 32 of 2,250 tasks vary across E0's 3 runs.
- **"Knows but doesn't act":** call behaviour predicts `need` at AUROC **0.52**, against the probe's 0.95. Calls happen on 93% of tasks that need a tool and 89% of tasks that don't. Environment base rate alone predicts `need` at 0.77, so report within-environment probe AUROCs before claiming the probe reads more than task type.
- **HistoricalYear is inverted:** it calls on 91% of tasks it could solve alone and 15% of tasks it would fail (AUROC 0.12). Check whether these hard tasks have low SE (the "hard but confident" H4 cell).
- **SE is confounded:** with fail_rate (Spearman 0.65) and with environment (mean SE 0.11 to 1.75 in the smoke test). That is why `d_SE^dm` is built within environment × tier and matched on fail rate.

## Deviations to log in REPORT.md

1. HF forward hooks instead of TransformerLens/nnsight (exact HF numerics).
2. Prompt source is `init_state`, including the ListManipulation contract message, for both vLLM and HF. E0's probe AUROC still matches the paper.
3. **fp32 for all HF passes.**
4. **E0 step 1 reduced** to `current`/`no_reasoning`, 3 runs (the plan uses no other setting).
5. **Sampling is T = 1, top_p = 1, top_k = −1, no repetition penalty.** When2Tool's vLLM wrapper otherwise inherits Qwen's `generation_config` (top_k = 20, top_p = 0.8).
6. **`act` is a greedy (T = 0) first reply on the cached prompts**, per mode. E0's `run_eval` used Qwen's default sampling; its mean is kept as `act_w2t`.
7. **Primary prompt mode `sparse_tool`, secondary `current`;** windowed TUL; stratified difference-in-means steering direction; activation patching added to E4 (all in PLAN.md's revision log).
8. Steering accuracy on all 2,250 tasks is probably too expensive; use the per-tier subset first.
9. `refs/main` in the HF cache was written by hand to pin the snapshot.

## Next steps

1. **Wait for the E1 no-tool stage to finish** (`results/e1/notool_samples_test.f8759166.parquet`).
2. **Config edit, then rename.** Add to `configs/qwen3_4b.yaml`:
   ```yaml
   modes:
     primary: sparse_tool
     secondary: current
   ```
   and under `tul:` replace `late_call_window: 8` with `windows: [1, 8, 32, 64]`; set `sampling.call_max_tokens: 512`. This changes the config hash. The no-tool stage does not read any of these keys, so **rename** `results/e1/notool_samples_{train,test}.f8759166.parquet` to the new hash instead of resampling (about an hour). Print the new hash with `python -c "from toolmech.config import load_config; print(load_config().hash)"` and log the rename as a deviation. Don't edit the config while a stage is running: the running process writes under the hash it started with.
3. **E1 call stage + merge:** `python -m toolmech.labels --splits train,test --stage calls`, then `--stage merge`. The two modes × 3,150 tasks × 21 samples should take about 30–40 min. Check that the late-call share in `sparse_tool` is near the smoke test's ~58%, and that call rates match the mode comparison.
4. **Look at HistoricalYear-hard** in `labels.parquet`: SE, fail_rate and call rate.
5. **E2:** `python -m toolmech.tul` (both modes, train + test). Choose the window per mode (smallest N with Spearman ρ ≥ 0.7 against `call_rate`). Report AUROC for `need` of TUL, calls and the probe side by side.
6. **Write the remaining modules.** `directions.py` (including `d_SE^dm` and `d_act^dm` per PLAN.md M3; `d_act^dm` needs E2's TUL for the `current` mode), `geometry.py`, `steer.py`, `patch.py` (M5b) and `plots.py` can start in parallel; the E3 analysis is CPU-only once states and labels exist. E3 needs per-layer **pre-norm** residuals for `sparse_tool` and `current` prompts: When2Tool's `extract_features` saved `current` states from `output_hidden_states` (its last layer is post-norm), so re-extract with `hf.capture` in fp32.
7. Before E3, write the one-page check of E0–E2 against the pass criteria (PLAN.md working rules).

## User instructions and preferences

- Implement PLAN.md; parallel agents are welcome. Ask when uncertain.
- The user likes plain-language explanations: say what a number means and why it matters, not only the number.
- Get explicit approval before plan-level changes; the changes above were approved on 2026-11-14.

## Gotchas

- `pgrep -f <pattern>` / `pkill -f <pattern>` match your own shell when the pattern is in your command line. Use `ps -eo pid,etimes,args | grep … | grep -v grep`.
- vLLM prints "Triton module has been replaced with a placeholder" at start-up; it is harmless (triton 3.2.0 is installed).
- When2Tool's `hard_no_tool` uses the **`no_tool`** prompt text and rejects calls; the comment in `extract_features.py` saying "current" is out of date.
- In `no_reasoning` mode, When2Tool rejects a tool call that comes after text ("reasoning is not allowed"), so late calls count differently in the harness than in our sampled call rate.
- `answer_key` must keep big integers exact (hash and matrix answers overflow `float`).
