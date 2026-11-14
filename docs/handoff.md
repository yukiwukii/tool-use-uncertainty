# Handoff: implementing PLAN.md (toolmech)

Status as of 2026-11-14 14:00 UTC. Read `PLAN.md` first, especially its **revision log** (6 items): the prompt mode, how tool use is measured, the steering directions and E4 all changed after E0. This file covers the environment, what has run and its results, what is running, the code, and what to do next.

## TL;DR

- **Machine:** local box at `/tool-use-uncertainty`, one **NVIDIA A40 (48 GB)**, 96 CPUs, 503 GB RAM. No Merlin, devbox or HDFS (the user handles any leftover Merlin jobs from the earlier handoff).
- **Done:** E0 (probe AUROC 0.9495, paper 0.948) and E1 (label table, passed). E3 code is written and tested.
- **Running:** E2 (`python -m toolmech.tul`, PID 20277): saves the last-prompt-token hidden states for E3 and the TUL table. 3 of 4 chunks were done at 13:57; expected finish ~14:20 UTC.
- **Next:** E2 checks, then run E3 (`toolmech.directions`, then `toolmech.geometry`).
- **Decisions made with the user** (all in PLAN.md):
  - `sparse_tool` is the primary prompt mode and `current` the secondary.
  - **Tool use is measured from the model's actual replies** (call rate over 20 samples, greedy decision), not from TUL. TUL is kept only as the activation-patching readout.
  - Steering uses stratified difference-in-means directions (`d_SE^dm`, `d_act^dm`); activation patching is a second E4 test.
  - fp32 for all HF passes.
- Git: 4 commits on `main` since the initial one, latest `7d2cfce`. Working tree clean apart from an unrelated `nvim-linux-x86_64.tar.gz` (not ours; leave it out of commits). Commit only when the user asks.

## Environment

- Project root `/tool-use-uncertainty`. Untracked by design: `when2tool/`, `.venv/`, `.hf_cache/`, `results/` (in `.gitignore`).
- `when2tool/`: github.com/Trustworthy-ML-Lab/when2tool at `ef0cf1e`. Data dumped with When2Tool's own HF loader to `when2tool/data/tasks_v1_{train,test,multihop_train,multihop_test}.json` (900 / 2250 / 180 / 450).
- `.venv/`: Python 3.11 via `uv` from `toolmech/requirements.txt` (torch 2.6.0+cu124, vllm 0.8.5, transformers 4.55.2; `setuptools` is needed for vLLM to start).
- `.hf_cache/`: `Qwen/Qwen3-4B-Instruct-2507`, snapshot `cdbee75f…`; `refs/main` was written by hand to point at it (offline loads by name fail without it).
- **Every command needs:**
  ```bash
  export PYTHONPATH=/tool-use-uncertainty HF_HOME=/tool-use-uncertainty/.hf_cache HF_HUB_OFFLINE=1 HF_HUB_ENABLE_HF_TRANSFER=0
  ```
  The machine sets `HF_HUB_ENABLE_HF_TRANSFER=1` globally without `hf_transfer` installed; tokenizer loads fail without the override.
- Tests: `.venv/bin/python -m pytest tests -q -p no:cacheprovider` (17 CPU + 7 GPU). Run the GPU tests only when the GPU is free: alongside vLLM (≈21 GB) or E2 they run out of memory.
- Git identity is set repo-locally to the author of the initial commit.
- Config hash is now **`9c3c9f8d`**. Artefacts are `results/<step>/<name>.9c3c9f8d.<ext>`. The hash covers the whole config: never edit `configs/qwen3_4b.yaml` while a stage is running, or its outputs land under a stale hash. `results/e1/HASH_CARRYOVER.txt` records the one carry-over (the no-tool samples were copied from hash `f8759166`, because the changed keys do not affect that stage).

## Results so far

### E0 (passed)
- All-layer probe for `need`, test AUROC **0.9495** (paper 0.948); within tier 0.92–0.94. Weakest environments: RegexMatch 0.75, CodeExecutor 0.81.
- Baseline eval (`current`, no reasoning, 3 runs): accuracy **0.888**. Without tools the model is correct on 47.6% of test tasks; 52% need a tool.
- Files: `when2tool/probe_data/qwen3-4b-instruct/`, `when2tool/outputs/qwen3-4b-instruct/current.json`, log `results/e0/e0.log`.

### E1 labels (passed: no missing labels; SE > 0 for 54% of medium tasks)
`results/e1/labels.9c3c9f8d.parquet`, one row per task (3,150: train + test). Columns:
- From 10 no-tool samples per task (T = 1, top_p = 1, top_k = −1): `fail_rate`, `SE`, `n_clusters`, `SE_abstain` ("I don't know" answers as one cluster), `abstain_rate_notool`.
- From 20 sampled + 1 greedy reply per task and mode: `act_<mode>` (greedy), `call_rate_<mode>`, `call_rate_first_<mode>`, `late_share_<mode>`, `abstain_rate_<mode>` (a **lower bound**: this run kept only the first 200 characters of each reply). Plain `act`, `call_rate` etc. are the primary mode (`sparse_tool`).
- `need` and `tier` from When2Tool/E0; `act_w2t` = E0's mean tool use over 3 runs (`current`, Qwen default sampling).
- Per-sample files: `results/e1/notool_samples_{split}.*.parquet`, `call_samples_{split}_{mode}.*.parquet`.

Findings (test split):

| | `sparse_tool` | `current` |
|---|---|---|
| Call rate | 0.17 | 0.91 |
| Tasks with mixed calls across samples | 30% | 6% |
| Calls that start after some text (sample level) | ~65% | ~1% |
| Call rate when a tool is needed / not needed | 0.27 / 0.06 | 0.93 / 0.88 |
| AUROC of calls for `need` (probe: 0.95) | **0.69** | **0.52** |

- **Uncertainty explains calls only weakly so far.** Spearman of SE with the call rate is 0.27 in `sparse_tool` (fail_rate: 0.39). Within environment × tier cells the median is **0.06** (positive in 61% of 33 cells). Low power: 50 tasks per cell, 63% of tasks never call.
- **SE is confounded:** Spearman 0.67 with fail_rate; mean SE by environment ranges from about 0.1 to 2.1.
- **Fictional-entity tasks** (HistoricalYear-hard, Retriever-hard) fail without tools 100% of the time with high SE, yet behave differently. HistoricalYear-hard never calls and answers "unknown" (≥ 58–64%) in both modes. Retriever-hard calls 94% in `current` but only 3% in `sparse_tool`, mostly guessing. So the same high-uncertainty state leads to abstaining, calling or guessing.
- Abstention detector (`labels.is_abstain`): 886 flagged no-tool answers, none graded correct.

### E2 (running; first-chunk checks)
- Hidden states: `results/e2/resid_<split>_<mode>.*.pt` with keys `ids`, `layers`, `resid` [n, 36, 2560] fp32, pre-norm, last prompt token. Verified against a direct forward pass (tolerance 1e-3).
- On train `sparse_tool`, the HF greedy decision agreed with vLLM's on 92.4% of tasks. Of the 68 disagreements, 86% are calls that come after token 64 (median about character 427 of vLLM's reply), beyond TUL's 64-token window. Only **9/900 (1.0%)** are true same-window disagreements, and those are borderline (median |TUL| 1.7). So the model behaves the same in both engines; the short window misses late calls.
- TUL_N vs the call rate on that chunk: Spearman 0.686 (N = 1) to 0.698 (N = 64). This is why tool use is now measured from replies.

## What is running

- E2: `nohup sh -c '… .venv/bin/python -m toolmech.tul --splits train,test …' > results/e2.log`, python PID 20277, started 12:47. About 13 min per 900 tasks. At 13:57 the last chunk (`test`/`current`) was running.
- Check it: `ls results/e2/; grep -E "^exit=|Traceback" results/e2.log`. Done when the log ends with `exit=0 elapsed=…` and `resid_test_current.*.pt` exists.

## Code (`toolmech/`)

| File | State |
|---|---|
| `config.py` | Done. |
| `prompts.py` | Done. Modes `current`, `sparse_tool`, `force_tool`, `no_tool`; caches in `results/prompts/`. |
| `hf.py` | Done, GPU-tested. fp32 default; residual capture and prefill-only steering hooks. |
| `labels.py` | Done, run (E1). Stages `notool` / `calls` / `merge`; `--limit N` writes to `results/e1_smoke/`. |
| `tul.py` | Done, running (E2). Windowed TUL plus residual capture from the same `generate` prefill. |
| `directions.py` | **Done, tested on synthetic data, not yet run.** Per layer, fit on train and decoded on test: difference in means (`need`, `act`, `diff`), ridge (`SE`, `fail_rate`, `call_rate`, `SE|fail`, `SE|diff`, `SE_abstain`), logistic (decoding), `d_SE^dm` (within env × tier, majority-correct tasks, SE thirds) and `d_act^dm` (`sparse_tool` only, greedy `act`). Scopes: all, within tier, within env × tier. Baselines: residual norm and surface features (token count, digit count, operand size, environment). |
| `geometry.py` | **Done, tested, not yet run.** Cosines at every layer; at the 3 best SE layers, 100 bootstrap refits, a shuffled-label null and Ledoit-Wolf whitened cosines; behaviour tables (SE vs call rate within tier and cell; the difficulty × uncertainty quadrants with call and abstain rates). |
| `steer.py`, `patch.py`, `conversations.py`, `corrupt.py`, `trajectory.py`, `plots.py`, `REPORT.md` | Not written. |
| `merlin/`, `gpu.sh` | Obsolete on this machine. |

## Next steps

1. **When E2 finishes:**
   - Rerun the HF-vs-vLLM agreement check on all four chunks: split disagreements into same-window ones and late calls, as above; the PLAN pass mark is ≥ 98% same-window agreement.
   - Run the GPU tests once more, then commit nothing new unless the code changed.
2. **E3:**
   - `python -m toolmech.directions` (both modes; CPU; time it), then `python -m toolmech.geometry` (bootstrap and null at the best layers; time it).
   - Before E3's write-up, do the one-page E0–E2 check against the pass criteria (PLAN.md working rules).
   - Report in plain language:
     - decoding curves for SE, need, act and diff, overall and within tier and cell, against the norm and surface baselines
     - the key cosines (`d_SE^dm` vs `d_act^dm`, `d_SE` vs `d_need` / `d_fail`, `d_act^dm` vs `d_diff`) with CIs and the null
     - the quadrant table
3. **Optional extensions the user was offered but did not ask for** (do not start without asking):
   - Kossen-style semantic entropy probe (binarised SE, logistic, AUROC) next to our ridge probe.
   - "Prefill" analyses: run HF over the prompt plus E1's actual reply to read states at the moment of a late call, or to score vLLM's tokens under HF. This needs full reply texts: E1 kept only 200 characters, so the replies must be regenerated.
   - E3b: per-run decision-point direction from tasks with mixed calls (same prompt, one calling run vs one non-calling run). The user said the current approach is fine ("what we have is okay").
4. **E4** (after E3): write `steer.py` (readouts = call rate and greedy decision from real replies; manipulation check on SE first; effects read against measured ΔSE) and `patch.py` (M5b; the readout is TUL read along a fixed reply so late calls are not missed; see PLAN.md M1 and M5b). Time a small run first. Running the full sweep only in `sparse_tool`, with a smaller `current` confirmation, halves the cost.
5. E5, then `REPORT.md` with the deviations below.

## Deviations to log in REPORT.md

1. HF forward hooks instead of TransformerLens/nnsight.
2. Prompt source is `init_state` (with the ListManipulation contract) for both vLLM and HF.
3. fp32 for all HF passes (bf16 moved TUL by up to 0.9 nats near the decision boundary).
4. E0 step 1 reduced to `current`/no-reasoning, 3 runs.
5. Sampling at T = 1, top_p = 1, top_k = −1, no repetition penalty (When2Tool otherwise inherits Qwen's top_k = 20 / top_p = 0.8).
6. `act` is the greedy first reply on the cached prompts, per mode; E0's mean is kept as `act_w2t`.
7. Primary mode `sparse_tool`, secondary `current`.
8. Tool use is measured from replies; TUL only for patching. `d_act^dm` only in `sparse_tool`.
9. Stratified difference-in-means steering directions; activation patching added to E4.
10. Steering accuracy on a per-tier subset first.
11. `refs/main` in the HF cache written by hand; the E1 no-tool samples were carried over from config hash `f8759166` to `9c3c9f8d` (see `HASH_CARRYOVER.txt`).
12. Abstention labels added (`SE_abstain`, `abstain_rate_*`); the E1 call-sample abstain rates are lower bounds.

## Working with this user

- They want plain-language explanations: say what a number means and why it matters; use small worked examples. They found AUROC-heavy and jargon-heavy answers confusing.
- They question design choices closely (why TUL, whether E1 and E2/E3 measure the same behaviour, what a "snapshot" is). Answer directly, admit uncertainty, and check with data where possible.
- Ask before plan-level changes or new experiments; the changes above were approved on 2026-11-14.

## Gotchas

- `pgrep -f` / `pkill -f` match your own shell when the pattern is in your command line; use `ps -eo pid,etimes,args | grep … | grep -v grep`.
- vLLM's "Triton module has been replaced with a placeholder" at start-up is harmless.
- When2Tool's `hard_no_tool` uses the **`no_tool`** prompt text and rejects calls (the comment in `extract_features.py` is wrong).
- In `no_reasoning` mode When2Tool rejects a tool call that follows text, so late calls count differently in the harness than in our sampled call rate.
- `answer_key` must keep big integers exact and handle sets; `is_abstain` must not match names ending in "na".
- The hidden state at the last prompt token is identical across all sampled runs of a task (causal attention; sampling starts after it). Per-run internals only differ inside the reply.
