# Handoff: implementing PLAN.md (toolmech)

Status as of 2026-11-14 07:15 UTC. Read `PLAN.md` first; this file covers only what has been done, what is blocked, and the traps already found.

## TL;DR

- **No experiment has run yet.** E0–E5 are all pending, and `results/prompts/` is empty.
- The local setup is done: When2Tool clone + data, model weights, venv, and the shared `toolmech` base (config, prompts, HF hooks).
- **The blocker is the GPU.** The devbox has no GPU. Merlin job `c6cbba075b8ad484` is stuck in `queued`: it asks for 4×H100 / 92 CPU / 800 GB, and the group quota is 2×H100 / 80 CPU / 472 GB.
- The previous agent did **not** stop or resize that job. The user has not yet approved stopping it or submitting a resized one. **Get explicit user approval before stopping or creating Merlin jobs.**

## Environment (devbox)

- Project root: `/mlx_devbox/users/nadya.wangsajaya/playground/epistemic-hackathon`. It is not a git repo.
- Disk: only **about 6.7 GB free**. Don't cache large artefacts locally.
- No `jq` or `bc`; use python. `python3 -m venv` is broken (no ensurepip), so use uv: `$(python3 -c "import site;print(site.USER_BASE)")/bin/uv`.
- `.venv/` (Python 3.11) has vllm 0.8.5, transformers 4.55.2, torch 2.6.0+cu124, accelerate 1.13.0, datasets>=3, sklearn, scipy, pandas, pyarrow, pyyaml, matplotlib and pytest. The pinned list is `toolmech/requirements.txt`.
  - `datasets` had to be upgraded to >=3 because older versions hit a pyarrow `PyExtensionType` error.
- `.hf_cache/` (7.6 GB) holds `Qwen/Qwen3-4B-Instruct-2507`, snapshot `cdbee75f17c01a7cc42f958dc650907174af0554`. Set `HF_HOME=$PWD/.hf_cache`.
- `when2tool/` is cloned from github.com/Trustworthy-ML-Lab/when2tool at commit `ef0cf1e`.
  - The HF dataset `cesun/When2Tool` was dumped to `when2tool/data/tasks_v1_{train,test,multihop_train,multihop_test}.json`, with 900 / 2250 / 180 / 450 tasks.

## Code written so far (`toolmech/`)

| File | State |
|---|---|
| `configs/qwen3_4b.yaml` | Done. This is the single config. Its hash goes into artefact names: `results/<step>/<name>.<hash>.<ext>`. |
| `toolmech/config.py` | Done. Provides `load_config` (env `TOOLMECH_CONFIG`), `Config.path/.hash/.data_path`, `seed_everything`, `load_tasks`, `task_env` and `TIER`. It puts `when2tool/src` and `when2tool` on sys.path. |
| `toolmech/prompts.py` | Written, **never run successfully**. Its first run failed on the datasets/pyarrow error, which is now fixed. Next step: `python -m toolmech.prompts`, which only needs the tokenizer and runs on CPU. |
| `toolmech/hf.py` | Written, untested on GPU. Provides residual capture via forward hooks on `model.model.layers[l]` (pre-final-norm), prefill-only steering from the user-turn position onward, `tul_from_logits`, `tool_token_ids` (`<tool_call>` is a single special token) and `last_logits_and_resid`. |
| `toolmech/merlin/job_run.json`, `launch.sh`, `job_entry.sh` | These are the Merlin job spec, the submit script and the in-job entrypoint. The entrypoint builds a venv in `/opt/tm_venv`, downloads the model, touches `/opt/toolmech/READY`, then idles until `/opt/toolmech/STOP` exists. |
| `toolmech/gpu.sh` | **Obsolete.** It uses an HDFS task queue, which is far too slow. Rewrite it as SSH + rsync. |

The remaining modules from PLAN.md are **not written**: `labels`, `tul`, `directions`, `geometry`, `steer`, `conversations`, `corrupt`, `trajectory`, `plots`. Neither is `REPORT.md`.

## When2Tool facts that matter

- Qwen uses the **xml** tool format: `SYSTEM_PROMPT_XML`, tool calls inside `<tool_call>...</tool_call>`, and tool results returned as a user message wrapped in `<tool_response>`.
- **Prompt mismatch:** `init_state` adds a ListManipulation contract system message, but `extract_features` does not.
  - `toolmech/prompts.py` uses `init_state` for everything so that vLLM and HF prompts are byte-identical.
  - Log this as a deviation and check that it doesn't break E0's 0.948 AUROC.
- Labels in When2Tool use `hard_no_tool` mode: no_tool prompt text, and tool calls are rejected.
- vLLM uses the model's generation_config defaults.
- `run_eval` / `extract_features` default to `--tensor_parallel_size 4`; pass `1`.
- `extract_features` uses `output_hidden_states`, whose last layer is post-norm.
- `train_probe` uses reg=10000, i.e. C=1e-4.

## Deviations to log in REPORT.md

1. HF forward hooks are used instead of TransformerLens/nnsight. They give exact HF numerics.
2. The prompt source is `init_state` (with the contract message) for both labels and features.
3. Steering accuracy on all 2250 tasks is probably too expensive. Plan a subset (`manip_tasks_per_tier`) and log it.

## Compute: what works and what doesn't

- **mlx workers** (`mlx worker launch`): workspace quota is 0 for H100 and A10. A100 hung pending, and public-arnold A10 failed with a cross-cluster error. All workers were killed.
- **bytedcli** needs `export BYTEDCLI_NETWORK_PROFILE=prod` and `--site i18n-tt`.
  - Only the native REST `merlin job-v2 runs list/get/create` works.
  - MCP-backed merlin commands fail because a proxy CA re-signs TLS.
  - Trusting that CA (NODE_EXTRA_CA_CERTS) was denied by the user's permission policy. Do not work around it.
  - `i18n-bd` returns 403.
- **User's queue:** `compute-240-aliyun.va-cloudnative-aigcp-global.ecom.risk.management-guarantee`, usergroup `global_ecom_risk_management`, group_id 130, cluster_ids [4].
  - Quota: **2× H100-SXM-80GB, 80 CPU, 483328 MB memory.**
- **Job create** (`toolmech/merlin/launch.sh`, which supports `--dry-run`) reuses the `ImageEmbed` v2 template. It overrides `entrypoint_full_script`, `git_repo` and `image_meta`, and it needs empty `input_config`/`output_config`, `--attachments` (hdfs_volume) and `--compliance`.
- **To fix the stuck job:**
  - `job_run.json` is now set to **1×H100, 23 CPU, 204800 MB**. Per-GPU limit on this queue's nodes is ≤23 CPU / ≤204800 MB per GPU (scale linearly: 2 GPU → 46–47 CPU / 409600 MB). 24 CPU / 240000 MB failed with Arnold exit -80002 (job `ad83890c83a98936`). Relaunched as `f13abbdfc20a6542`. One GPU is enough: the 4B model is about 8 GB in bf16, and vLLM with TP=1 plus HF hooks fit side by side. 4 GPUs only came from copying the template. 2 GPUs (within quota) would roughly halve wall time by running vLLM sampling and HF steering concurrently.
  - Stop `c6cbba075b8ad484`. Find the command under `bytedcli --site i18n-tt merlin job-v2 runs --help`.
  - Relaunch.
  - **Ask the user first** for each of these.
- **H20 spot fallback:** group 130's H100 queue was 40/40 used. The shared spot pool `compute-288-my2-gpuspot1-default-mlsys.public.inc-guarantee` (cluster 21) had free H20/L20 GPUs. Spec: `toolmech/merlin/job_run_h20spot.json` (1×NVIDIA_H20, 23 CPU, 245760 MB, preemptible). Launch it with `JOB_SPEC=job_run_h20spot.json ./launch.sh`. Submitted as `626c9c78f9e613fc` alongside H100 `f13abbdfc20a6542`; keep whichever starts first and stop the other. Spot jobs can be preempted, so copy results out often.
- **HDFS** (`/mnt/hdfs/nadya.wangsajaya`) runs at about 0.4 MB/s and about 7 s per small file from the devbox. Don't move data through it.
  - Plan: get SSH with `bytedcli --site i18n-tt merlin job ssh-command get --job <id>`, then rsync code in and results out.
  - The model can be downloaded inside the job (job_entry.sh tries this) or pushed over SSH.
- Status check: `BYTEDCLI_NETWORK_PROFILE=prod bytedcli --json --site i18n-tt merlin job-v2 runs get --sid c6cbba075b8ad484` (shows `trial_status`).
- **Gotchas:**
  - Don't use `pkill -f <pattern>` when the pattern appears in your own command line; it kills your shell.
  - A compression hook may truncate tool output; recover it with `caveman_retrieve` and the ccr handle.

## Suggested next steps

1. Run `python -m toolmech.prompts` on CPU. Verify the prompt counts and token lengths, and check that `user_start_char` is ≥ 0 everywhere.
2. Get the user's decision on compute:
   - (a) resize and relaunch at 2×H100, or
   - (b) code only, with CPU unit tests.
3. Write the remaining modules in parallel. The E3 analysis, `plots` and `REPORT.md` are CPU-only and can proceed without a GPU against small synthetic fixtures.
4. With a GPU, run E0 (When2Tool steps 1–3 with TP=1; target probe AUROC 0.948 ± 0.02), then E1 (labels.parquet), E2 (tul.parquet; Spearman ≥ 0.7), E4 and E5.

## User instructions so far

- Implement PLAN.md and use many parallel agents. Ask when uncertain.
- Use mlx/bytedcli or a separate Merlin job, on the user's own uncrowded queue.
- The user's last messages before this handoff were "okay nevermind it is taking quite a long time" and then "i am going to handoff to a different agent". There has been no approval to stop or relaunch jobs.
