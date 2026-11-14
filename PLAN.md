# Tool-Use Decisions in LLMs: Directions, Uncertainty and Circuits — Implementation Plan

Nov 14, 2026 · revised Nov 14, 2026 after E0 (see revision log)

## Revision log

Changes agreed with the user after E0 and the E1 smoke test; each is marked "(rev. Nov 14)" where it applies.

1. **Prompt mode.** In `current` mode Qwen3-4B calls a tool on 91% of test tasks, and its call decisions predict `need` at AUROC 0.52 (the probe: 0.95). `sparse_tool` is now the **primary** mode (calls on ~17% of tasks, ~33% of tasks mixed across samples); `current` is the **secondary** mode. Comparing the two treats the instruction as a decision threshold (Kumaran et al. 2026).
2. **Tool use is measured from outputs.** The behavioural measure everywhere is the sampled call rate (20 replies, T = 1) and the greedy call decision, read from the model's actual replies. TUL (M1) is kept only as the cheap continuous readout for activation patching (M5b), where sampling replies for every intervention is not feasible. Reason: in `sparse_tool` the model often writes 100–250 tokens before calling, so a first-token or 64-token TUL misses many calls (on the first E2 chunk, Spearman with the call rate was 0.69, and 86% of HF–vLLM disagreements were calls after token 64).
3. **Steering direction.** The direction steered in E4 is a **stratified difference in means** of SE within environment × tier cells, matched on fail rate (`d_SE^dm`, M3), not the ridge probe. The ridge probe stays the decoding readout (E3, E5).
4. **Activation patching** between matched high- and low-SE tasks is a second causal test in E4 (M5b).
5. **Numerics.** All HF passes run in fp32: in bf16, TUL near the decision boundary moved by up to 0.9 nats with batch composition.
6. **Tool-use direction.** `d_act` is also built as a stratified difference in means within environment × tier cells (`d_act^dm`, M3), so that it is compared with `d_SE^dm` on equal terms. It is built in `sparse_tool` only, from real calls; in `current` 91% of tasks call, so there is too little contrast within cells.

## Summary

The question: is a tool-trained LLM's decision to call a tool driven by its own uncertainty, and does calling the tool actually resolve that uncertainty? We answer it in three steps: compare an uncertainty direction with tool directions (geometry), steer along uncertainty and watch tool calls (causation), and track uncertainty turn by turn through multi-turn tool-use conversations (dynamics).

| Quantity                  | What it captures                                                                     | How it is measured                                                               |
| ------------------------- | ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| **Uncertainty** (`d_SE`)  | How much the model's answers would vary in meaning                                   | Ridge probe on semantic entropy (SE) for decoding; stratified difference in means (`d_SE^dm`) for steering (rev. Nov 14) |
| **Difficulty** (`d_diff`) | How hard the task looks from its surface (the benchmark's easy / medium / hard tier) | Probe; label = easy (0) vs hard (1), medium held out, as in When2Tool Appendix J |
| **Need** (`d_need`)       | Whether the task actually requires a tool                                            | When2Tool probe; label = model fails without tools                               |
| **Act** (`d_act`)         | Whether the model will call a tool                                                   | Probe; label = model actually called a tool                                      |
| **Call rate** (and TUL)   | How often the model calls (and, for patching only, how strongly it leans toward calling) | Sampled call rate and greedy decision from real replies; log-odds of the tool-call token for patching (rev. Nov 14) |

The implementing agent should produce:

1. A label table for every When2Tool task (need, SE, fail rate, act, call rate).
2. Per-layer directions `d_SE`, `d_need`, `d_act` plus the difficulty direction `d_diff`, with their pairwise geometry.
3. Steering curves along ±`d_SE^dm`, with a check that steering really changes uncertainty, plus `d_need`, `d_act`, `d_diff` and random directions for comparison, read against the measured change in SE; and activation patching between matched high- and low-SE tasks (rev. Nov 14).
4. Uncertainty trajectories across multi-turn conversations: before the call, at the call and after each tool result, single-hop and multi-hop, with correct and corrupted tool outputs.
5. A results write-up. Circuit analysis (EAP-IG, attribution graphs) is out of scope for now; see the appendix "Later: circuit analysis" at the end of this file.

**Hypotheses**

- **H1 (geometry):** `d_SE` is closely aligned with `d_need`, partly by construction, but much less with `d_act`.
- **H2 (causation):** steering along +`d_SE^dm` raises the call rate; −`d_SE^dm` cuts unnecessary calls while hard-task accuracy holds. Per unit of SE change, `d_SE^dm` moves calls more than control directions do, and patching the decision-point state from a high-SE task into a matched low-SE task raises the model's probability of calling (rev. Nov 14).
- **H3 (dynamics):** uncertainty drops after a correct tool result. If it drops just as much after a corrupted result, the model's post-tool confidence is not calibrated to the evidence.
- **H4 (difficulty vs uncertainty):** within a difficulty tier, semantic entropy still predicts tool calls. If instead `d_act` aligns with `d_diff` and not with `d_SE`, tool calls follow how hard a prompt looks, not the model's own uncertainty.

## Background

Three lines of work share one finding: the model represents X internally but acts on Y. This project applies all three methods to the tool-call decision.

| Source | Method we borrow | Finding we build on |
| --- | --- | --- |
| Zhao et al., *Wired for Overconfidence* (COLM 2026) | A logit-difference proxy (TSLD) for a discrete behaviour; clean/corrupted contrast design; ablation and steering | Stated confidence is driven by a compact mid-to-late-layer circuit that is separate from the factual-retrieval circuit |
| Sun et al., *LLM Agents Already Know When to Call Tools* (When2Tool, NeurIPS 2026) | Last-token hidden states, label = fails without tools, L2 logistic regression; 18 environments with easy/medium/hard tiers | Tool necessity is decodable at AUROC 0.89–0.96, yet the first-token tool likelihood predicts necessity worse than the probe |
| Kossen et al., *Semantic Entropy Probes* (2024) | Probe trained to predict semantic entropy from hidden states before generation | Semantic uncertainty is linearly decodable and generalises out of distribution better than accuracy probes |
| Stolfo et al., *Confidence Regulation Neurons* (NeurIPS 2024) | None directly; it is a caution | Part of confidence is carried by the residual norm and an unembedding null space via LayerNorm, not by a direction |

What is new here: When2Tool shows the necessity signal is decodable, but never tests it causally and never relates it to uncertainty. Semantic entropy probes measure uncertainty but have not been used in agent loops. This project links the two, tests causation by steering, and follows uncertainty through multi-turn tool use.

## Model and harness

Use **Qwen/Qwen3-4B-Instruct-2507** and fork the **When2Tool** repo as the agent harness. When2Tool already runs this exact model (key `qwen3-4b-instruct`), provides 18 tool environments with real tool execution, labels necessity, and extracts last-token hidden states. Everything else plugs into its outputs.

| Repo | Role in this project | What to reuse | What to add or change |
| --- | --- | --- | --- |
| [When2Tool](https://github.com/Trustworthy-ML-Lab/when2tool) (+ [HF dataset](https://huggingface.co/datasets/cesun/When2Tool)) | **Primary harness**: tasks, tool environments, agent loop, necessity labels, hidden-state extraction | `run_eval.py` (5 prompt modes), `extract_features.py` (tensor `n_tasks × n_layers × d`), `train_probe.py`, `AgentModel` (vLLM and HF backends) | Save per-layer states separately; add behaviour labels, TUL and semantic-entropy labels; add single-layer and diff-in-means directions |
| [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) | Hooked model for reading states and steering | `HookedTransformer` residual-stream hooks | Verify that Qwen3-4B-Instruct-2507 loads and matches HF logits (see pitfalls) |
| [nnsight](https://nnsight.net) | Fallback backend if TransformerLens cannot load the model | Hooks on HF modules | State-reading and steering hooks |
| [semantic_uncertainty](https://github.com/jlko/semantic_uncertainty) (Farquhar/Kossen) | Reference for semantic-entropy computation and SEP training | `compute_uncertainty_measures.py` logic | Replace NLI clustering with exact-match or numeric-equality clustering for short answers |

Why this model: it has no thinking mode, so the tool decision is made early in the first assistant turn and TUL can be read directly (at the first token in `current` mode; over a short window in `sparse_tool`, see M1). It is also small enough for steering sweeps and multi-turn sampling on one 80 GB GPU, and the When2Tool paper reports a probe AUROC of 0.948 for it.

One harness, two passes: run all generation (baseline evaluation, no-tool sampling for labels and semantic entropy) through **vLLM** for speed. Run everything that needs hidden states or hooks (features, TUL, steering, multi-turn state reads) through **HF or TransformerLens** with the exact same chat template and prompt strings.

**Prompt modes (rev. Nov 14).** When2Tool appends one policy sentence to every task; the modes differ only in that sentence. `current`: "You can choose to use a tool or not." `sparse_tool`: "Tool calls are expensive in this mode; default to zero tool calls…". `force_tool` and `no_tool` require or forbid tools. M1, M2 (`act`, `call_rate`), M3 (`d_act`), M5 and E2–E4 are run in **`sparse_tool` (primary) and `current` (secondary)**. Treat the mode as a decision threshold: fit call ~ SE + mode (+ SE × mode) to tell a shift of the threshold from a change in how strongly calls depend on uncertainty (Kumaran et al. 2026, Phase 4). E5 uses `force_tool` and the default mode as before.

Multi-turn conversations come from the When2Tool harness itself: it executes real tools and returns their output as a tool message, and `run_multihop_pipeline.sh` runs 3-step tool-call chains. The agent needs one addition here: a hook that can **replace a tool's output** before it goes back to the model (for the corrupted-output condition in E5), and a way to **save hidden states at chosen positions** in each turn.

## Measurements

M1–M5 are taken at one position: the last prompt token, right before the first assistant token. M6 adds positions later in the conversation. All of them use the When2Tool prompts exactly as its harness builds them.

### M1. Tool-use log-odds (TUL): how much the model wants to call

```latex
\mathrm{TUL}(x) = \log p(t_{\mathrm{tool}} \mid x) - \log\big(1 - p(t_{\mathrm{tool}} \mid x)\big)
```

Here t_tool is the first token of a tool call. For Qwen3-Instruct with the native `tools=` chat template this is the special token `<tool_call>`. First check how When2Tool's harness asks for tool calls. If it uses its own JSON format, t_tool is that format's first token, and if that token is ambiguous (e.g. `{`), use the logsumexp over the set of tokens that begin a call.

```python
tool_ids = [tok.convert_tokens_to_ids("<tool_call>")]  # confirm against the harness
lp = model(ids).logits[0, -1].float().log_softmax(-1)
lp_tool = torch.logsumexp(lp[tool_ids], 0)
TUL = lp_tool - torch.log1p(-lp_tool.exp())
```

Validation: sample 20 generations per task at T = 1 and compute the empirical call rate. Spearman ρ between TUL and call rate must be at least 0.7. Also report the share of calls that begin after some text (e.g. "Let me check…"); if it is above 10%, TUL misses those calls and must be redefined over the first few positions.

**Windowed TUL (rev. Nov 14).** The late-call share is ~1% in `current` but ~58% in `sparse_tool`, so TUL is redefined along the greedy reply:

```latex
\mathrm{TUL}_N(x) = \operatorname{logit}\Big(1 - \prod_{t < N'} \big(1 - p_t(t_{\mathrm{tool}} \mid x, y_{<t})\big)\Big)
```

where y is the greedy continuation and N' = min(N, the step at which the greedy path calls a tool or ends). N = 1 is the first-token TUL above. Compute N ∈ {1, 8, 32, 64} from one `generate(output_logits=True)` pass (`toolmech/tul.py`) and use the smallest N whose Spearman ρ with the sampled call rate passes 0.7 in each mode. Calls on non-greedy paths are still missed; the sampled call rate stays the behavioural ground truth.

**Role of TUL (rev. Nov 14, second revision).** Tool use is measured from the model's replies (call rate, greedy decision) everywhere it is affordable. TUL is used only as the readout for activation patching (M5b), which needs one cheap continuous number per intervention; there, it is read at the decision point of a fixed reply (see M5b) so late calls are not missed.

### M2. Labels

| Label | Definition | Source |
| --- | --- | --- |
| `need` (binary) | 1 if the model fails the task when tools are disabled | When2Tool `extract_features.py` |
| `fail_rate` (continuous) | Share of k = 10 no-tool samples at T = 1 that are wrong | New: vLLM sampling |
| `SE` (continuous) | Semantic entropy over the same k samples | New: cluster answers by normalised exact match (numbers parsed and compared numerically), then the entropy of cluster frequencies |
| `act` (binary) | 1 if the greedy (T = 0) first reply calls a tool, per mode: `sparse_tool` (primary) and `current` (rev. Nov 14) | New: vLLM on the cached prompts; E0's `run_eval.py` majority kept as `act_w2t` |
| `call_rate` (continuous) | Share of 20 sampled first replies (T = 1, top_p = 1, top_k = −1) that call a tool, per mode | New |
| `tier` | easy / medium / hard | Dataset field |

Use the k no-tool samples for both `fail_rate` and `SE`, so the labels come from one sampling pass.

For the multi-turn analysis (M6), add one more label per turn: `SE_post`, the semantic entropy of k = 10 final answers sampled **after** the tool result is in the context, with the same clustering as `SE`.

### M3. Directions, per layer

For each layer l, compute two directions per label from last-token residual states H_l (taken before the final norm):

- **Difference in means** (used for geometry and steering): d = mean(H | y=1) − mean(H | y=0), normalised.
- **Logistic probe** (used for decoding AUROC): fit on standardised features, then map back with w_raw = w / σ before normalising.
- **Ridge regression** for continuous labels (`fail_rate`, `SE`, `call_rate`): d = coef / ‖coef‖.

Report AUROC (or Spearman ρ for continuous labels) on the When2Tool test split for every layer, plus the all-layer concatenated probe to reproduce the paper's number. Keep separate directions per layer; never take cosines between all-layer probes.

Two cautions for the comparison the project rests on:

- `d_need` is labelled by failure, so it is partly an "I will get this wrong" direction. Some alignment with `d_SE` is expected by construction. Also fit `d_fail` (ridge on `fail_rate`) and report the `d_SE`–`d_need` cosine next to the `d_SE`–`d_fail` cosine. Then fit `d_SE|fail` (SE with fail_rate regressed out) to isolate uncertainty that is not just predicted error.
- `d_act` is the actual tool-use direction. Its relation to `d_SE` is the main geometric result, because it says whether the urge to call tracks uncertainty. Compare the stratified versions `d_act^dm` and `d_SE^dm` as the headline (rev. Nov 14); the plain versions mix in task type.

**Steering direction for uncertainty (`d_SE^dm`, rev. Nov 14).** A plain high-vs-low-SE mean difference would mostly encode task type: in the E1 smoke test, mean SE ranged from 0.11 (CodeExecutor) to 1.75 (Decoding), and SE correlated with fail_rate at Spearman 0.65. Build the direction within matched cells instead:

1. Group tasks by environment × tier (train split).
2. Within each cell, keep tasks whose majority no-tool answer is correct, or match fail_rate within ±0.1, so uncertainty is separated from predicted error (as Kumaran et al. use correct trials only).
3. Within each cell, take the mean residual of the top-third SE tasks minus the mean of the bottom-third.
4. Average the per-cell differences, weighting by cell size, and normalise. Skip cells with fewer than 3 tasks in either group.

Report how many cells and tasks contribute, and the cosine of `d_SE^dm` with plain `d_SE`, `d_diff` and the environment-mean directions. Fit `d_SE^dm` per layer, like every other direction.

**Tool-use direction (`d_act^dm`, rev. Nov 14).** The plain difference in means `d_act` has the same task-type problem, and it is worse in `current`: the 9% of tasks that never call come almost entirely from Retriever, HistoricalYear and GameRule, so plain `d_act` would mostly encode "this is a knowledge-recall question". Build it within environment × tier cells, per prompt mode:

- **`sparse_tool`:** within each cell, mean residual of tasks with `act` = 1 minus tasks with `act` = 0 (greedy decision). Skip cells with fewer than 3 tasks on either side.
- **`current`:** not built (rev. Nov 14). Most cells have no tasks with `act` = 0, and building it from TUL would make it partly defined by a readout.
- Average the per-cell differences weighted by cell size, then normalise, as for `d_SE^dm`.

Report the contributing cells and tasks and the cosine of `d_act^dm` with plain `d_act`. `d_act^dm` is used in three places: the geometry comparison with `d_SE^dm` (H1, H4), as the **positive control** in E4 steering (pushing along it must move calls, otherwise the steering setup itself is broken), and as the comparison 1-D patch in M5b.

**Difficulty direction (`d_diff`).** Fit it per layer on the benchmark tier: easy = 0, hard = 1, medium held out (the difficulty-only probe in When2Tool Appendix J). Tier is set by the benchmark designers from surface properties (operand size, entity obscurity, procedure length) and is the same for every model. Uncertainty is specific to one model and varies a lot within a tier: Qwen3-4B-Instruct gets 74% of easy and 20% of hard tasks right without tools. Also fit `d_SE|diff`, the SE direction after regressing tier out of the SE labels, to isolate uncertainty that difficulty does not explain.

### M4. Geometry

For each pair of directions at the same layer, report the cosine with two references:

- **Stability:** 100 bootstrap refits of each direction give a confidence interval.
- **Null:** directions fit on shuffled labels give the chance level, which is about 1/√d.

Also report the cosine after whitening (multiply by C^(−1/2), where C is the activation covariance at that layer). This stops a few high-variance dimensions from dominating.

### M5. Steering

Add α · d to the residual stream at layer l, at every prompt position from the start of the user turn onward, with α = k · std(H_l · d) for k from −4 to 4 in steps of 1. Steer along ±`d_SE^dm` first (rev. Nov 14), then plain `d_SE`, `d_need`, `d_act`, `d_diff` and a random direction of equal norm for comparison.

Record two readouts per k, both from the model's actual replies: the greedy call decision and the call rate over 20 samples (rev. Nov 14: TUL is not a steering readout), in both prompt modes. Report them per difficulty tier, together with task accuracy.

**Manipulation check (required):** under the same steering, sample k = 10 no-tool answers and recompute SE. Along +`d_SE^dm`, SE must rise with k, and along −`d_SE^dm` it must fall. If SE does not move, a flat tool-use curve means nothing, because the steering did not change uncertainty.

**Read effects against the change in SE (rev. Nov 14).** A large push off the usual activations can raise SE and tool calls just by disrupting the model, so the manipulation check alone does not show that uncertainty was changed specifically. For every direction, plot Δcall-rate against the **measured** ΔSE, not against k, and compare slopes: the uncertainty account predicts that `d_SE^dm` moves calls more than the controls at the same ΔSE. Also report the share of steered no-tool answers that are unparseable (no `\boxed{}`) at each k, and drop strengths where it exceeds 10%.

### M5b. Activation patching between matched tasks (rev. Nov 14)

Every prompt in a mode ends with the same response-policy suffix and assistant header (~30–40 tokens). At those positions the tokens are identical across tasks and differ only in what they have read from the task, so states can be swapped between tasks without aligning lengths.

1. **Pairs.** Pair a high-SE task with a low-SE task from the same environment and tier, with matched fail rate where possible (top vs bottom SE third, as in `d_SE^dm`). Aim for at least 300 pairs. Optionally add constructed minimal pairs from When2Tool's generators (same template and digit count; for Retriever and HistoricalYear, well-known vs obscure entities, as in Ferrando et al. 2025).
2. **Metric.** Windowed TUL (continuous and linear in the logits, as Zhang & Nanda 2024 and Heimersheim & Nanda 2024 recommend). Normalised effect = (TUL_patched − TUL_recipient) / (TUL_donor − TUL_recipient). Patch both ways (high → low and low → high).
3. **Layer sweep.** At each layer, patch the residual at all suffix positions at once; then split by position at the best layers.
4. **Does the patch carry uncertainty?** At the best layer, patch and sample k = 10 no-tool answers; the recipient's SE should move toward the donor's.
5. **Is `d_SE^dm` the channel?** Patch only the component along `d_SE^dm` and compare it with the full-residual patch and a random 1-D patch. Makelov et al. (2023) show that 1-D subspace patches can act through pathways the model does not normally use, so keep both controls.
6. **Optional: DAS.** Learn the subspace that best transfers TUL under interchange (Boundless DAS, Wu et al. 2023) and compare it with `d_SE^dm`, `d_diff` and `d_act`.

### M6. Uncertainty across a multi-turn conversation

Read the uncertainty probe at fixed positions in each tool-use conversation:

| Position | Where in the conversation | Question |
| --- | --- | --- |
| **P1. Before the call** | Last token of the user turn, before the assistant replies | How uncertain is the model when it decides? |
| **P2. At the call** | Last token of the tool-call JSON, before the tool result | Does committing to a call change its state? |
| **P3. After the result** | Last token before the next assistant turn, with the tool message in context | Did the tool output resolve the uncertainty? |

In multi-hop tasks, P2 and P3 repeat for each call (P2₁, P3₁, P2₂, P3₂, …), giving a trajectory of uncertainty across turns.

A probe trained at P1 may not transfer to P3, because the context is different. Train a separate ridge probe at P3 on `SE_post`, and report both the P1 probe applied at P3 and the P3-specific probe. Use the ground-truth `SE_post` from sampling as the reference, so every probe readout can be checked against a measured value.

## Experiments

Run E0–E5 in order. Each step writes its outputs to disk, so a failed later step never forces earlier ones to rerun.

```
E0 Reproduce ──► E1 Labels ──► E3 Directions ──probe──► E5 Before/after
                    │               │
                    ▼               ▼ steer along d_SE
                 E2 TUL ──readout──► E4 Steering
```

E1 labels feed every later step. The E3 uncertainty probe is the readout tracked through conversations in E5; the direction steered in E4 is the stratified difference in means `d_SE^dm`, also built in E3 (rev. Nov 14).

| Step | Goal | Inputs | Outputs | Pass criterion |
| --- | --- | --- | --- | --- |
| **E0. Reproduce** | Confirm the harness works | When2Tool repo, `./run_pipeline.sh qwen3-4b-instruct --steps 1,2,3` | Baseline accuracy and calls; all-layer probe | Probe AUROC within 0.02 of the paper's 0.948 |
| **E1. Labels** | Build the full label table | E0 tasks; vLLM sampling | `labels.parquet` with need, fail_rate, SE, act, call_rate, tier per task | No missing labels; SE is non-zero for at least 30% of medium tasks |
| **E2. Hidden states and TUL** | Save the states E3 needs; check TUL is usable as the patching readout | HF model | `resid_<split>_<mode>.pt` (all layers, last prompt token); `tul.parquet`; HF-vs-vLLM agreement on the greedy decision, split into same-window disagreements and late calls; AUROC of calls for `need` next to the probe's (the When2Tool gap) | HF and vLLM agree on ≥ 98% of greedy decisions within the same window (rev. Nov 14) |
| **E3. Directions and geometry** | Is uncertainty the same direction as difficulty, need, or action? | Per-layer states, E1 labels | Per-layer decoding curves; cosine heatmap of `d_SE`, `d_SE\|fail`, `d_fail`, `d_need`, `d_act`, `d_diff`, `d_SE\|diff`, plus `d_SE^dm` and `d_act^dm` per mode (rev. Nov 14), with CIs and the null; difficulty × uncertainty quadrant table | SE probe Spearman ρ ≥ 0.5 on test (target); bootstrap CI width under 0.1 |
| **E4. Causal tests of uncertainty** | Does uncertainty cause tool calls? | `d_SE^dm` and comparison directions at the best 3 layers; matched high/low-SE pairs (rev. Nov 14) | Steering: SE (manipulation check), call rate, greedy decision and accuracy against k and against measured ΔSE, per tier and mode. Patching: per-layer normalised effect on the probability of calling, SE transfer, 1-D vs full patch | Manipulation check passes; random control moves the call rate by less than 0.05; patching effect reported with bootstrap CIs over pairs |
| **E5. Uncertainty before and after tool use** | Does the tool call resolve uncertainty, and is that justified? | Multi-turn conversations from the harness, single-hop and multi-hop; correct and corrupted tool outputs | Uncertainty at P1, P2, P3 (and per hop) by condition; probe readout vs sampled `SE_post` | P3-specific probe Spearman ρ ≥ 0.5 against `SE_post` |

### E3 details: difficulty versus uncertainty

1. **Geometry.** Report the cosines of `d_act` with `d_diff` and with `d_SE|diff`. If `d_act` is close to `d_diff` and far from `d_SE|diff`, the urge to call tracks surface difficulty, not uncertainty.
2. **Within-tier prediction.** Inside each tier, report Spearman ρ between SE and call rate. A clear correlation inside a tier means uncertainty adds information beyond difficulty. Repeat inside each environment × tier cell (rev. Nov 14): environment alone predicts `need` at AUROC 0.77, so a tier-level correlation can still be driven by task type.
3. **Off-diagonal cases.** Split tasks into four quadrants by tier (easy or hard) and SE (below or above the median within that tier). Report the call rate in each.

|  | Low SE | High SE |
| --- | --- | --- |
| **Easy** | Expected: few calls | Model is unsure on a task that looks easy |
| **Hard** | Confidently answers (often wrong) | Expected: many calls |

The two off-diagonal cells decide the question. If easy/high-SE tasks get few calls and hard/low-SE tasks get many, the model calls on appearance. If the pattern is reversed, it calls on uncertainty.

### E4 details: causal tests of uncertainty (rev. Nov 14)

**Steering (M5)**

1. Pick the 3 layers where the `d_SE` probe decodes best on the test split.
2. Run the manipulation check first, on 300 tasks (100 per tier), along ±`d_SE^dm`. Stop and report if SE does not move with k.
3. Run the M5 sweep along +`d_SE^dm` and −`d_SE^dm` in `sparse_tool` (primary) and `current`. Start with a stratified subset (`manip_tasks_per_tier` per tier) and extend to all 2,250 test tasks if time allows; log the subset as a deviation.
4. Repeat the sweep with plain `d_SE`, `d_need`, `d_act`, `d_diff` and a random direction. Compare them by Δcall-rate per unit of measured ΔSE, not per unit of k.
5. Headline numbers: change in call rate on easy tasks and change in accuracy on hard tasks at k = −2 and k = +2, and the Δcall-rate/ΔSE slope of `d_SE^dm` against the controls.

**Patching (M5b)**

6. Build at least 300 matched pairs from the test split (same environment and tier, high vs low SE, fail rate matched). Run the layer sweep in both directions in `sparse_tool`, then repeat at the best layers in `current`.
7. At the best layer, run the SE-transfer check and the 1-D `d_SE^dm` patch against the full and random patches.
8. Interpretation. Steering and patching agree → strong evidence that uncertainty drives calls. Patching moves TUL but the 1-D `d_SE^dm` patch does not → the decision reads uncertainty-related state that `d_SE^dm` does not capture. Neither moves calls beyond the controls → calls do not follow the model's uncertainty.

### E5 details: uncertainty across multi-turn tool use

1. **Conversations.** Run the When2Tool agent loop in `force_tool` mode so every task has at least one call, including easy ones. Also run the default mode, to compare voluntary calls with forced ones. For multi-hop, use the 450 multi-hop test tasks (3 calls each).
2. **Conditions.** For each conversation, run the tool turn twice: once with the real tool output, and once with a **corrupted** output of the same type and similar token length. Examples: one digit changed in a calculator result, the fact for a different entity from the retriever, a different value from the code runner.
3. **Readouts.** At P1, P2 and P3 (and each hop in multi-hop), save the residual state and read the `d_SE` probe score. At P1 and each P3, also sample k = 10 answers to measure the true SE and `SE_post`.
4. **Contrasts to report:**
    - ΔSE = SE(P3) − SE(P1), split by need (0 or 1) and by tool output (correct or corrupted).
    - Acceptance rate: how often the final answer adopts the corrupted value, and whether `SE_post` is higher in the conversations where it does.
    - Multi-hop: SE per hop, and whether the model makes another call when uncertainty stays high.
5. **Interpretation.** A drop in uncertainty after a correct result and a rise (or no drop) after a corrupted one means the model updates on evidence. A drop in both cases means a tool message alone lowers uncertainty, whatever it says.

## Validation and pitfalls

The biggest risk is a probe that detects surface features (digit count, entity rarity) rather than need. The controls below are required, not optional.

**Controls**

- **Within-tier signal:** report AUROC and Spearman ρ inside each tier separately. A direction that only separates easy from hard is a difficulty detector.
- **Surface-feature baseline:** fit the same probes on simple prompt features (token count, digit count, operand magnitude, environment one-hot). The hidden-state probe must beat this baseline by at least 0.05 AUROC.
- **Held-out environments:** train on 3 of 5 environments per category and test on the rest, as When2Tool Appendix D does.
- **Norm channel:** add ‖h‖ as a single feature and probe the raw residual stream (before RMSNorm). Stolfo et al. show confidence can live in the norm and an unembedding null space, so report whether ‖h‖ alone predicts SE.
- **Steering control:** random directions of equal norm, plus the same sweep at a layer where the probe is weak.

Additional controls for the uncertainty and multi-turn steps:

- **Circularity of need and uncertainty:** report `d_SE|fail` next to `d_SE` (M3). A conclusion about uncertainty must hold for `d_SE|fail`, not only for `d_SE`.
- **Corruption realism:** a corrupted output must keep the tool's format and a similar token length. Report the share of corrupted outputs the model flags as wrong, so trivially detectable corruptions can be excluded.
- **Forced versus voluntary calls:** report E5 separately for `force_tool` and default-mode conversations. Forced calls on easy tasks are the cleanest test of whether a tool message lowers uncertainty by itself.

**Pitfalls**

| Pitfall | Symptom | Fix |
| --- | --- | --- |
| Chat template mismatch between vLLM and HF | TUL disagrees with sampled behaviour | Build prompt strings once, save them, and feed the same strings to both backends |
| TransformerLens numerics differ from HF for Qwen3 | Logits differ by more than 1e-2 | Use TransformerLens 3 `TransformerBridge` with compatibility mode, or use the nnsight fallback |
| Saturated probe logits | Score is flat inside tiers | Use the ridge probes on fail_rate and SE for any claim about scale |
| P1 probe applied at P3 | Readouts drift for reasons unrelated to uncertainty | Use the P3-specific probe and check every readout against sampled `SE_post` |
| Tool message changes length | Position indices shift between correct and corrupted runs | Locate P2 and P3 by special tokens (`</tool_call>`, the start of the next assistant turn), never by fixed offsets |
| Steering leaks into later turns | Steered runs diverge in tool arguments, not just in the decision | In E4 steer only the first assistant decision; do not steer in E5 |

**Not in scope:** fine-tuning, circuit analysis (see the appendix), thinking-mode models, and models other than Qwen3-4B-Instruct-2507 (except one replication of E3–E4 in Llama-3.1-8B-Instruct if time allows).

## Deliverables and layout

The agent should fork When2Tool and add one package, `toolmech/`, without editing the upstream files beyond thin hooks for state saving and tool-output replacement.

```
when2tool/                 # upstream fork; small hooks in extract_features.py and the agent loop
toolmech/
  prompts.py               # builds and caches exact prompt strings (shared by vLLM and HF)
  labels.py                # E1: no-tool sampling, fail_rate, SE clustering, call_rate
  tul.py                   # M1 / E2
  directions.py            # M3: diff-in-means, logistic, ridge per layer; d_SE|fail, d_diff, d_SE|diff
  geometry.py              # M4 / E3: cosines, bootstrap, null, whitening
  steer.py                 # M5 / E4: sweeps, manipulation check
  patch.py                 # M5b / E4: matched pairs, suffix-position patching, 1-D vs full patches (rev. Nov 14)
  conversations.py         # E5: multi-turn runs, tool-output hook, position finder (P1, P2, P3)
  corrupt.py               # E5: per-environment corrupted tool outputs
  trajectory.py            # E5: probe readouts and SE_post per position and hop
  plots.py
configs/qwen3_4b.yaml      # model id, layers, k, thresholds, seeds
results/                   # parquet, .pt and figures, one folder per step
REPORT.md                  # final write-up
```

**Figures in `REPORT.md`**

1. Per-layer decoding curves (AUROC or ρ) for SE, fail_rate, need, act, call_rate and difficulty; plus call rate in the four difficulty × uncertainty quadrants.
2. Cosine heatmap of `d_SE`, `d_SE|fail`, `d_fail`, `d_need` and `d_act`, plus `d_diff` and `d_SE|diff`, at the best layer, with the null band.
3. Manipulation check: SE of sampled answers against steering strength k along ±`d_SE^dm`.
4. Steering curves: call rate and greedy decision against k and against measured ΔSE for `d_SE^dm`, `d_SE`, `d_need`, `d_act^dm`, `d_diff` and random, split by tier and prompt mode, with hard-task accuracy.
4b. Patching: normalised TUL effect by layer (both directions), SE transfer, and the 1-D `d_SE^dm` patch against the full and random patches (rev. Nov 14).
5. Uncertainty at P1 → P2 → P3, split by need and by correct or corrupted tool output.
6. Multi-hop trajectories: SE per hop, and the rate of another call when SE stays high.

**Compute estimate** (one 80 GB GPU; rough, for planning only)

| Step | Approximate cost |
| --- | --- |
| E0 baseline and features | A few GPU-hours with vLLM |
| E1 sampling (3,150 tasks × 30 samples) | 2–4 GPU-hours with vLLM |
| E2–E3 forward passes and probes | Under 2 GPU-hours |
| E4 steering sweeps (9 strengths × 6 directions × 2 modes, plus sampling for the manipulation check) | 8–16 GPU-hours (rev. Nov 14: doubled for two modes; use the per-tier subset first) |
| E4 patching (≈300 pairs × 36 layers × 2 directions, plus SE sampling at the best layer) | 2–4 GPU-hours |
| E5 conversations (single-hop and multi-hop, 2 conditions, SE sampling at each P3) | 6–12 GPU-hours |

These are estimates, not measurements; the agent should time E0 first and rescale.

**Working rules for the agent**

- Fix seeds and save every intermediate artefact with the config hash in its filename.
- Before E3, write a one-page check of E0–E2 numbers against the pass criteria and stop if any fail.
- Log every deviation from this plan in `REPORT.md`.

## References

- Zhao et al., *Wired for Overconfidence: A Mechanistic Perspective on Inflated Verbalized Confidence in LLMs*, COLM 2026.
- Sun et al., *LLM Agents Already Know When to Call Tools – Even Without Reasoning*, NeurIPS 2026, arXiv:2605.09252. Code: [when2tool](https://github.com/Trustworthy-ML-Lab/when2tool); data: [cesun/When2Tool](https://huggingface.co/datasets/cesun/When2Tool).
- Kossen et al., [Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs](https://arxiv.org/abs/2406.15927), 2024. Code base: [jlko/semantic_uncertainty](https://github.com/jlko/semantic_uncertainty).
- Stolfo et al., [Confidence Regulation Neurons in Language Models](https://arxiv.org/abs/2406.16254), NeurIPS 2024.
- Park, Choe and Veitch, [The Linear Representation Hypothesis and the Geometry of Large Language Models](https://arxiv.org/abs/2311.03658), ICML 2024.
- Marks and Tegmark, *The Geometry of Truth*, 2023 (difference-in-means directions for steering).
- Kumaran, Daw, Osindero, Veličković and Patraucean, [Causal Evidence that Language Models use Confidence to Drive Behavior](https://arxiv.org/abs/2603.22161), Nature Machine Intelligence 2026 (difference-in-means confidence steering on correct trials; confidence vs threshold decomposition).
- Zhang and Nanda, [Towards Best Practices of Activation Patching in Language Models](https://arxiv.org/abs/2309.16042), ICLR 2024.
- Heimersheim and Nanda, [How to use and interpret activation patching](https://arxiv.org/abs/2404.15255), 2024.
- Makelov, Lange and Nanda, [Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching](https://arxiv.org/abs/2311.17030), 2023.
- Wu, Geiger, Icard, Potts and Goodman, [Interpretability at Scale: Identifying Causal Mechanisms in Alpaca](https://arxiv.org/abs/2305.08809) (Boundless DAS), 2023.
- Ferrando, Obeso, Rajamanoharan and Nanda, [Do I Know This Entity? Knowledge Awareness and Hallucinations in Language Models](https://arxiv.org/abs/2411.14257), ICLR 2025.
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) (v3, `TransformerBridge`; Qwen3 adapter available).

---

# Appendix: Later: circuit analysis

This section holds everything about EAP-IG and attribution graphs. It is **not part of the current plan**: run it only after E3–E5 are done, and only if the steering results call for a mechanism.

## When to run it

- E4 shows that steering along `d_SE` changes tool calls, and you want to know which components carry uncertainty into the decision.
- Or E4 shows that it does not, and you want to find the components that produce unnecessary calls without reading uncertainty.

## Method: EAP-IG on over-calling

1. **Pairs.** Inside one When2Tool environment and tier, pair over-call tasks (need = 0, act = 1) with correct-direct tasks (need = 0, act = 0) whose prompts have the same token length. If none exists, edit only the task operand until lengths match.
2. **Filter.** Keep pairs with ΔTUL = TUL(clean) − TUL(corrupt) ≥ 2 nats, so opposite-signed pairs do not cancel. Aim for at least 200 pairs across at least 5 environments.
3. **Attribution.** Run `attribute` with method `EAP-IG-inputs`, ig_steps = 5 and a metric returning mean TUL (TUL is defined in M1).
4. **Validation.** Prune to the top n edges for n from 100 to 10,000. Report faithfulness (circuit alone) and completeness (circuit removed). Pass if under 1% of edges recovers at least 70% of ΔTUL.
5. **Ablations.** Ablate the top 10 components one at a time; report the over-call rate on need = 0 tasks and accuracy on hard tasks.
6. **Link to the directions.** Project `d_SE` and `d_need` onto the inputs of each top component (through W_V and W_O for a head, through the input weights for an MLP). A small projection means the component that drives calling does not read the uncertainty or necessity signal.

## Optional: feature-level attribution graphs

Use [circuit-tracer](https://github.com/decoderesearch/circuit-tracer) (`ReplacementModel`, `attribute`) to name the features flowing through the circuit, for 5–10 over-call prompts. Only do this if transcoders exist for Qwen3-4B-Instruct-2507 itself. The published Qwen3-4B transcoders may target the hybrid Qwen3-4B, not the 2507 Instruct release.

## Tools

| Repo | Role | What to reuse | What to add |
| --- | --- | --- | --- |
| [EAP-IG](https://github.com/hannamw/eap-ig) | Circuit discovery on heads and MLPs | `Graph`, `attribute(..., method='EAP-IG-inputs', ig_steps=5)`, `evaluate_graph` | A TUL metric function; a pair CSV built from When2Tool tasks |
| [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) | Hooked model EAP-IG runs on | `HookedTransformer` with split QKV, attention-result and MLP-input hooks | Check that the model loads and matches HF logits |
| [nnsight](https://nnsight.net) | Fallback if TransformerLens cannot load the model | Hooks on HF modules | A small node-level attribution-patching script |
| [circuit-tracer](https://github.com/decoderesearch/circuit-tracer) | Feature-level graphs | `ReplacementModel`, `attribute` | Only with matching transcoders |

## Pitfalls

| Pitfall | Symptom | Fix |
| --- | --- | --- |
| GQA in Qwen3 | EAP-IG errors on head-level KV edges | Enable `ungroup_grouped_query_attention`, or attribute at head-output level only |
| Token-length mismatch in pairs | EAP-IG cannot align positions | Discard the pair; never pad with filler tokens inside the task text |
| Gradient cancellation | Weak, noisy attribution | Keep only pairs with ΔTUL ≥ 2 and report the bucket sizes |
| Bf16 precision in gradients | Unstable edge scores across seeds | Run attribution in float32 for the 4B model |

## Deliverables and cost

- Code: `toolmech/circuit.py` (pairs, EAP-IG wrapper, faithfulness, ablations).
- Figures: faithfulness and completeness against the number of edges; component map (layer × head and MLP); single-component ablation table.
- Compute: roughly 2–6 GPU-hours on one 80 GB GPU for 200 pairs (an estimate, not a measurement).

## References

- Hanna et al., [Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms](https://arxiv.org/abs/2403.17806), COLM 2024.
- Zhao et al., *Wired for Overconfidence*, COLM 2026: EAP-IG with a logit-difference metric, edge-level faithfulness and completeness checks.
- Ameisen et al., [Circuit Tracing: Revealing Computational Graphs in Language Models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html), 2025.