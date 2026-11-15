---
name: toolmech-decisions
description: Decisions on the toolmech (tool-use uncertainty) project made with the user on 2026-11-14 that deviate from PLAN.md
metadata:
  node_type: memory
  type: project
  originSessionId: ffa6709d-428e-472f-b0bc-bfba94b9d479
  modified: 2026-11-14T13:57:52.669Z
---

Running on the local A40 box at /tool-use-uncertainty, not Merlin/devbox (user handles any leftover Merlin jobs).

- Primary analysis prompt mode is `sparse_tool`, with `current` as the comparison (user approved 2026-11-14). Reason: in `current` the model calls on 91% of tasks and call behaviour has AUROC 0.52 for need, so there is almost no decision to study; `sparse_tool` gives ~33% mixed tasks.
- Tool use is measured from the model's actual replies (call rate, greedy decision); TUL is kept only as the patching readout (user approved 2026-11-14 after TUL missed late calls in `sparse_tool`). `d_act^dm` is built in `sparse_tool` only.
- The user considered per-run "prefill"/decision-point directions (E3b) and decided the current last-prompt-token approach is fine; don't reopen without a reason.
- All HF passes run in fp32: bf16 moved TUL by up to 0.9 nats near the decision boundary.
- Approved and written into PLAN.md (revision log): steering direction is a stratified difference in means of SE within env × tier, matched on fail rate (`d_SE^dm`); activation patching on the shared response-policy suffix is a second E4 causal test (M5b).
- The user prefers plain-language explanations; they found AUROC-heavy answers confusing.

**Why:** these change what every later step measures.
**How to apply:** keep both modes in E1–E4 outputs; docs/handoff.md holds the operational state.
