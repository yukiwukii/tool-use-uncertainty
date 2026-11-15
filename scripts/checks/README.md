Ad-hoc checks from the overnight run.
- vsteer_check.py: steering in vLLM vs HF at L18 (act^dm, k=±4); outputs vs_vllm.json / vs_hf.json. Tool-token log-prob shifts matched (11.35 vs 11.39, 4.62 vs 4.43, 9.03 vs 9.44).
- e2check.py: E2 same-window agreement check (all >= 0.98).
