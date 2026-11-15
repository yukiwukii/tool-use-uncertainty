#!/usr/bin/env bash
# E0 (reduced): When2Tool steps 2,3 + step 1 restricted to current/no_reasoning. TP=1 on one A40.
set -euo pipefail
ROOT=/tool-use-uncertainty
export HF_HOME=$ROOT/.hf_cache HF_HUB_OFFLINE=1 HF_HUB_ENABLE_HF_TRANSFER=0 PATH=$ROOT/.venv/bin:$PATH
cd $ROOT/when2tool
M=qwen3-4b-instruct; P=./probe_data/$M; O=./outputs/$M; mkdir -p $P $O
t() { local s=$(date +%s); "$@"; echo "[timing] $1 $2: $(( $(date +%s)-s ))s"; }
echo "== step 2: extract features"; t python src/extract_features.py --model_path $M --output_dir $P \
  --data_path ./data/tasks_v1_train.json --data_path_test ./data/tasks_v1_test.json --tensor_parallel_size 1
echo "== step 3: train probe";      t python src/train_probe.py --data_dir $P --mode no_reasoning --reg 10000 --all_layers
echo "== step 1: eval current/no_reasoning"; t python src/run_eval.py --model_path $M --backend vllm \
  --data_path ./data/tasks_v1_test.json --record_mode lite --prompt_mode current --reasoning_mode no_reasoning \
  --output_path $O/ --tensor_parallel_size 1 --n_runs 3
echo "== E0 done"
