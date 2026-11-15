#!/bin/sh
# Overnight vLLM chain (one vLLM engine at a time; HF jobs run alongside in run_hf_chain.sh).
cd /tool-use-uncertainty
export VLLM_USE_V1=0 PYTHONPATH=$PWD HF_HOME=$PWD/.hf_cache HF_HUB_OFFLINE=1 HF_HUB_ENABLE_HF_TRANSFER=0
retry() {  # vLLM start-up can fail if the HF job's memory moves during profiling: retry
  name=$1; shift
  for i in 1 2 3 4; do
    s=$(date +%s); echo "== $(date -u +%H:%M) start $name (try $i)"
    "$@" && { echo "== $(date -u +%H:%M) done $name in $(( $(date +%s)-s ))s"; return 0; }
    echo "== $(date -u +%H:%M) FAILED $name (try $i)"; sleep 30
  done
  return 1
}
retry steerAB .venv/bin/python -m toolmech.steer --phases A,B --gpu_util 0.5 >> results/e4_steer.log 2>&1
retry conv .venv/bin/python -m toolmech.conversations --gpu_util 0.5 >> results/e5_conv.log 2>&1
touch results/e5/CONV_DONE
retry sepost .venv/bin/python -m toolmech.trajectory --stage sepost --gpu_util 0.5 >> results/e5_sepost.log 2>&1
retry steerCDEF .venv/bin/python -m toolmech.steer --phases C,D,E,F --gpu_util 0.5 >> results/e4_steer.log 2>&1
echo "== $(date -u +%H:%M) chain finished"
