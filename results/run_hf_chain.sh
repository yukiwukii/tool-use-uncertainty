#!/bin/sh
# Overnight HF chain: E4 patching (already running as its own process), then E5 residuals.
cd /tool-use-uncertainty
export PYTHONPATH=$PWD HF_HOME=$PWD/.hf_cache HF_HUB_OFFLINE=1 HF_HUB_ENABLE_HF_TRANSFER=0
while ps -eo args | grep -v grep | grep -q "toolmech.patch"; do sleep 60; done
echo "== $(date -u +%H:%M) patch finished"
while [ ! -f results/e5/CONV_DONE ]; do sleep 60; done
s=$(date +%s); echo "== $(date -u +%H:%M) start e5 resid"
.venv/bin/python -m toolmech.trajectory --stage resid >> results/e5_resid.log 2>&1
echo "== $(date -u +%H:%M) e5 resid exit=$? in $(( $(date +%s)-s ))s"
