#!/usr/bin/env bash
# Entrypoint of the Merlin GPU job: prepare a GPU host that the devbox drives
# over SSH (see toolmech/gpu.sh). Work dir /opt/toolmech, venv /opt/tm_venv,
# HF cache /opt/hf_cache. Stays up until /opt/toolmech/STOP exists or
# MAX_HOURS pass; stop the job from the devbox when done.
set -uo pipefail
MAX_HOURS=${MAX_HOURS:-96}
HLOG=/mnt/hdfs/nadya.wangsajaya/toolmech/job.log
mkdir -p /opt/toolmech
log() { echo "[$(date '+%F %T')] $*" | tee -a /opt/toolmech/job.log; echo "[$(date '+%F %T')] $*" >> "$HLOG" 2>/dev/null; }
log "start host=$(hostname) gpus=$(nvidia-smi -L | wc -l)"

IDX=https://bytedpypi.byted.org/simple/
pip3 install -q uv -i $IDX >/dev/null 2>&1 || pip install -q uv -i $IDX
UV=$(command -v uv || echo "$(python3 -m site --user-base)/bin/uv")
$UV venv -q -p python3.11 /opt/tm_venv
cat > /opt/toolmech_req.txt <<'EOF'
vllm==0.8.5
transformers==4.55.2
accelerate==1.13.0
datasets>=3
scikit-learn
scipy
pandas
pyarrow
pyyaml
matplotlib
pytest
EOF
$UV pip install -q --python /opt/tm_venv/bin/python --index-url $IDX -r /opt/toolmech_req.txt
log "venv: $(/opt/tm_venv/bin/python -c 'import vllm,torch;print(vllm.__version__,torch.cuda.is_available())' 2>&1)"

export HF_HOME=/opt/hf_cache
if /opt/tm_venv/bin/python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-4B-Instruct-2507')" >> /opt/toolmech/job.log 2>&1; then
  log "model downloaded"
else
  log "model download FAILED (push it over ssh)"
fi
touch /opt/toolmech/READY
log "READY"

end=$(( $(date +%s) + MAX_HOURS * 3600 ))
while [ ! -f /opt/toolmech/STOP ] && [ "$(date +%s)" -lt "$end" ]; do sleep 60; done
log "exit"
