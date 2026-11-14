#!/usr/bin/env bash
# Create the Merlin GPU runner job from job_run.json. Pass --dry-run to preview.
set -euo pipefail
J="${JOB_SPEC:-$(dirname "${BASH_SOURCE[0]}")/job_run.json}"  # override with JOB_SPEC=path
g(){ python3 -c "import json;v=json.load(open('$J'))['$1'];print(v if isinstance(v,str) else json.dumps(v))"; }
export BYTEDCLI_NETWORK_PROFILE=prod NO_COLOR=1
bytedcli --json --site i18n-tt merlin job-v2 runs create "$@" \
  --name "$(g name)" --description "$(g description)" --job-config "$(g job_config)" \
  --compliance "$(g compliance)" --resource-config "$(g resource_config)" --attachments "$(g attachments)"
