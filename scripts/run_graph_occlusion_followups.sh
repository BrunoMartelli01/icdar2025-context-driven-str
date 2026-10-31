#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 3 || $# -gt 4 ]]; then
  printf '%s\n' 'Usage: run_graph_occlusion_followups.sh PYTHON CONFIG OUTPUT [--smoke]' >&2
  exit 2
fi
task_python=$1
task_config=$2
task_output=$3
task_smoke=()
if [[ ${4:-} == --smoke ]]; then task_smoke=(--smoke); fi
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
for task_stage in threshold continuation fragment_data fragment_graph contour_data contour_graph report; do
  "$task_python" - "$task_config" "$task_output" "$task_stage" <<'PY'
import json,sys,subprocess,time,os
from pathlib import Path
c=json.loads(Path(sys.argv[1]).read_text());out=Path(sys.argv[2]);stage=sys.argv[3]
required=c['runtime']['minimum_free_gpu_bytes']+c['runtime']['admission_margin_bytes']
while True:
    free=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*1024**2
    if free>=required:break
    out.mkdir(parents=True,exist_ok=True);path=out/'status.json';tmp=out/'status.json.tmp'
    tmp.write_text(json.dumps(dict(status='WAITING_FOR_GPU',stage=stage,free_bytes=free,required_bytes=required,pid=os.getppid()))+'\n');tmp.replace(path)
    print('WAITING_FOR_GPU',stage,free,required,flush=True)
    time.sleep(c['runtime']['admission_poll_seconds'])
PY
  nvidia-smi --query-gpu=name,uuid,memory.total,memory.used --format=csv
  "$task_python" -m scripts.run_graph_occlusion_followups --mode approximate --config "$task_config" --output "$task_output" --stage "$task_stage" "${task_smoke[@]}"
done
