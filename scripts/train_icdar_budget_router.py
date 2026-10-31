"""Approximate per-dataset YOLO/U-Net gates with separate selection/calibration."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

from scripts.icdar_budget_data import read,sha,prepare_population
from scripts.run_yolo_f1_quality_study import seal
from recognition.icdar_budget_router import budget_policy

REPO=Path(__file__).resolve().parents[1]


def prepare(args):
    if args.mode!='approximate' or args.config.resolve().parent!=REPO/'configs/approximate' or args.paths.resolve().parent!=REPO/'configs/approximate':
        raise ValueError('Explicit approximate namespace required')
    c=read(args.config);paths=read(args.paths)
    if c['mode']!='approximate' or paths['mode']!='approximate':raise ValueError('Scientific mode mismatch')
    for dataset in ('icdar13','icdar2015'):budget_policy(c,dataset)
    for file,digest in c['pinned_inputs'].items():
        if sha(file)!=digest:raise ValueError('Qualified source changed: '+file)
    records,parts,metadata=prepare_population(paths,c)
    source=read(paths['icdar2015_selection']);candidates={'icdar13':['tt_corrected','tt_raw'],
        'icdar2015':['tt_corrected','tt_raw']+[v['name'] for v in source['candidates']]}
    if args.smoke:
        for ds,p in parts.items():
            for role,ids in p.items():p[role]=ids[:c['smoke'][role]]
        for ds,data in records.items():
            data['train']=[s for s in data['train'] if s['sample_id'] in set(parts[ds]['fit']+parts[ds]['selection']+parts[ds]['calibration'])]
            data['test']=[s for s in data['test'] if s['sample_id'] in set(parts[ds]['test'])]
    root=Path(paths['output_root'])/('smoke' if args.smoke else 'full')
    folders=['models','recognition','localize','scripts','segmentation','deepsolo','caption','scoring','pipeline','research']
    code={str(p.relative_to(REPO)):sha(p) for folder in folders if (REPO/folder).exists() for p in (REPO/folder).rglob('*.py')}
    import torch,numpy,sklearn
    plan=dict(mode='approximate',claim_label='approximate reproduction',config=c,paths=paths,records=records,partitions=parts,
        populations=metadata,candidates=candidates,icdar2015_native_selection=source,config_sha256=sha(args.config),paths_sha256=sha(args.paths),
        code_sha256=code,result_class='smoke-test result' if args.smoke else 'approximate reproduction',
        hardware=dict(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,numpy=numpy.__version__,sklearn=sklearn.__version__),
        test_used_for_selection=False,scope='New dataset-specific trained gates; no unchanged TotalText router transfer',
        classified_limitations=c['scientific_choices'])
    seal(root/'resolved_configuration.json',plan)
    for relative,digest in code.items():
        p=root/'code_snapshot'/relative;p.parent.mkdir(parents=True,exist_ok=True)
        if sha(REPO/relative)!=digest:raise ValueError('Source changed while freezing')
        if p.exists() and sha(p)!=digest:raise ValueError('Frozen code changed')
        if not p.exists():shutil.copyfile(REPO/relative,p)
    shutil.copytree(REPO/'configs',root/'code_snapshot/configs',dirs_exist_ok=True)
    # Existing qualified legacy exporter receives only explicit RGB records.
    export=dict(mode='approximate',claim_label='approximate reproduction',config=dict(c,paths=dict(read(paths['corrected_transfer_config'])['paths'],
        **{k:paths[k] for k in ['deepsolo_profile','deepsolo_runtime','deepsolo_source']})),
        population=dict(test={ds+'_train':data['train'] for ds,data in records.items()}))
    seal(root/'fallback_population.json',export)
    print(json.dumps(dict(status='PREPARED',root=str(root),populations={d:{k:len(v) for k,v in p.items()} for d,p in parts.items()})),flush=True)
    return root


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--mode');p.add_argument('--config',type=Path);p.add_argument('--paths',type=Path)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--run-root',type=Path)
    p.add_argument('--stage',choices=['yolo_detect','yolo_read','unet_selection','unet_fit','train','unet_test','evaluate'],default='train');args=p.parse_args()
    root=args.run_root.resolve() if args.run_root else prepare(args)
    if args.prepare_only:return
    if REPO!=(root/'code_snapshot').resolve():
        subprocess.run([sys.executable,'-u','-m','scripts.train_icdar_budget_router','--run-root',str(root),'--stage',args.stage],cwd=root/'code_snapshot',
            env={**os.environ,'PYTHONPATH':str(root/'code_snapshot')},check=True);return
    from scripts.icdar_budget_stages import execute
    execute(root,args.stage)


if __name__=='__main__':main()
