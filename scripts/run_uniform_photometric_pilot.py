"""Prepare matched continuations and qualify the IC13 pilot before other domains."""

import argparse
import copy
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.run_uniform_unet_training import read,write
from segmentation.uniform_unet_fusion import file_hash,aggregate_counts
from scripts.icdar_budget_data import key


def prepare(path):
    c=read(path);base=Path(c['output_root'])
    if c['mode']!='approximate' or path.resolve().parent.name!='approximate':raise ValueError('Explicit approximate continuation profile required')
    for name,digest in c['pinned_inputs'].items():
        if file_hash(name)!=digest:raise ValueError('Changed qualified continuation source: '+name)
    base.mkdir(parents=True,exist_ok=False);write(base/'resolved_configuration.json',c)
    repo=Path(__file__).resolve().parents[1]
    for arm,parameters in c['variants'].items():
        for role in ('smoke','full'):
            parent=base/arm/role;manifest={}
            for package in c['snapshot_packages']:
                for src in (repo/package).rglob('*.py'):
                    relative=src.relative_to(repo);dest=parent/'code_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(src,dest);manifest[str(relative)]=file_hash(dest)
            write(parent/'source_manifest.json',manifest)
            for ds in c['dataset_order']:
                source=Path(c['source_training_root'])/ds;plan=copy.deepcopy(read(source/'resolved_configuration.json'))
                complete=read(source/'complete.json');assert read(source/'independent_training_audit.json')['status']=='PASS'
                checkpoint=source/'best.pt'
                if file_hash(checkpoint)!=complete['checkpoint_sha256']:raise ValueError('Changed native U-Net checkpoint')
                plan['config']['training'].update(c['training_override'])
                plan['config'].update(c.get('loss_override',{}))
                if parameters is not None:plan['config']['photometric_augmentation']=parameters
                plan['initialization']['source_checkpoint']=str(checkpoint)
                plan['smoke']=role=='smoke';plan['result_class']='smoke-test result' if role=='smoke' else 'approximate reproduction'
                if role=='smoke':
                    plan['partitions']={name:ids[:(8 if name=='fit' else 4)] for name,ids in plan['partitions'].items()}
                directory=source/'selection'/f"epoch_{complete['selected_epoch']:03d}"
                expected=aggregate_counts(read(directory/'samples'/(key(s)+'.json'))['counts'] for s in plan['partitions']['selection'])
                active=set(plan['partitions']['fit']+plan['partitions']['selection'])
                for record in plan['train_records']:
                    if record['sample_id'] not in active:continue
                    for field in ('image','annotation','mask','validity','separator','owners'):
                        if file_hash(record[field+'_path'])!=record[field+'_sha256']:raise ValueError('Changed training record: '+record['sample_id'])
                plan.update(root_config_sha256=file_hash(path),source_manifest_sha256=file_hash(parent/'source_manifest.json'),
                    experiment=c.get('experiment_kind','uniform_matched_photometric_continuation_v11'),continuation_arm=arm,
                    continuation_control=dict(source_plan_sha256=file_hash(source/'resolved_configuration.json'),
                        source_checkpoint_sha256=complete['checkpoint_sha256'],expected_initial_validation=expected))
                if 'matched_control_root' in c:
                    control=Path(c['matched_control_root'])/role/ds;reference=read(control/'resolved_configuration.json')
                    unchanged={k:v for k,v in plan['config'].items() if k not in c.get('loss_override',{})}
                    if (unchanged!=reference['config'] or plan['partitions']!=reference['partitions'] or
                            plan['initialization']!=reference['initialization'] or plan['train_records']!=reference['train_records']):
                        raise ValueError('Proposed loss intervention differs from its matched control in other scientific inputs')
                    assert read(control/'independent_training_audit.json')['status']=='PASS'
                    plan['matched_control']=dict(root=str(control),plan_sha256=file_hash(control/'resolved_configuration.json'),
                        complete_sha256=file_hash(control/'complete.json'),audit_sha256=file_hash(control/'independent_training_audit.json'))
                write(parent/ds/'resolved_configuration.json',plan)
    print('Prepared continuation plans for',len(c['dataset_order']),'domains; no TEST RGB or annotation opened.',flush=True)


def run(path):
    c=read(path);base=Path(c['output_root']);ds=c['pilot_dataset'];stages=[]
    lock=(base/'pilot.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if read(base/'resolved_configuration.json')!=c:raise ValueError('Continuation configuration changed')
    write(base/'launch.json',dict(pid=os.getpid(),config_sha256=file_hash(path),started_unix=time.time()))
    for arm in c['variants']:
        for role in ('smoke','full'):
            parent=base/arm/role;root=parent/ds;plan=read(root/'resolved_configuration.json')
            if not (root/'complete.json').exists():
                while True:
                    free=shutil.disk_usage(base).free
                    if free<plan['config']['minimum_free_disk_bytes']:raise RuntimeError('Continuation disk reserve exhausted')
                    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.free,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip()
                    if len(gpu.splitlines())!=1:raise RuntimeError('Qualified single GPU required')
                    if int(gpu.split(',')[-2])*1024**2>=plan['config']['minimum_free_gpu_bytes']:break
                    write(base/'status.json',dict(status='WAITING_GPU',pid=os.getpid(),arm=arm,role=role,observed=gpu));time.sleep(20)
                write(base/(arm+'_'+role+'_resources.json'),dict(gpu=gpu,free_disk_bytes=free))
                with (base/(arm+'_'+role+'_train.log')).open('a') as log:
                    child=subprocess.Popen([sys.executable,'-u','-m','scripts.run_uniform_unet_training','--stage','train','--run-root',str(root)],cwd=parent/'code_snapshot',stdout=log,stderr=subprocess.STDOUT)
                    write(base/'status.json',dict(status='TRAINING',pid=os.getpid(),child_pid=child.pid,arm=arm,role=role,dataset=ds))
                    if child.wait()!=0:raise RuntimeError('Continuation training failed; preserve '+str(root))
            if not (root/'independent_training_audit.json').exists():
                with (base/(arm+'_'+role+'_audit.log')).open('a') as log:
                    subprocess.run([sys.executable,'-u','-m','scripts.audit_uniform_unet_training','--run-root',str(root)],cwd=parent/'code_snapshot',stdout=log,stderr=subprocess.STDOUT,check=True)
            if read(root/'independent_training_audit.json')['status']!='PASS':raise RuntimeError('Continuation audit did not pass')
            stages.append(dict(arm=arm,role=role,complete=read(root/'complete.json')))
    write(base/'pilot_results.json',dict(status='PILOT_AUDITED',mode='approximate',claim_label='approximate reproduction',dataset=ds,
        stages=stages,test_used=False,other_domains_trained=False,actual_new_deepsolo_calls=0,
        scope=c.get('result_scope','Matched ten-epoch no-augmentation/photo continuation on IC13 native FIT only; no router/TEST or three-domain target claim.')))
    write(base/'status.json',dict(status='PILOT_AUDITED',pid=os.getpid(),dataset=ds))
    print('Matched IC13 continuation pilot audited.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--stage',choices=['prepare','run'],required=True);a=p.parse_args()
    try:prepare(a.config) if a.stage=='prepare' else run(a.config)
    except BaseException as exc:
        import traceback
        c=read(a.config);base=Path(c['output_root'])
        if base.exists():write(base/f'failure_{time.time_ns()}.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
