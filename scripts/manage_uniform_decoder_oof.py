"""Freeze and supervise resource-aware decoder-heldout training and exports."""

import argparse
import copy
import fcntl
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.run_uniform_unet_training import read,write
from recognition.common_budget_pipeline import grouped_folds
from segmentation.uniform_unet_fusion import file_hash
from scripts.uniform_decoder_folds import check_fold

REPO=Path(__file__).resolve().parents[1]


def prepare(path):
    c=read(path)
    if c['mode']!='approximate' or path.resolve().parent!=REPO/'configs/approximate':raise ValueError('Explicit approximate OOF profile required')
    for p,digest in c['pinned_inputs'].items():
        if file_hash(p)!=digest:raise ValueError('Changed OOF source artifact: '+p)
    root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=False)
    jobs=[]
    for ds in c['dataset_order']:
        source=Path(c['training_root'])/ds;original=read(source/'resolved_configuration.json');complete=read(source/'complete.json')
        if read(source/'independent_training_audit.json')['status']!='PASS' or complete['selected_epoch']<1:raise ValueError('A genuinely retrained qualified deployment model is required')
        lookup={r['sample_id']:r for r in original['train_records']};fit=original['partitions']['fit']
        keys={s:lookup[s]['rgb_sha256'] for s in fit};folds=grouped_folds(fit,keys,c['folds'],c['seed'])
        for i,held in enumerate(folds):
            plan=copy.deepcopy(original);plan['partitions']['fit']=sorted(set(fit)-set(held));plan['partitions']['decoder_heldout']=held
            plan['config']['training']['epochs']=complete['selected_epoch']
            plan['fold_contract']=dict(fold=i,original_fit_ids=fit,fixed_epochs=complete['selected_epoch'],global_emission_policy=str(source/'emission_selection/frozen.json'),
                heldout_scope='Decoder gradient exclusion; pretrained encoder already has TRAIN exposure')
            plan['artifact_sha256']=c['neural_artifact_sha256'];plan['root_config_sha256']=file_hash(path)
            check_fold(plan);directory=root/ds/('fold_'+str(i));write(directory/'resolved_configuration.json',plan)
            jobs.append(dict(dataset=ds,fold=i,root=str(directory),heldout_ids=held,fit_images=len(plan['partitions']['fit']),epochs=complete['selected_epoch']))
    smoke=copy.deepcopy(read(Path(jobs[0]['root'])/'resolved_configuration.json'))
    smoke['partitions']['fit']=smoke['partitions']['fit'][:8];smoke['partitions']['decoder_heldout']=smoke['partitions']['decoder_heldout'][:4]
    smoke['fold_contract']['original_fit_ids']=smoke['partitions']['fit']+smoke['partitions']['decoder_heldout']
    smoke['fold_contract']['fixed_epochs']=1;smoke['config']['training']['epochs']=1;smoke['smoke']=True
    smoke['result_class']='smoke-test result';check_fold(smoke);write(root/'smoke/resolved_configuration.json',smoke)
    snapshot=root/'source_snapshot';sources={}
    for package in c['snapshot_packages']:
        for src in (REPO/package).rglob('*.py'):
            dst=snapshot/src.relative_to(REPO);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
            sources[str(src.relative_to(REPO))]=file_hash(dst)
            if file_hash(src)!=sources[str(src.relative_to(REPO))]:raise ValueError('OOF source changed during freeze')
    shutil.copytree(REPO/'configs',snapshot/'configs')
    write(root/'source_manifest.json',sources);write(root/'resolved_configuration.json',c)
    write(root/'jobs_frozen.json',dict(status='FROZEN',mode='approximate',jobs=jobs,config_sha256=file_hash(path),test_used_for_selection=False))
    print('FROZEN DECODER OOF',len(jobs),'fits',sum(j['fit_images']*j['epochs'] for j in jobs),'training image-epochs',flush=True)


def aggregate(root,ds,jobs):
    index={};folds=[]
    for job in jobs:
        if job['dataset']!=ds:continue
        folder=Path(job['root']);audit=read(folder/'heldout/independent_audit.json');entries=read(folder/'heldout/index.json')
        if audit['status']!='PASS' or audit['index_sha256']!=file_hash(folder/'heldout/index.json'):raise ValueError('Heldout export qualification changed')
        if set(index)&set(entries):raise ValueError('Repeated decoder-heldout sample')
        for sid,entry in entries.items():
            if file_hash(entry['path'])!=entry['sha256']:raise ValueError('Changed heldout observation')
            sample=read(entry['path'])
            if sample['ground_truth_used'] or not sample['decoder_fit_excluded']:raise ValueError('Invalid decoder-heldout inference boundary')
        index.update(entries);folds.append(dict(fold=job['fold'],root=job['root'],audit_sha256=file_hash(folder/'heldout/independent_audit.json')))
    source=read(root/'resolved_configuration.json')['training_root'];plan=read(Path(source)/ds/'resolved_configuration.json')
    if set(index)!=set(plan['partitions']['fit']):raise ValueError('Incomplete native FIT decoder-heldout bank')
    bank=root/ds/'bank';write(bank/'index.json',index)
    write(bank/'independent_audit.json',dict(status='PASS',images=len(index),folds=folds,index_sha256=file_hash(bank/'index.json'),
        decoder_fit_excluded=True,upstream_encoder_exposure=True,source_training_plan_sha256=file_hash(Path(source)/ds/'resolved_configuration.json'),
        scope='All five independently audited neural fits/exports, unique native FIT coverage; not fully nested or pretraining-unseen OOF'))


def run(path):
    c=read(path);root=Path(c['output_root']);frozen=read(root/'jobs_frozen.json')
    if frozen['config_sha256']!=file_hash(path):raise ValueError('Changed frozen decoder OOF configuration')
    def status(**fields):write(root/'supervisor_status.json',dict(recorded_unix=time.time(),**fields))
    with (root/'supervisor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def execute(folder,stage,gpu):
            if gpu:
                while True:
                    if shutil.disk_usage(root).free<c['minimum_free_disk_bytes']:raise RuntimeError('OOF disk reserve exhausted')
                    output=subprocess.run(['nvidia-smi','--query-gpu=name,memory.free,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip()
                    if len(output.splitlines())!=1:raise RuntimeError('Expected the declared single-GPU host')
                    free=int(output.split(',')[-2].strip())*1024**2
                    if free>=c['minimum_free_gpu_bytes']:break
                    status(status='WAITING_GPU',stage=stage,run_root=str(folder),observed=output,supervisor_pid=__import__('os').getpid())
                    time.sleep(20)
                write(folder/'hardware'/f'{stage}_{time.time_ns()}.json',dict(observed=output))
            log=folder/(stage+'.log')
            with log.open('a') as stream:
                child=subprocess.Popen([sys.executable,'-u','-m','scripts.uniform_decoder_folds','--stage',stage,'--run-root',str(folder)],
                    cwd=root/'source_snapshot',stdout=stream,stderr=subprocess.STDOUT)
                while child.poll() is None:
                    status(status='RUNNING',stage=stage,run_root=str(folder),child_pid=child.pid,supervisor_pid=__import__('os').getpid());time.sleep(10)
                if child.returncode:raise RuntimeError(f'OOF {stage} failed: {folder}; see {log}')
        work=[dict(dataset='smoke',root=str(root/'smoke'))]+frozen['jobs']
        for job in work:
            folder=Path(job['root'])
            for stage,result,gpu in [('train','training_complete.json',True),('audit','independent_training_audit.json',False),
                                     ('export','heldout/complete.json',True),('audit-export','heldout/independent_audit.json',False)]:
                if not (folder/result).exists():execute(folder,stage,gpu)
            if read(folder/'independent_training_audit.json')['status']!='PASS' or read(folder/'heldout/independent_audit.json')['status']!='PASS':raise RuntimeError('OOF qualification did not pass')
            last=folder/'last.pt';final=folder/'final.pt';complete=read(folder/'training_complete.json')
            if last.exists() and c['remove_redundant_completed_optimizer_checkpoint']:
                if file_hash(last)!=complete['optimizer_checkpoint_sha256'] or file_hash(final)!=complete['checkpoint_sha256']:raise ValueError('Completed checkpoint changed before compaction')
                receipt=dict(status='IN_PROGRESS',removed_path=str(last),removed_sha256=complete['optimizer_checkpoint_sha256'],bytes=last.stat().st_size,
                    preserved_final_path=str(final),preserved_final_sha256=complete['checkpoint_sha256'],
                    reason='Completed fixed-epoch fold; independent training audit verified identical final model tensors. Preserve inference weights/history; optimizer continuation state is redundant for this completed fold.')
                write(folder/'optimizer_checkpoint_compaction.json',receipt);last.unlink();receipt['status']='COMPLETE';write(folder/'optimizer_checkpoint_compaction.json',receipt)
            ds=job['dataset']
            if ds!='smoke' and job['fold']==c['folds']-1:aggregate(root,ds,frozen['jobs'])
        status(status='ALL_DECODER_HELDOUT_BANKS_QUALIFIED',datasets=c['dataset_order'],supervisor_pid=__import__('os').getpid(),next_stage='Refit the same common router using decoder-heldout FIT observations')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--stage',choices=['prepare','run'],required=True);a=p.parse_args()
    if a.stage=='prepare':prepare(a.config)
    else:
        try:run(a.config)
        except BaseException as exc:
            root=Path(read(a.config)['output_root']);write(root/f'supervisor_failure_{time.time_ns()}.json',dict(status='FAILED',error=repr(exc)));raise
