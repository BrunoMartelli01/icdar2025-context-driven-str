"""Refit one common head family on matching views; never opens TEST outputs."""

import argparse
import copy
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from scripts.run_uniform_unet_training import read,write
from segmentation.uniform_unet_fusion import checked_mode,file_hash


def prepare(path):
    c=read(path);checked_mode(path,c)
    for name,digest in c['pinned_inputs'].items():
        if file_hash(name)!=digest:raise ValueError('Changed head-refit input: '+name)
    root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=False);write(root/'resolved_configuration.json',c)
    repo=Path(__file__).resolve().parents[1];manifest={}
    for package in c['snapshot_packages']:
        for p in (repo/package).rglob('*.py'):
            relative=p.relative_to(repo);dest=root/'source_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,dest);manifest[str(relative)]=file_hash(dest)
    write(root/'source_manifest.json',manifest)
    write(root/'contract.json',dict(config_sha256=file_hash(path),source_manifest_sha256=file_hash(root/'source_manifest.json'),test_used=False))


def refreshed(config):
    result=copy.deepcopy(config)
    for name in result['pinned_inputs']:
        if '/projects/' in name:result['pinned_inputs'][name]=file_hash(name)
    return result


def run(path):
    c=read(path);checked_mode(path,c);root=Path(c['output_root']);banks=Path(c['matching_bank_root'])
    assert read(root/'contract.json')['config_sha256']==file_hash(path)
    for name,digest in c['pinned_inputs'].items():
        if file_hash(name)!=digest:raise ValueError('Changed head-refit source: '+name)
    for name,digest in read(root/'source_manifest.json').items():assert file_hash(root/'source_snapshot'/name)==digest
    def verify_live_sources(config):
        manifest=read(root/'source_manifest.json')
        for name in config['pinned_inputs']:
            p=Path(name)
            if p.suffix=='.py' and p.is_relative_to(c['repository_root']):
                relative=str(p.relative_to(c['repository_root']))
                if relative in manifest and file_hash(p)!=manifest[relative]:
                    raise ValueError('Live source differs from executing frozen snapshot: '+name)
    def status(**values):write(root/'status.json',dict(pid=os.getpid(),**values))
    def execute(module,args,tag):
        with (root/(tag+'.log')).open('a') as log:
            child=subprocess.Popen([sys.executable,'-u','-m',module,*args],cwd=root/'source_snapshot',stdout=log,stderr=subprocess.STDOUT)
            status(status='RUNNING_REFIT_OR_AUDIT',module=module,tag=tag,child_pid=child.pid)
            if child.wait()!=0:raise RuntimeError('Head stage failed; inspect '+str(root/(tag+'.log')))
    with (root/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not all((banks/'full'/d/'bank/independent_audit.json').exists() for d in c['dataset_order']):
            launch=read(banks/'launch.json');cmd=Path('/proc')/str(launch['pid'])/'cmdline'
            if not cmd.exists() or b'build_uniform_multiscale_word_banks' not in cmd.read_bytes():
                raise RuntimeError('Matching bank producer is no longer live; preserve incomplete banks')
            status(status='WAITING_LIVE_MATCHING_BANK_PRODUCER',producer_pid=launch['pid'],producer=read(banks/'status.json'))
            time.sleep(20)
        verify_live_sources(read(c['base_utility_config']))
        fit=refreshed(read(c['base_utility_config']))
        fit.update(output_root=c['utility_output_root'],alternate_fallback_fractions=[],hard_budgets=c['hard_budgets'],
            observation_frontend_policy=str(banks/'frontend_policy.json'),
            fit_export_overrides={d:str(banks/'full'/d/'bank') for d in c['dataset_order']},
            validation_export_overrides={d:str(banks/'full'/d/'validation_bank') for d in c['dataset_order']})
        for p in [banks/'frontend_policy.json']+[banks/'full'/d/k/n for d in c['dataset_order'] for k in ('bank','validation_bank') for n in ('index.json','independent_audit.json')]:
            fit['pinned_inputs'][str(p)]=file_hash(p)
        fit['scientific_choices'].update(c['scientific_choices'])
        target=Path(c['utility_config'])
        if target.exists():assert read(target)==fit
        else:write(target,fit)
        for ds in c['dataset_order']:
            out=Path(fit['output_root'])/ds
            if not (out/'selection_frozen.json').exists():execute('scripts.train_uniform_utility_fusion',['--config',str(target),'--dataset',ds],'fit_'+ds)
            if not (out/'independent_audit.json').exists():execute('scripts.audit_uniform_utility_fusion',['--config',str(target),'--dataset',ds],'audit_'+ds)
            assert read(out/'independent_audit.json')['status']=='PASS'
        verify_live_sources(read(c['base_union_config']))
        union=refreshed(read(c['base_union_config']))
        for name in ('hard_budgets','alternate_fallback_fractions','observation_frontend_policy','fit_export_overrides','validation_export_overrides'):
            union[name]=fit[name]
        union.update(base_utility_config=str(target),output_root=c['union_output_root'])
        union['pinned_inputs'].update(fit['pinned_inputs']);union['pinned_inputs'][str(target)]=file_hash(target)
        for ds in c['dataset_order']:
            out=Path(fit['output_root'])/ds
            for name in ('selection_frozen.json','independent_audit.json','word_probabilities.json'):
                union['pinned_inputs'][str(out/name)]=file_hash(out/name)
        union['scientific_choices'].update(c['scientific_choices']);upath=Path(c['union_config'])
        if upath.exists():assert read(upath)==union
        else:write(upath,union)
        if not (Path(union['output_root'])/'pooled_union_model.joblib').exists():execute('scripts.train_uniform_word_union',['--config',str(upath)],'union_fit')
        execute('scripts.train_uniform_word_union',['--config',str(upath),'--audit'],'union_audit')
        results={};gains=[]
        for ds in c['dataset_order']:
            q=c['hard_budgets'][ds]['maximum_fallback_images'];out=Path(union['output_root'])/ds
            f=read(out/'selection_frozen.json');a=read(out/'independent_audit.json');assert a['status']=='PASS' and a['frozen_sha256']==file_hash(out/'selection_frozen.json')
            candidate=f['policies'][str(q)];reference=read(c['reference_screens'][ds])['results']['scaled']
            assert candidate['deepsolo_images']==reference['deepsolo_images'] and candidate['total_images']==reference['total_images']
            delta=candidate['counts']['f1']-reference['counts']['f1'];gains.append(delta)
            results[ds]=dict(transferred=reference,matched_refit=candidate,gain=delta)
        macro=sum(gains)/len(gains)
        selected='matched_refit' if macro>c['minimum_macro_gain'] and min(gains)>=-c['maximum_domain_loss'] else 'transferred'
        write(root/'shared_selection_frozen.json',dict(status='FROZEN',mode='approximate',claim_label='approximate reproduction',
            results=results,macro_gain=macro,selected=selected,config_sha256=file_hash(path),source_sha256=file_hash(__file__),
            test_used_for_selection=False,actual_new_neural_calls=0,
            scope='One common native-versus-matching-refit decision using reused VAL; same explicit hard quotas. Full RGB/runtime qualification and full TEST remain required.'))
        status(status='ALL_MATCHED_HEADS_AUDITED',selected=selected)
        print('MATCHING HEADS AUDITED',selected,'macro gain',macro,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--stage',choices=['prepare','run'],required=True);a=p.parse_args()
    try:prepare(a.config) if a.stage=='prepare' else run(a.config)
    except BaseException as exc:
        root=Path(read(a.config)['output_root'])
        if root.exists():write(root/f'failure_{time.time_ns()}.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
