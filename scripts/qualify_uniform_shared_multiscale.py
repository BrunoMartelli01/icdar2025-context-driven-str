"""Sequential three-domain VAL screen with a predeclared shared selection rule."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from scripts.run_uniform_unet_training import read, write
from segmentation.uniform_unet_fusion import file_hash


def run(path):
    c=read(path)
    if c['mode']!='approximate' or path.resolve().parent.name!='approximate':raise ValueError('Approximate shared profile required')
    for name,digest in c['pinned_inputs'].items():
        if file_hash(name)!=digest:raise ValueError('Changed shared-screen input: '+name)
    root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=False)
    write(root/'resolved_configuration.json',c)
    results={'icdar13':Path(c['completed_icdar13_screen'])} if c.get('completed_icdar13_screen') else {}
    candidate_module=c.get('candidate_module','scripts.qualify_uniform_multiscale_masks')
    for ds,job in c['jobs'].items():
        write(root/'status.json',dict(status='RUNNING',dataset=ds,stage='actual_scaled_export'))
        if 'existing_scale_root' not in job:
            subprocess.run([sys.executable,'-u','-m','scripts.qualify_uniform_scaled_frontend',
                '--config',job['scale_config'],'--stage','export'],check=True)
        scale=Path(read(job['scale_config'])['output_root'])
        if 'existing_scale_root' in job and scale!=Path(job['existing_scale_root']):
            raise ValueError('Existing scale root differs from its declared export profile')
        blend=read(job['blend_template']);blend['scaled_export']=str(scale/'scaled')
        for p in [scale/'scaled/index.json',scale/'scaled/complete.json',scale/'native_parity_audit.json',scale/'native_probe/complete.json']:
            blend['pinned_inputs'][str(p)]=file_hash(p)
        blend_path=Path(job['resolved_blend_config'])
        if blend_path.exists():raise ValueError('Preserve existing resolved blend profile')
        write(blend_path,blend)
        write(root/'status.json',dict(status='RUNNING',dataset=ds,stage='blend_export_and_audits'))
        if 'existing_blend_root' not in job:
            subprocess.run([sys.executable,'-u','-m',candidate_module,
                '--config',str(blend_path),'--stage','export'],check=True)
        else:
            old=Path(job['existing_blend_root']);target=Path(blend['output_root'])
            old_config=read(old/'resolved_configuration.json')
            for field in ('dataset','larger_weight','scaled_export','utility_config','hard_budgets'):
                if old_config[field]!=blend[field]:raise ValueError('Reuse would change scientific blend inputs')
            complete=read(old/'scaled/complete.json')
            assert complete['status']=='COMPLETE' and not complete['ground_truth_used']
            target.mkdir(parents=True,exist_ok=False);write(target/'resolved_configuration.json',blend)
            (target/'scaled').symlink_to(old/'scaled',target_is_directory=True)
            shutil.copyfile(old/'source_map_evidence.json',target/'source_map_evidence.json')
            write(target/'export_reuse_receipt.json',dict(source=str(old),source_complete_sha256=file_hash(old/'scaled/complete.json'),
                actual_new_neural_calls=0,scope='Byte-identical previously completed export; scoring-only quota-reference repair.'))
            repo=Path(__file__).resolve().parents[1];manifest={}
            for name in ('scripts/qualify_uniform_scaled_frontend.py','scripts/qualify_uniform_multiscale_masks.py',
                         'scripts/qualify_uniform_multiscale_native_dependency.py','scripts/qualify_uniform_shared_multiscale.py',
                         'segmentation/uniform_multiscale_masks.py'):
                dest=target/'scoring_code_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(repo/name,dest);manifest[name]=file_hash(dest)
            write(target/'scoring_code_manifest.json',manifest)
        subprocess.run([sys.executable,'-m','scripts.qualify_uniform_multiscale_native_dependency',
            '--config',str(blend_path)],check=True)
        for stage in ('score','audit'):
            subprocess.run([sys.executable,'-u','-m',candidate_module,
                '--config',str(blend_path),'--stage',stage],check=True)
        results[ds]=Path(blend['output_root'])
    summary={};gains=[];hashes={};candidate_name=c.get('candidate_name','probability_mean')
    for ds,folder in results.items():
        f=read(folder/'selection_frozen.json');a=read(folder/'independent_audit.json')
        if not (a['status']=='PASS' and read(folder/'independent_map_audit.json')['status']=='PASS' and not f['test_used_for_selection']):
            raise ValueError('All independent VAL audits must pass')
        assert a['frozen_sha256']==file_hash(folder/'selection_frozen.json')
        values=f['results'];delta=values['scaled']['counts']['f1']-values['baseline']['counts']['f1'];gains.append(delta)
        summary[ds]={'baseline':values['baseline'],candidate_name:values['scaled'],'gain':delta}
        for name in ('selection_frozen.json','independent_audit.json','independent_map_audit.json'):
            hashes[str(folder/name)]=file_hash(folder/name)
    mean=sum(gains)/len(gains)
    selected=candidate_name if mean>c['minimum_macro_gain'] and min(gains)>=-c['maximum_domain_loss'] else 'baseline'
    write(root/'shared_selection_frozen.json',dict(status='FROZEN',mode='approximate',claim_label='approximate reproduction',
        results=summary,macro_gain=mean,selected=selected,config_sha256=file_hash(path),source_sha256=file_hash(__file__),
        test_used_for_selection=False,actual_new_deepsolo_calls=0,evidence_sha256=hashes,
        scope='Three-domain VAL inference screen with transferred original heads. Matching banks/refits and full TEST remain required; no goal-completion claim.'))
    write(root/'status.json',dict(status='SHARED_SCREEN_AUDITED',selected=selected))
    print('SHARED MULTISCALE',selected,'macro gain',mean,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    try:run(a.config)
    except BaseException as exc:
        root=Path(read(a.config)['output_root'])
        if root.exists():write(root/f'failure_{time.time_ns()}.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
