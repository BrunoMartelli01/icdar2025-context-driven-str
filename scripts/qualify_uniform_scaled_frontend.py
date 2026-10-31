"""RGB-only resolution export, native parity and hard-budget validation replay."""

import argparse
from pathlib import Path
import shutil

import numpy as np
from threadpoolctl import threadpool_limits

from scripts.run_uniform_unet_training import read,write
from scripts.icdar_budget_data import key,predictions
from pipeline.uniform_word_pipeline import input_records
from pipeline.uniform_scaled_frontend import export
from recognition.uniform_encoder_context import raw_context
from recognition.uniform_utility_deployment import UniformUtilityPipeline
from recognition.uniform_risk_fusion import route
from recognition.hard_fallback_budget import validate_policy,batch_quota,verify_route
from recognition.uniform_utility_fusion import paired_f1_lower
from scripts.audit_uniform_risk_fusion import count
from segmentation.uniform_unet_fusion import file_hash,annotation,word_counts,aggregate_counts


def inputs(path):
    c=read(path)
    if c['mode']!='approximate' or path.resolve().parent.name!='approximate':raise ValueError('Explicit approximate resolution profile required')
    for name,digest in c['pinned_inputs'].items():
        if file_hash(name)!=digest:raise ValueError('Changed resolution input: '+name)
    u=read(c['utility_config']);source=Path(u['training_root'])/c['dataset'];plan=read(source/'resolved_configuration.json')
    lookup={r['sample_id']:r for r in plan['train_records']};ids=plan['partitions']['selection']
    return c,u,source,plan,lookup,ids


def generate(path):
    c,u,source,plan,lookup,ids=inputs(path);root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=False)
    write(root/'resolved_configuration.json',c);repo=Path(__file__).resolve().parents[1];manifest={}
    for package in c['snapshot_packages']:
        for p in (repo/package).rglob('*.py'):
            relative=p.relative_to(repo);dest=root/'code_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,dest);manifest[str(relative)]=file_hash(dest)
    write(root/'source_manifest.json',manifest)
    if shutil.disk_usage(root).free<c['minimum_free_disk_bytes']:raise RuntimeError('Insufficient resolution export disk reserve')
    records=input_records(dict(samples=[dict(sample_id=s,image_path=lookup[s]['image_path']) for s in ids]))
    for r in records:
        assert r['rgb_sha256']==lookup[r['sample_id']]['rgb_sha256'] and r['image_sha256']==lookup[r['sample_id']]['image_sha256']
    import subprocess
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.free,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip()
    if len(gpu.splitlines())!=1 or int(gpu.split(',')[-2])*1024**2<c['minimum_free_gpu_bytes']:raise RuntimeError('Qualified single GPU or free memory unavailable')
    write(root/'resource_before.json',dict(gpu=gpu,free_disk_bytes=shutil.disk_usage(root).free))
    probe=records[:c['qualification_images']];export(source,probe,root/'native_probe',plan['config']['image_hw'][0])
    index=read(source/'export_train/index.json');context=raw_context(u,{r['sample_id']:r for r in probe})
    actual=read(root/'native_probe/index.json')
    for r in probe:
        sid=r['sample_id'];a=read(actual[sid]['path']);b=read(index[sid]['path'])
        with np.load(a['maps_path'],allow_pickle=False) as x,np.load(source/'export_train/maps'/(key(sid)+'.npz'),allow_pickle=False) as y:
            np.testing.assert_array_equal(x['probabilities'],y['probabilities']);np.testing.assert_array_equal(x['encoder_context'],context[sid])
        fields=['raw_text','bbox','crop_rgb_sha256','sequence_confidence_native','token_probabilities']
        assert [[v[k] for k in fields] for v in a['raw_rows']]==[[v[k] for k in fields] for v in b['raw_rows']]
    write(root/'native_parity_audit.json',dict(status='PASS',images=len(probe),maps_and_context_exact=True,raw_readings_and_crop_hashes_exact=True))
    export(source,records,root/'scaled',c['canvas'])
    print('Scaled export COMPLETE:',len(records),'RGB images at',c['canvas'],'native probe parity PASS.',flush=True)


def evaluate(path,audit=False):
    c,u,source,plan,records,ids=inputs(path);root=Path(c['output_root']);assert read(root/'native_parity_audit.json')['status']=='PASS'
    ds=c['dataset'];q=c['hard_budgets'][ds]['maximum_fallback_images'];contract=validate_policy(c['hard_budgets'],ds,q)
    k=batch_quota(contract,q,len(plan['partitions']['test']),len(ids));pipeline=UniformUtilityPipeline(Path(c['utility_config']),ds,q,split='train')
    old_context=raw_context(u,{s:records[s] for s in ids});observed={};raws={};decisions={};hashes={}
    for label,folder in [('baseline',source/'export_train'),('scaled',root/'scaled')]:
        index=read(folder/'index.json');observed[label]={};raws[label]={}
        for sid in ids:
            entry=index[sid];assert file_hash(entry['path'])==entry['sha256'];sample=read(entry['path']);assert not sample['ground_truth_used']
            mp=folder/'maps'/(key(sid)+'.npz');assert file_hash(mp)==sample['maps_sha256'];hashes[str(mp)]=sample['maps_sha256']
            hashes[entry['path']]=entry['sha256'];raws[label][sid]=sample['raw_rows']
            with np.load(mp,allow_pickle=False) as z:
                context=old_context[sid] if label=='baseline' else z['encoder_context']
                observed[label][sid]=pipeline.observe(sample['raw_rows'],records[sid]['image_wh'],z['probabilities'],context)
        if audit:
            eligible=[s for s in ids if observed[label][s]['rows']] if pipeline.policy['guard'] else ids
            accepted=sorted(eligible,key=lambda s:(-observed[label][s]['score'],records[s]['rgb_sha256'],s))[:len(ids)-k]
            decisions[label]=dict(accepted_ids=sorted(accepted),fallback_ids=sorted(set(ids)-set(accepted)),deepsolo_images=k,total_images=len(ids),deepsolo_percentage=100*k/len(ids))
        else:decisions[label]=route(observed[label],{s:records[s]['rgb_sha256'] for s in ids},k,pipeline.policy['guard'])
        verify_route(decisions[label],ids,contract,q)
    if not audit:write(root/'decisions_before_fallback.json',dict(status='FROZEN',decisions=decisions,ground_truth_used=False,fallback_outputs_loaded=False))
    else:
        frozen_decisions=read(root/'decisions_before_fallback.json')
        assert not frozen_decisions['ground_truth_used'] and not frozen_decisions['fallback_outputs_loaded']
        for label,d in decisions.items():
            for field in d:assert d[field]==frozen_decisions['decisions'][label][field]
    teacher=read(u['fallback_train_sources'][ds])['native_benchmark'];truth={};valid={}
    for sid in ids:truth[sid],valid[sid]=annotation(records[sid])
    per_image={};results={};triples={}
    for label in observed:
        accepted=set(decisions[label]['accepted_ids']);per_image[label]={}
        for sid in ids:
            flags=[r['ignored'] for r in predictions(raws[label][sid],records[sid],valid[sid])];o=observed[label][sid]
            rows=o['rows'] if sid in accepted else pipeline.combine_after_fallback(o,teacher[sid]['rows'])
            ff=[flags[int(r['source_row'])] if sid in accepted or r.get('fixed_light_complement') else r['ignored'] for r in rows]
            per_image[label][sid]=dict(rows=rows,ignored=ff,counts=(count if audit else word_counts)(rows,truth[sid],ff))
        counts=aggregate_counts(v['counts'] for v in per_image[label].values());triples[label]=np.array([[per_image[label][s]['counts'][t] for t in ('tp','fp','fn')] for s in ids])
        results[label]=dict(counts=counts,deepsolo_images=k,total_images=len(ids),deepsolo_percentage=100*k/len(ids),actual_new_deepsolo_calls=0)
    reference=read(Path(u['output_root'])/ds/'selection_frozen.json')['policies'][str(q)]
    original_k=len(ids)-len(reference['accepted_validation_ids'])
    reference_check=None
    if original_k==k:
        assert results['baseline']['counts']==reference['counts'] and set(decisions['baseline']['accepted_ids'])==set(reference['accepted_validation_ids'])
    else:
        # The retained pre-hard-budget selection used ceil on IC15 (46/100).
        # First reproduce that original reference exactly, then compare both
        # current branches at the required floor quota (45/100).
        eligible=[s for s in ids if observed['baseline'][s]['rows']] if pipeline.policy['guard'] else ids
        original_accepted=set(sorted(eligible,key=lambda s:(-observed['baseline'][s]['score'],records[s]['rgb_sha256'],s))[:len(ids)-original_k])
        assert original_accepted==set(reference['accepted_validation_ids'])
        values=[]
        for sid in ids:
            flags=[r['ignored'] for r in predictions(raws['baseline'][sid],records[sid],valid[sid])]
            o=observed['baseline'][sid]
            rows=o['rows'] if sid in original_accepted else pipeline.combine_after_fallback(o,teacher[sid]['rows'])
            ff=[flags[int(r['source_row'])] if sid in original_accepted or r.get('fixed_light_complement') else r['ignored'] for r in rows]
            values.append((count if audit else word_counts)(rows,truth[sid],ff))
        original_counts=aggregate_counts(values)
        assert original_counts==reference['counts']
        reference_check=dict(original_counts=original_counts,original_deepsolo_images=original_k,total_images=len(ids),
            original_deepsolo_percentage=100*original_k/len(ids),current_hard_deepsolo_images=k,
            current_hard_deepsolo_percentage=100*k/len(ids),original_counts_and_routes_exact=True)
    indices=np.random.default_rng(c['seed']).integers(len(ids),size=(c['bootstrap_draws'],len(ids)))
    lower=paired_f1_lower(triples['baseline'],triples['scaled'],indices,c['bootstrap_alpha'])
    selected='scaled' if results['scaled']['counts']['f1']>results['baseline']['counts']['f1']+c['selection_tolerance'] else 'baseline'
    result=dict(status='FROZEN',mode='approximate',claim_label='approximate reproduction',dataset=ds,results=results,selected=selected,
        scaled_paired_lower_gain=lower,config_sha256=file_hash(path),source_sha256=file_hash(__file__),test_used_for_selection=False,
        context_scope=c.get('context_scope','Actual encoder context at each resolution; transferred768-trained PCA/word/image/fusion heads. Matching refits remain required before final claims.'))
    if reference_check is not None:result['original_reference_verification']=reference_check
    if audit:
        assert result==read(root/'selection_frozen.json') and per_image==read(root/'per_image.json')
        assert hashes==read(root/'evidence_sha256.json')
        write(root/'independent_audit.json',dict(status='PASS',candidate_image_replays=2*len(ids),frozen_sha256=file_hash(root/'selection_frozen.json'),
            scope='Independent score sorting and Counter recount; shared declared features/fusion/bootstrap. Native actual-neural parity qualified separately.'))
    else:
        write(root/'selection_frozen.json',result);write(root/'per_image.json',per_image);write(root/'evidence_sha256.json',hashes)
    print('Scaled validation','AUDIT PASS' if audit else 'FROZEN',{name:round(r['counts']['f1'],6) for name,r in results.items()},'selected',selected,'fallback',k,'/',len(ids),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--stage',choices=['export','score','audit'],required=True);a=p.parse_args()
    try:
        with threadpool_limits(limits=2):generate(a.config) if a.stage=='export' else evaluate(a.config,a.stage=='audit')
    except BaseException as exc:
        import time,traceback
        root=Path(read(a.config)['output_root'])
        if root.exists():write(root/f'failure_{time.time_ns()}.json',dict(error=repr(exc),traceback=traceback.format_exc(),out_of_memory='out of memory' in str(exc).lower()))
        raise
