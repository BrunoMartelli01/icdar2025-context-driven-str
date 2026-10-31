"""Compare retained neural checkpoints on the complete frozen validation pipeline."""

import argparse
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from scripts.run_uniform_unet_training import read,write
from scripts.export_uniform_unet import rows_with_sources
from scripts.icdar_budget_data import key,predictions
from segmentation.uniform_unet_fusion import file_hash,annotation,word_counts,aggregate_counts
from recognition.uniform_encoder_context import raw_context
from recognition.uniform_utility_deployment import UniformUtilityPipeline
from recognition.uniform_risk_fusion import route
from recognition.hard_fallback_budget import validate_policy,batch_quota,verify_route
from recognition.uniform_utility_fusion import paired_f1_lower


def run(path):
    c=read(path)
    if c['mode']!='approximate' or path.resolve().parent.name!='approximate':raise ValueError('Explicit approximate validation replay required')
    for name,digest in c['pinned_inputs'].items():
        if file_hash(name)!=digest:raise ValueError('Changed retained checkpoint input: '+name)
    root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=False);write(root/'resolved_configuration.json',c)
    ds=c['dataset'];u=read(c['utility_config']);source=Path(u['training_root'])/ds;plan=read(source/'resolved_configuration.json')
    ids=plan['partitions']['selection'];records={r['sample_id']:r for r in plan['train_records']};q=c['hard_budgets'][ds]['maximum_fallback_images']
    contract=validate_policy(c['hard_budgets'],ds,q);k=batch_quota(contract,q,len(plan['partitions']['test']),len(ids))
    pipeline=UniformUtilityPipeline(Path(c['utility_config']),ds,q,split='train')
    context=raw_context(u,{s:records[s] for s in ids});index=read(source/'export_train/index.json')
    canonical=read(source/'emission_selection/frozen.json')['selected']['policy']
    observations={};raws={};decisions={};evidence={}
    for label,case in c['cases'].items():
        observations[label]={};raws[label]={}
        if case is not None:
            folder=Path(case['root']);p=read(folder/'resolved_configuration.json');a=read(folder/'independent_training_audit.json')
            assert a['status']=='PASS' and p['partitions']['selection']==ids
            if 'encoder_adaptation' in p['config'] and not case.get('actual_encoder_context',False):
                raise ValueError('Adapted encoder requires actual context from its own forward')
            geometry={name:value for name,value in p['config']['inference'].items() if name!='minimum_transcript_probability'}
            assert geometry=={name:value for name,value in canonical.items() if name!='minimum_transcript_probability'},'Generate readings at the qualified canonical crop geometry first'
        for sid in ids:
            if case is None:
                entry=index[sid];sample_path=Path(entry['path']);assert file_hash(sample_path)==entry['sha256']
                sample=read(sample_path);assert not sample['ground_truth_used'];raw=sample['raw_rows'];mp=source/'export_train/maps'/(key(sid)+'.npz')
            else:
                directory=folder/'selection'/f"epoch_{case['epoch']:03d}";sample_path=directory/'samples'/(key(sid)+'.json');sample=read(sample_path)
                assert not sample['ground_truth_used_for_prediction'];mp=directory/'maps'/(key(sid)+'.npz')
                raw,_=rows_with_sources(sample['raw_rows'],0.,records[sid]['image_wh'])
            assert sample['image_sha256']==records[sid]['image_sha256'] and file_hash(mp)==sample['maps_sha256']
            evidence[str(sample_path)]=file_hash(sample_path);evidence[str(mp)]=sample['maps_sha256'];raws[label][sid]=raw
            with np.load(mp,allow_pickle=False) as z:
                vector=z['encoder_context'] if case is not None and case.get('actual_encoder_context',False) else context[sid]
                observations[label][sid]=pipeline.observe(raw,records[sid]['image_wh'],z['probabilities'],vector)
        decision=route(observations[label],{s:records[s]['rgb_sha256'] for s in ids},k,pipeline.policy['guard'])
        verify_route(decision,ids,contract,q);decisions[label]=decision
    write(root/'decisions_before_fallback.json',dict(status='FROZEN',decisions=decisions,ground_truth_used=False,fallback_outputs_loaded=False))
    teacher=read(u['fallback_train_sources'][ds])['native_benchmark'];truth={};valid={}
    for sid in ids:truth[sid],valid[sid]=annotation(records[sid])
    results={};per_image={};triples={}
    for label,case in c['cases'].items():
        accepted=set(decisions[label]['accepted_ids']);per_image[label]={}
        for sid in ids:
            flags=[r['ignored'] for r in predictions(raws[label][sid],records[sid],valid[sid])];o=observations[label][sid]
            rows=o['rows'] if sid in accepted else pipeline.combine_after_fallback(o,teacher[sid]['rows'])
            ff=[flags[int(r['source_row'])] if sid in accepted or r.get('fixed_light_complement') else r['ignored'] for r in rows]
            per_image[label][sid]=dict(rows=rows,ignored=ff,counts=word_counts(rows,truth[sid],ff),action='light' if sid in accepted else 'deepsolo')
        counts=aggregate_counts(v['counts'] for v in per_image[label].values());triples[label]=np.array([[per_image[label][s]['counts'][t] for t in ('tp','fp','fn')] for s in ids])
        results[label]=dict(counts=counts,deepsolo_images=k,total_images=len(ids),deepsolo_percentage=100*k/len(ids),actual_new_neural_calls=0)
    frozen=read(Path(u['output_root'])/ds/'selection_frozen.json')['policies'][str(q)]
    assert results['baseline']['counts']==frozen['counts'] and set(decisions['baseline']['accepted_ids'])==set(frozen['accepted_validation_ids'])
    indices=np.random.default_rng(c['seed']).integers(len(ids),size=(c['bootstrap_draws'],len(ids)))
    for label in results:
        results[label]['paired_lower_gain']=0. if label=='baseline' else paired_f1_lower(triples['baseline'],triples[label],indices,c['bootstrap_alpha'])
    best=max(r['counts']['f1'] for r in results.values())
    selected=next(label for label,r in results.items() if r['counts']['f1']>=best-c['selection_tolerance'])
    write(root/'per_image.json',per_image);write(root/'evidence_sha256.json',evidence)
    write(root/'selection_frozen.json',dict(status='FROZEN',mode='approximate',dataset=ds,results=results,selected=selected,
        test_used_for_selection=False,config_sha256=file_hash(path),source_sha256=file_hash(__file__),
        scope='Complete frozen image/word/fusion pipeline on TRAIN-heldout validation. Endpoint checkpoints retained; diagnostic bootstrap is reported, not a promotion gate. Reused validation/historical exposure remain.'))
    print('JOINT VALIDATION',[(label,round(r['counts']['f1'],6)) for label,r in results.items()],'selected',selected,'DeepSolo',k,'/',len(ids),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=2):run(a.config)
