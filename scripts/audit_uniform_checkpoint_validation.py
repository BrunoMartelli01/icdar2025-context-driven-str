"""Independently rank and recount retained-checkpoint validation comparisons."""

import argparse
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from scripts.run_uniform_unet_training import read,write
from scripts.export_uniform_unet import rows_with_sources
from scripts.icdar_budget_data import key,predictions
from scripts.audit_uniform_risk_fusion import count
from segmentation.uniform_unet_fusion import file_hash,annotation,aggregate_counts
from recognition.uniform_encoder_context import raw_context
from recognition.uniform_utility_deployment import UniformUtilityPipeline
from recognition.uniform_utility_fusion import paired_f1_lower


def audit(path):
    c=read(path);root=Path(c['output_root']);frozen=read(root/'selection_frozen.json')
    assert frozen['config_sha256']==file_hash(path) and not frozen['test_used_for_selection']
    for name,digest in c['pinned_inputs'].items():assert file_hash(name)==digest
    for name,digest in read(root/'evidence_sha256.json').items():assert file_hash(name)==digest
    ds=c['dataset'];u=read(c['utility_config']);source=Path(u['training_root'])/ds;plan=read(source/'resolved_configuration.json')
    ids=plan['partitions']['selection'];records={r['sample_id']:r for r in plan['train_records']};contract=c['hard_budgets'][ds];q=contract['maximum_fallback_images']
    assert len(plan['partitions']['test'])==contract['reference_images'] and contract['batch_rounding']=='floor'
    k=q*len(ids)//contract['reference_images'];pipeline=UniformUtilityPipeline(Path(c['utility_config']),ds,q,split='train')
    context=raw_context(u,{s:records[s] for s in ids});index=read(source/'export_train/index.json');teacher=read(u['fallback_train_sources'][ds])['native_benchmark']
    decisions=read(root/'decisions_before_fallback.json');saved=read(root/'per_image.json');triples={}
    assert decisions['status']=='FROZEN' and not decisions['ground_truth_used'] and not decisions['fallback_outputs_loaded']
    for label,case in c['cases'].items():
        observed={};raws={}
        for sid in ids:
            if case is None:
                sample=read(index[sid]['path']);raw=sample['raw_rows'];mp=source/'export_train/maps'/(key(sid)+'.npz')
            else:
                folder=Path(case['root'])/'selection'/f"epoch_{case['epoch']:03d}"
                sample=read(folder/'samples'/(key(sid)+'.json'));raw,_=rows_with_sources(sample['raw_rows'],0.,records[sid]['image_wh']);mp=folder/'maps'/(key(sid)+'.npz')
            raws[sid]=raw
            with np.load(mp,allow_pickle=False) as z:
                vector=z['encoder_context'] if case is not None and case.get('actual_encoder_context',False) else context[sid]
                observed[sid]=pipeline.observe(raw,records[sid]['image_wh'],z['probabilities'],vector)
        eligible=[s for s in ids if observed[s]['rows']] if pipeline.policy['guard'] else ids
        accepted=set(sorted(eligible,key=lambda s:(-observed[s]['score'],records[s]['rgb_sha256'],s))[:len(ids)-k])
        d=decisions['decisions'][label]
        assert accepted==set(d['accepted_ids']) and set(ids)-accepted==set(d['fallback_ids'])
        assert d['deepsolo_images']==k and d['total_images']==len(ids) and abs(d['deepsolo_percentage']-100*k/len(ids))<1e-12
        values=[]
        for sid in ids:
            truth,valid=annotation(records[sid]);flags=[r['ignored'] for r in predictions(raws[sid],records[sid],valid)]
            rows=observed[sid]['rows'] if sid in accepted else pipeline.combine_after_fallback(observed[sid],teacher[sid]['rows'])
            ff=[flags[int(r['source_row'])] if sid in accepted or r.get('fixed_light_complement') else r['ignored'] for r in rows]
            got=count(rows,truth,ff);expected=saved[label][sid]
            assert rows==expected['rows'] and ff==expected['ignored'] and got==expected['counts']
            assert expected['action']==('light' if sid in accepted else 'deepsolo');values.append(got)
        result=frozen['results'][label];assert aggregate_counts(values)==result['counts']
        assert result['deepsolo_images']==k and result['total_images']==len(ids)
        triples[label]=np.array([[v[t] for t in ('tp','fp','fn')] for v in values])
    indices=np.random.default_rng(c['seed']).integers(len(ids),size=(c['bootstrap_draws'],len(ids)))
    for label in triples:
        lower=0. if label=='baseline' else paired_f1_lower(triples['baseline'],triples[label],indices,c['bootstrap_alpha'])
        assert lower==frozen['results'][label]['paired_lower_gain']
    best=max(r['counts']['f1'] for r in frozen['results'].values())
    selected=next(label for label in c['cases'] if frozen['results'][label]['counts']['f1']>=best-c['selection_tolerance'])
    assert selected==frozen['selected']
    write(root/'independent_audit.json',dict(status='PASS',candidates=len(c['cases']),unique_images=len(ids),candidate_image_replays=len(ids)*len(c['cases']),
        deepsolo_images_per_candidate=k,total_images_per_candidate=len(ids),deepsolo_percentage_per_candidate=100*k/len(ids),
        actual_new_neural_calls=0,selected=selected,frozen_sha256=file_hash(root/'selection_frozen.json'),source_sha256=file_hash(__file__),
        scope='Independent score sorting and Counter recount of all saved predictions; shared declared feature/fusion/bootstrap primitives.'))
    print('Checkpoint pipeline validation audit PASS:',len(ids)*len(c['cases']),'candidate-image replays.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=2):audit(a.config)
