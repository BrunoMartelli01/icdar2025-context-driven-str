"""Privileged fixed-output routing bound; never a deployable image selector."""

import argparse
from pathlib import Path

from scripts.run_uniform_unet_training import read,write
from scripts.evaluate_uniform_unet_fusion import fallback_rows
from scripts.icdar_budget_data import predictions
from scripts.analyze_essential_oracle import fixed_acceptance_oracle
from segmentation.uniform_unet_fusion import annotation,file_hash,word_counts,aggregate_counts
from recognition.uniform_utility_deployment import UniformUtilityPipeline
from recognition.hard_fallback_budget import validate_policy
from recognition.budget_target_oracle import target_residual


def run(config,evaluation,dataset,output):
    c=read(config);e=read(evaluation);u=read(c['utility_config'])
    q=c['hard_budgets'][dataset]['maximum_fallback_images'];contract=validate_policy(c['hard_budgets'],dataset,q)
    source=Path(u['training_root'])/dataset;plan=read(source/'resolved_configuration.json')
    ids=plan['partitions']['test'];assert len(ids)==contract['reference_images']
    parent=Path(u['output_root'])/dataset/'evaluation';decisions=read(parent/'decisions_before_fallback.json')
    assert file_hash(parent/'observations.json')==decisions['observations_sha256']
    observed=read(parent/'observations.json')[str(q)];index=read(source/'export_test/index.json')
    pipeline=UniformUtilityPipeline(Path(c['utility_config']),dataset,q)
    for path,digest in e['pinned_inputs'].items():assert file_hash(path)==digest
    spec=e['fallback_test_sources'][dataset];archive=read(spec['path']) if spec['kind']!='icdar13_files' else None
    records={r['sample_id']:r for r in plan['test_records']};branches={};fused={};eligible=[]
    for sid in ids:
        entry=index[sid];assert file_hash(entry['path'])==entry['sha256'];sample=read(entry['path'])
        assert not sample['ground_truth_used']
        truth,valid=annotation(records[sid]);flags=[r['ignored'] for r in predictions(sample['raw_rows'],records[sid],valid)]
        o=observed[sid];light=word_counts(o['rows'],truth,[flags[int(r['source_row'])] for r in o['rows']])
        if o['rows']:eligible.append(sid)
        heavy,expected,_=fallback_rows(e,dataset,sid,archive)
        hc=word_counts(heavy,truth,[r['ignored'] for r in heavy])
        if expected is not None:assert hc==expected
        rows=pipeline.combine_after_fallback(o,heavy)
        ff=[flags[int(r['source_row'])] if r.get('fixed_light_complement') else r['ignored'] for r in rows]
        fc=word_counts(rows,truth,ff)
        branches[sid]=dict(light=light,deepsolo=hc);fused[sid]=dict(light=light,deepsolo=fc)
    accepted=set(decisions['decisions'][str(q)]['accepted_ids'])
    recounted=aggregate_counts(v['light' if s in accepted else 'deepsolo'] for s,v in fused.items())
    assert recounted==read(parent/'results.json')['results'][str(q)]['counts']
    eligible=eligible if pipeline.policy['guard'] else ids
    result=dict(status='PRIVILEGED_DIAGNOSTIC_ONLY',mode='approximate',dataset=dataset,
        deployed_counts=recounted,deepsolo_images=q,total_images=len(ids),deepsolo_percentage=100*q/len(ids),
        native_fallback_bound=fixed_acceptance_oracle(branches,eligible,len(ids)-q),
        fixed_fusion_bound=fixed_acceptance_oracle(fused,eligible,len(ids)-q),
        native_target_feasibility=target_residual(branches,eligible,str(plan['target'])),
        fixed_fusion_target_feasibility=target_residual(fused,eligible,str(plan['target'])),
        diagnostic_cache_access=dict(deepsolo_images=len(ids),total_images=len(ids),deepsolo_percentage=100.),
        actual_new_neural_calls=0,test_ground_truth_used_for_diagnostic=True,operational_selection=False,
        config_sha256=file_hash(config),evaluation_config_sha256=file_hash(evaluation),source_sha256=file_hash(__file__),
        scope='Exact-cardinality GT oracle for fixed current branches; not a learned router, generalization result or attainable guarantee.')
    if output.exists():raise ValueError('Preserve previous bound diagnostic')
    write(output,result)
    print(dataset,'privileged bounds',result['native_fallback_bound']['f1'],result['fixed_fusion_bound']['f1'],
          'logical fallback',q,'/',len(ids),'actual new calls0',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--evaluation-config',type=Path,required=True)
    p.add_argument('--dataset',choices=['icdar13','icdar2015','totaltext'],required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.config,a.evaluation_config,a.dataset,a.output)
