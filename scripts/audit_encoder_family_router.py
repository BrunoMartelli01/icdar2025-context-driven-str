"""Independent split, normalization, checkpoint selection and routing-count audit."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from recognition.encoder_family_router import EncoderFamilyRouter, collate_observations
from recognition.encoder_family_inference import predict_encoder_gate
from scripts.audit_lightweight_unet_router import counts, assert_counts

read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();root=args.run
    plan=read(root/'resolved_configuration.json');c=plan['source_config'];frozen=read(root/'frozen_choices.json')
    result=read(root/'results.json');receipt=read(root/'completion.json');checks=Counter()
    assert receipt['results_sha256']==sha(root/'results.json') and result['frozen_choices_sha256']==sha(root/'frozen_choices.json')
    assert frozen['resolved_configuration_sha256']==sha(root/'resolved_configuration.json')
    assert plan['mode']=='approximate' and frozen['no_refit'] and not frozen['test_used'] and not plan['test_used']
    for path,digest in read(root/'test_sources.json')['input_sha256'].items():
        assert sha(path)==digest,path;checks['pinned_source_files']+=1
    fit=plan['fit_ids'];select=plan['selection_ids'];cal=plan['calibration_ids'];ids=sorted(next(iter(read(root/'test_scores.json').values())))
    assert len(fit)==1128 and len(select)==62 and len(cal)==63 and len(ids)==300
    assert len(set(fit+select+cal+ids))==1553
    reference=read(Path(c['paths']['budget_reference'])/'frozen_choices.json')
    assert select==reference['selection_ids'] and cal==reference['calibration_ids'];checks['disjoint_roles']+=1
    compact=Path(c['paths']['lightweight_run']);scalar={};branches={}
    for role in ['fit','calibration','selection','test']:
        schema=read(compact/(role+'_feature_schema.json'));branches.update(read(compact/(role+'_branches.json')))
        with np.load(compact/(role+'_features.npz'),allow_pickle=False) as z:x=np.concatenate([z['ocr'],z['mask']],1)
        scalar.update({s:v.astype(np.float32) for s,v in zip(schema['ids'],x)})
    x=np.stack([scalar[s] for s in fit]).astype(np.float64);mean=x.mean(0);scale=x.std(0);scale[scale==0]=1
    np.testing.assert_array_equal(mean,plan['feature_mean']);np.testing.assert_array_equal(scale,plan['feature_scale'])
    raw=np.asarray([[branches[s]['light']['tp']-branches[s]['deepsolo']['tp'],branches[s]['light']['detected_words']-branches[s]['deepsolo']['detected_words']] for s in fit],np.float32)
    np.testing.assert_array_equal(raw.mean(0),plan['inference_settings']['target_mean'])
    assert max(float(raw.std()),c['minimum_target_scale'])==plan['inference_settings']['target_scale']
    assert counts(branches,fit,[])['f1']==plan['inference_settings']['reference_f1'];checks['fit_only_statistics']+=1
    histories=read(root/'training_history.json');scores=read(root/'test_scores.json');curves={};intervals={};maximum_error={}
    source=Path(c['paths']['token_source']);readings=read(source/'test/corrected.json')
    train_readings=read(source/'train/corrected.json');held_observations={}
    for sid in select+cal:
        path=source/'train/features/corrected'/(sid.replace(':','__')+'.npz')
        with np.load(path,allow_pickle=False) as z:
            held_observations[sid]=dict(texts=[r['raw_text'] for r in train_readings[sid]],geometry=z['word_geometry'],
                scalars=((scalar[sid]-mean)/scale).astype(np.float32),patches=z['patches'][:196].copy(),
                patch_geometry=z['patch_geometry'][:196].copy(),patch_valid=z['patch_valid'][:196].copy(),
                word_text=z['word_text'],word_image=z['word_image'],queries=z['queries'])
    records={r['sample_id']:r for r in read(source/'resolved_configuration.json')['records']['test']}
    map_trace=read(compact/'test_feature_inputs.json');original_settings=read(compact/'resolved_configuration.json')['source_config']['map_features']
    torch.set_num_threads(c['cpu_threads']);single_subset=ids[::max(1,len(ids)//12)]
    for arm,choice in frozen['arm_choices'].items():
        checkpoints=[r['selection'] for r in histories[arm] if 'selection' in r]
        best=max(checkpoints,key=lambda r:(r['selection_summary']['f1'],-r['epoch']))
        assert best['epoch']==choice['epoch'] and best['checkpoint_sha256']==choice['checkpoint_sha256']
        assert sha(choice['checkpoint_path'])==choice['checkpoint_sha256'];checks['validation_checkpoint_selections']+=1
        checkpoint=torch.load(choice['checkpoint_path'],map_location='cpu',weights_only=True)
        assert checkpoint['fit_ids']==fit and checkpoint['scalar_mean']==plan['feature_mean'] and checkpoint['scalar_scale']==plan['feature_scale']
        assert checkpoint['inference_settings']==plan['inference_settings']
        assert sum(v.numel()*v.element_size() for v in checkpoint['state_dict'].values())==choice['head_tensor_bytes']
        model=EncoderFamilyRouter(arm,c['architecture'],c['seed']).eval();model.load_state_dict(checkpoint['state_dict'],strict=True)
        held_scores={};settings=checkpoint['inference_settings']
        for population in [select,cal]:
            for start in range(0,len(population),c['batch_size']):
                batch_ids=population[start:start+c['batch_size']]
                with torch.inference_mode():pred=model(collate_observations([held_observations[s] for s in batch_ids],arm,'cpu')).numpy()
                pred=pred*settings['target_scale']+np.asarray(settings['target_mean'])
                for sid,prediction in zip(batch_ids,pred):
                    eligible=bool(train_readings[sid]) and all(r['raw_text'].strip() for r in train_readings[sid])
                    held_scores[sid]=float(2*prediction[0]-settings['reference_f1']*prediction[1]) if eligible else None
        order=sorted((s for s in select if held_scores[s] is not None),key=lambda s:(-held_scores[s],s))
        assert_counts(counts(branches,select,order[:round(len(select)*.5)]),choice['selection_summary'])
        for policy in choice['policies']:
            order=sorted((s for s in cal if held_scores[s] is not None),key=lambda s:(-held_scores[s],s))
            boundary=order[round(len(cal)*policy['target_calibration_acceptance'])-1]
            assert boundary==policy['tie_sample_id'] and abs(held_scores[boundary]-policy['threshold'])<2e-4
            assert_counts(counts(branches,cal,order[:round(len(cal)*policy['target_calibration_acceptance'])]),policy['calibration_summary'])
            checks['independent_calibration_score_and_cut_replays']+=1
        checks['independent_selection_score_and_rank_replays']+=1
        errors=[]
        for sid in single_subset:
            path=source/'test/features/corrected'/(sid.replace(':','__')+'.npz')
            with np.load(path,allow_pickle=False) as z:
                allowed=[]
                if arm in ['clip','both']:allowed+=['patches','patch_geometry','patch_valid','word_text','word_image']
                if arm in ['blip','both']:allowed+=['queries']
                arrays={k:(z[k][:196].copy() if k in ['patches','patch_geometry','patch_valid'] else z[k]) for k in allowed}
                geometry=z['word_geometry']
            observation=dict(texts=[r['raw_text'] for r in readings[sid]],geometry=geometry,
                scalars=((scalar[sid]-mean)/scale).astype(np.float32),**arrays)
            # Independently decode the normalized paired outputs from saved checkpoint metadata.
            with torch.inference_mode():pred=model(collate_observations([observation],arm,'cpu'))[0].numpy()
            settings=checkpoint['inference_settings'];pred=pred*settings['target_scale']+np.asarray(settings['target_mean'])
            value=float(2*pred[0]-settings['reference_f1']*pred[1])
            expected=scores[arm][sid]
            if expected is not None:errors.append(abs(value-expected));assert abs(value-expected)<2e-4
            mp=Path(map_trace[sid]['maps_path'])
            if sha(mp)!=map_trace[sid]['maps_sha256']:raise ValueError('Changed predicted map')
            with np.load(mp,allow_pickle=False) as z:probabilities=z['probabilities']
            deployed=predict_encoder_gate(model,checkpoint,choice['policies'],sid,readings[sid],records[sid]['image_wh'],
                probabilities,original_settings,context_arrays=arrays,device='cpu')
            if expected is None:assert deployed['predicted_utility'] is None
            else:assert abs(deployed['predicted_utility']-expected)<2e-4
            for policy in result['arms'][arm]['policies']:
                assert deployed['use_unet'][str(policy['target_calibration_acceptance'])]==(sid in policy['accepted_ids'])
                checks['cpu_single_image_decisions']+=1
            checks['cpu_feature_and_score_replay']+=1
        maximum_error[arm]=max(errors,default=0.)
        for policy,calpolicy in zip(result['arms'][arm]['policies'],choice['policies']):
            assert policy['threshold']==calpolicy['threshold'] and policy['tie_sample_id']==calpolicy['tie_sample_id']
            keep=[s for s,v in scores[arm].items() if v is not None and (v>policy['threshold'] or v==policy['threshold'] and s<=policy['tie_sample_id'])]
            assert_counts(counts(branches,ids,keep),policy);checks['independent_test_routes']+=1
        curves[arm]=[]
        for fraction in [0,.1,.2,.3,.4,.45,.5,.55,.6,.7,.8,.9,1.]:
            order=sorted((s for s in ids if scores[arm][s] is not None),key=lambda s:(-scores[arm][s],s))
            keep=order[:round(len(ids)*fraction)];point=counts(branches,ids,keep)
            curves[arm].append(dict(target_unet_fraction=fraction,**point))
            if fraction==.5:assert_counts(point,result['arms'][arm]['exact50'])
            checks['quota_curve_points']+=1
        primary=next(p for p in result['arms'][arm]['policies'] if p['target_calibration_acceptance']==.5)
        keep=set(primary['accepted_ids']);triples=np.asarray([[branches[s]['light' if s in keep else 'deepsolo'][k] for k in ['tp','fp','fn']] for s in ids])
        rng=np.random.default_rng(c['seed']);values=[]
        for _ in range(c['bootstrap_draws']):
            tp,fp,fn=triples[rng.integers(len(ids),size=len(ids))].sum(0);values.append(2*tp/(2*tp+fp+fn))
        intervals[arm]=dict(f1_bootstrap95=np.quantile(values,[.025,.975]).tolist(),deepsolo_images=primary['deepsolo_images'],
            total_images=len(ids),deepsolo_percent=primary['deepsolo_percent'])
    best=max(v['selection_summary']['f1'] for v in frozen['arm_choices'].values())
    candidates=[v for v in frozen['arm_choices'].values() if v['selection_summary']['f1']>=best-c['selection_tolerance']]
    chosen=min(candidates,key=lambda v:(v['frozen_encoder_bytes']+v['head_tensor_bytes'],-v['selection_summary']['f1'],v['arm']))['arm']
    assert chosen==frozen['primary_arm']==result['primary_arm'];checks['validation_cost_selection']+=1
    report=dict(status='PASS',mode='approximate',checks=dict(checks),results_sha256=sha(root/'results.json'),
        max_cpu_single_gpu_batch_score_error=maximum_error,intervals=intervals,quota_curves=curves,
        bootstrap_scope='Fixed frozen routes, image resampling; excludes training and model-selection uncertainty.',
        primary_arm=chosen,analysis_code_sha256=sha(__file__),
        scope='Authenticated sources, disjoint fit/selection/CAL/TEST, FIT-only normalization, earliest best checkpoint and encoder cost selection, independent count sums, CPU prediction-only feature/score/deployment replay on12 TEST samples per arm. No independent neural refit.')
    with args.output.open('x') as handle:json.dump(report,handle,indent=2);handle.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k!='quota_curves'}))


if __name__=='__main__':main()
