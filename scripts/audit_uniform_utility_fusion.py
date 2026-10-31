"""Refit compact models and independently recount the common utility study."""

import argparse
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from scripts.run_uniform_unet_training import read, write
from scripts.icdar_budget_data import key, predictions
from scripts.audit_uniform_risk_fusion import count, normalize
from segmentation.uniform_unet_fusion import annotation, file_hash, aggregate_counts
from recognition.uniform_lexical_risk import lexicon, prepare_words
from recognition.uniform_utility_fusion import image_vector, paired_f1_lower
from recognition.common_budget_pipeline import word_features, fit_word_risk, fit_count_head, predict_counts
from recognition.uniform_risk_fusion import combine
from recognition.uniform_observation_bank import observation_sources, observation_map_path
from recognition.uniform_encoder_context import training_context


def audit(config_path, dataset):
    c=read(config_path); source=Path(c['training_root'])/dataset; out=Path(c['output_root'])/dataset
    plan=read(source/'resolved_configuration.json'); frozen=read(out/'selection_frozen.json')
    assert frozen['status']=='FROZEN' and not frozen['test_used_for_selection']
    assert frozen['config_sha256']==file_hash(config_path)
    for name,digest in frozen['model_sha256'].items(): assert file_hash(out/name)==digest
    policy=frozen['lexical_policy']; dictionary=lexicon(c['lexicons'][dataset],'train')
    index,expert_condition=observation_sources(c,dataset,source,plan)
    if 'fitting_observations' in frozen:assert frozen['fitting_observations']==expert_condition
    lookup={r['sample_id']:r for r in plan['train_records']}
    fit,val=plan['partitions']['fit'],plan['partitions']['selection']; ids=fit+val
    context=training_context(c,{s:lookup[s] for s in ids})
    scenes,truth,flags,prepared,xword,yword,maps={},{},{},{},{},{},{}
    for sid in ids:
        e=index[sid];assert file_hash(e['path'])==e['sha256'];o=read(e['path']);scenes[sid]=o
        assert not o['ground_truth_used'] and o['rgb_sha256']==lookup[sid]['rgb_sha256'] and o['image_wh']==lookup[sid]['image_wh']
        truth[sid],valid=annotation(lookup[sid]);flags[sid]=[r['ignored'] for r in predictions(o['raw_rows'],lookup[sid],valid)]
        prepared[sid],trace=prepare_words(o['raw_rows'],dictionary,policy)
        xword[sid]=word_features(o['raw_rows'],trace,o['image_wh'])
        words={normalize(g['text']) for g in truth[sid] if normalize(g['text'])}
        chosen={int(r['source_row']):r for r in prepared[sid]}
        yword[sid]=np.array([int(bool(normalize(chosen.get(i,r)['raw_text'])) and normalize(chosen.get(i,r)['raw_text']) in words) for i,r in enumerate(o['raw_rows'])])
        mp=observation_map_path(source,sid,o);assert file_hash(mp)==o['maps_sha256']
        with np.load(mp,allow_pickle=False) as z:maps[sid]=z['probabilities']
    folds=read(out/'word_folds.json')['folds'];probabilities=read(out/'word_probabilities.json')
    assert sorted(s for f in folds for s in f['heldout_ids'])==sorted(fit)
    for i,(samples,held) in enumerate([(f['fit_ids'],f['heldout_ids']) for f in folds]+[(fit,val)]):
        if i<len(folds):
            assert not set(samples)&set(held) and set(samples)|set(held)==set(fit)
            assert file_hash(out/f'word_fold_{i}.joblib')==folds[i]['sha256']
        xx=np.concatenate([xword[s][np.logical_not(flags[s])] for s in samples]);yy=np.concatenate([yword[s][np.logical_not(flags[s])] for s in samples])
        model=fit_word_risk(xx,yy,c['seed']+i if i<len(folds) else c['seed'],c['word_risk'])
        for sid in held:
            got=model.predict_proba(xword[sid])[:,1] if len(xword[sid]) else []
            np.testing.assert_allclose(got,probabilities[sid],rtol=0,atol=1e-12)
    fallback=read(c['fallback_train_sources'][dataset])['native_benchmark'];teacher={s:fallback[s]['counts'] for s in ids}
    reference=aggregate_counts(teacher[s] for s in fit)['f1'];scales={s:len(scenes[s]['raw_rows'])+1 for s in ids}
    matrices,observations,light={}, {}, {}
    count_refits=0
    for ai,admission in enumerate(c['admission_thresholds']):
        obs={};matrix={};lc={}
        for sid in ids:
            p=probabilities[sid];rows=[dict(r,word_probability=float(p[int(r['source_row'])])) for r in prepared[sid] if p[int(r['source_row'])]>=admission]
            _,trace=prepare_words(scenes[sid]['raw_rows'],dictionary,policy);kept={int(r['source_row']) for r in rows}
            trace=[dict(t,accepted=i in kept,changed=i in kept and t['changed']) for i,t in enumerate(trace)]
            obs[sid]=dict(rows=rows,probabilities=p,trace=trace)
            matrix[sid]=image_vector(scenes[sid]['raw_rows'],scenes[sid]['image_wh'],maps[sid],obs[sid],c['map_settings'],
                context[sid] if context is not None else None)
            lc[sid]=count(rows,truth[sid],[flags[sid][int(r['source_row'])] for r in rows])
        observations[admission],matrices[admission],light[admission]=obs,matrix,lc
        xx=np.stack([matrix[s] for s in fit]);yy=np.array([[lc[s]['tp']-teacher[s]['tp'],lc[s]['detected_words']-teacher[s]['detected_words']] for s in fit])/np.array([scales[s] for s in fit])[:,None]
        recipes=c['count_recipes'] if 'count_recipes' in c else [dict(kind='ridge',alpha=a) for a in c['ridge_alphas']]
        for ri,recipe in enumerate(recipes):
            rebuilt=fit_count_head(xx,yy,np.array([scales[s] for s in fit]),recipe,c['seed'])
            saved=joblib.load(out/f'count_{ai}_{ri}.joblib')
            np.testing.assert_allclose(predict_counts(rebuilt,xx),predict_counts(saved,xx),rtol=0,atol=1e-10)
            count_refits+=1
    gate_winners={};gate_replays=0
    for trial in read(out/'gate_trials.json'):
        admission=trial['admission_threshold'];obs=observations[admission];matrix=matrices[admission]
        bundle=joblib.load(out/trial['model']);pairs=predict_counts(bundle,np.stack([matrix[s] for s in val]))
        scores={s:(2*p[0]-reference*p[1])*scales[s] for s,p in zip(val,pairs)}
        quota=trial['test_fallback_images'];total=len(plan['partitions']['test'])
        if 'hard_budgets' in c:
            contract=c['hard_budgets'][dataset]
            assert total==contract['reference_images'] and quota<=contract['maximum_fallback_images'] and contract['batch_rounding']=='floor'
            k=quota*len(val)//total
        else:k=round(len(val)*quota/total)
        assert trial['deepsolo_images']==k and trial['total_images']==len(val)
        assert abs(trial['deepsolo_percentage']-100*k/len(val))<1e-12
        eligible=[s for s in val if obs[s]['rows']] if trial['guard'] else val
        accepted=set(sorted(eligible,key=lambda s:(-scores[s],lookup[s]['rgb_sha256'],s))[:len(val)-k])
        assert len(accepted)==len(val)-k and accepted==set(trial['accepted_validation_ids'])
        counts=aggregate_counts(light[admission][s] if s in accepted else teacher[s] for s in val)
        assert counts==trial['counts'];gate_replays+=len(val)
        if quota not in gate_winners or counts['f1']>gate_winners[quota]['counts']['f1']:gate_winners[quota]=trial
    indices=np.random.default_rng(c['seed']).integers(len(val),size=(c['bootstrap_draws'],len(val)))
    fusion_replays=0
    for q,trials in read(out/'fusion_trials.json').items():
        gate=gate_winners[int(q)];obs=observations[gate['admission_threshold']];accepted=set(gate['accepted_validation_ids'])
        baseline={s:light[gate['admission_threshold']][s] if s in accepted else teacher[s] for s in val}
        base=np.array([[baseline[s][k] for k in ('tp','fp','fn')] for s in val])
        for t in trials:
            counts=[]
            for sid in val:
                if sid in accepted:counts.append(baseline[sid]);continue
                rows=combine(obs[sid],fallback[sid]['rows'],t['fusion'],t['complement'])
                ff=[flags[sid][int(r['source_row'])] if r.get('fixed_light_complement') else r['ignored'] for r in rows]
                counts.append(count(rows,truth[sid],ff))
            assert aggregate_counts(counts)==t['counts'];fusion_replays+=len(val)
            identity=not t['fusion']['enabled'] and t['complement']['confidence'] is None
            lower=0. if identity else paired_f1_lower(base,np.array([[r[k] for k in ('tp','fp','fn')] for r in counts]),indices,c['bootstrap_alpha'])
            assert abs(lower-t['paired_lower_gain'])<1e-12 and t['eligible']==(identity or lower>0)
        eligible=[t for t in trials if t['eligible']];best=max(t['counts']['f1'] for t in eligible)
        chosen=min((t for t in eligible if t['counts']['f1']>=best-c['selection_tolerance']),key=lambda t:(t['changed_words'],-t['counts']['f1']))
        selected=frozen['policies'][q]
        assert selected['model']==gate['model'] and selected['admission_threshold']==gate['admission_threshold']
        assert selected['fusion']==chosen['fusion'] and selected['complement']==chosen['complement'] and selected['counts']==chosen['counts']
    write(out/'independent_audit.json',dict(status='PASS',word_model_refits=len(folds)+1,count_head_refits=count_refits,
        gate_image_replays=gate_replays,fusion_image_replays=fusion_replays,
        frozen_sha256=file_hash(out/'selection_frozen.json'),source_sha256=file_hash(__file__),
        scope='Independent Counter and independent ranking; exact model refits and paired-bound reconstruction; shared feature/geometry primitives'))
    print('UTILITY AUDIT PASS',dataset,count_refits,'count refits',gate_replays+fusion_replays,'image replays',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--dataset',choices=['icdar13','icdar2015','totaltext'],required=True)
    a=p.parse_args()
    with threadpool_limits(limits=2):audit(a.config,a.dataset)
