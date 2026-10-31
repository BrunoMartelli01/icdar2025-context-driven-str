"""Fit/audit one pooled word fusion model with unchanged qualified image gates."""

import argparse
from collections import Counter
import copy
from pathlib import Path
import shutil

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from scripts.run_uniform_unet_training import read,write
from scripts.icdar_budget_data import predictions
from scripts.audit_uniform_risk_fusion import count
from segmentation.uniform_unet_fusion import annotation,file_hash,aggregate_counts,normalize
from recognition.uniform_observation_bank import observation_sources
from recognition.uniform_lexical_risk import lexicon,prepare_words
from recognition.fixed_expert_budget import fixed_words_with_risk
from recognition.uniform_risk_fusion import combine
from recognition.uniform_utility_fusion import paired_f1_lower
from recognition.uniform_word_union import candidates,training_labels,select,FEATURE_SCHEMA


def data(c):
    base=read(c['base_utility_config']); result={}
    for ds in c['datasets']:
        source=Path(c['training_root'])/ds;plan=read(source/'resolved_configuration.json')
        parent=Path(base['output_root'])/ds;frozen=read(parent/'selection_frozen.json')
        audit=read(parent/'independent_audit.json')
        assert audit['status']=='PASS' and audit['frozen_sha256']==file_hash(parent/'selection_frozen.json')
        for name,digest in frozen['model_sha256'].items():assert file_hash(parent/name)==digest
        index,_=observation_sources(base,ds,source,plan);probs=read(parent/'word_probabilities.json')
        dictionary=lexicon(base['lexicons'][ds],'train');lookup={r['sample_id']:r for r in plan['train_records']}
        teacher=read(base['fallback_train_sources'][ds])['native_benchmark'];scenes={}
        for sid in plan['partitions']['fit']+plan['partitions']['selection']:
            e=index[sid];assert file_hash(e['path'])==e['sha256'];sample=read(e['path']);assert not sample['ground_truth_used']
            truth,valid=annotation(lookup[sid]);flags=[r['ignored'] for r in predictions(sample['raw_rows'],lookup[sid],valid)]
            light,trace=prepare_words(sample['raw_rows'],dictionary,frozen['lexical_policy']);light=fixed_words_with_risk(light,probs[sid])
            heavy=teacher[sid]['rows'];rows,x=candidates(light,heavy,sample['image_wh'],c['learned_word_union']['matching_iou'])
            ignored=[flags[int(r['source_row'])] if r['union_source']=='light' else r['ignored'] for r in rows]
            labels=training_labels(rows,truth,ignored)
            assert sum(labels)==sum((Counter(normalize(r['raw_text']) for r,f in zip(rows,ignored) if not f and normalize(r['raw_text'])) & Counter(normalize(g['text']) for g in truth if normalize(g['text']))).values())
            scenes[sid]=dict(raw=sample['raw_rows'],image_wh=sample['image_wh'],light=light,trace=trace,probabilities=probs[sid],heavy=heavy,truth=truth,flags=flags,
                rows=rows,x=x,y=labels,ignored=ignored,rgb_sha256=lookup[sid]['rgb_sha256'])
        result[ds]=dict(plan=plan,parent=parent,frozen=frozen,scenes=scenes)
    return result


def matrices(c,domains):
    protected={v['scenes'][s]['rgb_sha256'] for v in domains.values() for s in v['plan']['partitions']['selection']}
    # Also exclude every official TEST RGB, even if a future corpus reuses it in FIT.
    for ds in c['datasets']:
        p=read(Path(c['training_root'])/ds/'resolved_configuration.json');protected.update(r['rgb_sha256'] for r in p['test_records'])
    blocks=[];labels=[];sizes=[];used=set();fit_ids={}
    for ds,v in domains.items():
        xx=[];yy=[];fit_ids[ds]=[]
        for sid in v['plan']['partitions']['fit']:
            s=v['scenes'][sid];rgb=s['rgb_sha256']
            if rgb in protected or rgb in used:continue
            used.add(rgb);fit_ids[ds].append(sid);keep=np.logical_not(s['ignored'])
            xx.append(s['x'][keep]);yy.append(s['y'][keep])
        x=np.concatenate(xx);y=np.concatenate(yy);blocks.append(x);labels.append(y);sizes.append(len(y))
    total=sum(sizes);weights=np.concatenate([np.full(n,total/(len(sizes)*n)) for n in sizes])
    return np.concatenate(blocks),np.concatenate(labels),weights,fit_ids


def validation(c,v,model):
    ids=v['plan']['partitions']['selection'];scenes=v['scenes'];settings=c['learned_word_union'];policies={};trials={}
    probabilities={s:model.predict_proba(scenes[s]['x'])[:,1] if len(scenes[s]['x']) else [] for s in ids}
    bootstrap=np.random.default_rng(c['seed']).integers(len(ids),size=(c['bootstrap_draws'],len(ids)))
    for q,gate in v['frozen']['policies'].items():
        accepted=set(gate['accepted_validation_ids']);baseline={};base_rows={}
        for sid in ids:
            s=scenes[sid];rows=[r for r in s['light'] if r['word_probability']>=gate['admission_threshold']]
            observation=dict(rows=rows,probabilities=s['probabilities'],trace=s['trace'])
            rows=rows if sid in accepted else combine(observation,s['heavy'],gate['fusion'],gate['complement'])
            flags=[s['flags'][int(r['source_row'])] if sid in accepted or r.get('fixed_light_complement') else r['ignored'] for r in rows]
            base_rows[sid]=rows;baseline[sid]=count(rows,s['truth'],flags)
        base=np.array([[baseline[s][k] for k in ['tp','fp','fn']] for s in ids]);base_aggregate=aggregate_counts(baseline.values())
        assert base_aggregate==gate['counts']
        candidates_trial=[dict(threshold=None,counts=base_aggregate,changed_words=0,lower_gain=0.,eligible=True)]
        for threshold in settings['thresholds']:
            counts=[];changes=0
            for sid in ids:
                s=scenes[sid]
                if sid in accepted:counts.append(baseline[sid]);continue
                rows=select(s['rows'],probabilities[sid],threshold,settings['conflict_iou'])
                flags=[s['flags'][int(r['source_row'])] if r.get('fixed_light_complement') else r['ignored'] for r in rows]
                counts.append(count(rows,s['truth'],flags))
                a=Counter(normalize(r['raw_text']) for r in rows);b=Counter(normalize(r['raw_text']) for r in base_rows[sid]);changes+=sum((a-b).values())+sum((b-a).values())
            lower=paired_f1_lower(base,np.array([[r[k] for k in ['tp','fp','fn']] for r in counts]),bootstrap,c['bootstrap_alpha'])
            candidates_trial.append(dict(threshold=threshold,counts=aggregate_counts(counts),changed_words=changes,lower_gain=lower,eligible=lower>0.))
        eligible=[t for t in candidates_trial if t['eligible']];best=max(t['counts']['f1'] for t in eligible)
        choice=min((t for t in eligible if t['counts']['f1']>=best-c['selection_tolerance']),key=lambda t:(t['changed_words'],-t['counts']['f1']))
        policies[q]=dict(gate,learned_union_threshold=choice['threshold'],counts=choice['counts'],union_paired_lower_gain=choice['lower_gain'])
        trials[q]=candidates_trial
    return policies,trials


def run(path,audit):
    c=read(path)
    if c['mode']!='approximate' or path.resolve().parent.name!='approximate':raise ValueError('Explicit approximate union profile required')
    if c['learned_word_union'].get('feature_schema')!=FEATURE_SCHEMA:raise ValueError('Explicit compatible word-union feature schema required')
    for p,h in c['pinned_inputs'].items():
        if file_hash(p)!=h:raise ValueError('Changed fusion source: '+p)
    domains=data(c);x,y,weights,fit_ids=matrices(c,domains)
    settings=c['union_classifier']
    model=HistGradientBoostingClassifier(max_iter=settings['iterations'],max_leaf_nodes=settings['leaves'],
        min_samples_leaf=settings['minimum_leaf'],l2_regularization=settings['l2'],early_stopping=False,
        random_state=c['seed']).fit(x,y,sample_weight=weights)
    root=Path(c['output_root']);global_model=root/'pooled_union_model.joblib'
    if audit:
        saved=joblib.load(global_model);np.testing.assert_allclose(saved.predict_proba(x),model.predict_proba(x),rtol=0,atol=1e-12)
    else:
        root.mkdir(parents=True,exist_ok=False);joblib.dump(model,global_model);write(root/'fit_scope.json',dict(fit_ids=fit_ids,rows=len(y),class_counts=np.bincount(y).tolist(),equal_domain_weights=True,protected_rgb_excluded=True))
    for ds,v in domains.items():
        policies,trials=validation(c,v,model);out=root/ds
        if audit:
            frozen=read(out/'selection_frozen.json');assert frozen['policies']==policies and read(out/'union_trials.json')==trials
            for name,digest in frozen['model_sha256'].items():assert file_hash(out/name)==digest
            assert file_hash(out/'union_model.joblib')==file_hash(global_model)
            write(out/'independent_audit.json',dict(status='PASS',frozen_sha256=file_hash(out/'selection_frozen.json'),source_sha256=file_hash(__file__),
                scope='Refit shared pooled classifier and all validation thresholds; inherited independently qualified unchanged image/word heads; independent Counter metric'))
        else:
            out.mkdir();
            for p in v['parent'].glob('*.joblib'):shutil.copyfile(p,out/p.name)
            shutil.copyfile(global_model,out/'union_model.joblib');write(out/'union_trials.json',trials)
            frozen=copy.deepcopy(v['frozen']);frozen.update(policies=policies,config_sha256=file_hash(path),model_sha256={p.name:file_hash(p) for p in out.glob('*.joblib')},
                base_selection_sha256=file_hash(v['parent']/'selection_frozen.json'),union_fit_ids=fit_ids,source_sha256=file_hash(__file__),test_used_for_selection=False)
            write(out/'selection_frozen.json',frozen)
    print('WORD UNION', 'AUDIT PASS' if audit else 'FROZEN', 'all3 domains;',len(y),'pooled word candidates',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--audit',action='store_true');a=p.parse_args()
    with threadpool_limits(limits=2):run(a.config,a.audit)
