"""Validation-only pooled light-word classifier, with native cut retained as control."""

import argparse
from pathlib import Path
import shutil

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from scripts.run_uniform_unet_training import read,write
from scripts.train_uniform_word_union import data
from segmentation.uniform_unet_fusion import file_hash,aggregate_counts
from scripts.audit_uniform_risk_fusion import count
from recognition.common_budget_pipeline import word_features
from recognition.uniform_lexical_risk import lexicon,prepare_words
from recognition.fixed_expert_budget import training_word_labels
from recognition.uniform_word_admission import admit,select_threshold
from recognition.uniform_pooled_gate import domain_weights,macro_lower_gain


def run(path,audit=False):
    c=read(path)
    if c['mode']!='approximate' or path.resolve().parent.name!='approximate':raise ValueError('Explicit approximate word-admission profile required')
    for name,digest in c['pinned_inputs'].items():
        if file_hash(name)!=digest:raise ValueError('Changed word-admission input: '+name)
    root=Path(c['output_root'])
    if not audit:
        root.mkdir(parents=True,exist_ok=False);write(root/'resolved_configuration.json',c)
        sources={};repo=Path(__file__).resolve().parents[1]
        for package in c['snapshot_packages']:
            for src in (repo/package).rglob('*.py'):
                rel=src.relative_to(repo);dest=root/'code_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(src,dest);sources[str(rel)]=file_hash(dest)
        write(root/'source_manifest.json',sources)
    base=read(c['base_pipeline_config']);domains=data(base);x,y,domain=[],[],[];scenes={};baseline={};base_triples={};policies={}
    protected={r['rgb_sha256'] for v in domains.values() for r in v['plan']['test_records']}
    protected.update(v['scenes'][s]['rgb_sha256'] for v in domains.values() for s in v['plan']['partitions']['selection'])
    used=set();fit_ids={}
    for ds,v in domains.items():
        original=read(Path(base['output_root'])/ds/'selection_frozen.json');q=c['hard_budgets'][ds]['maximum_fallback_images'];gate=original['policies'][str(q)]
        policy=dict(original['lexical_policy'],minimum_sequence_probability=c['minimum_sequence_probability']);policies[ds]=policy
        dictionary=lexicon(base['lexicons'][ds],'train');scenes[ds]={};fit_ids[ds]=[];baseline[ds]=[]
        for sid in v['plan']['partitions']['fit']+v['plan']['partitions']['selection']:
            s=v['scenes'][sid];fit=sid in v['plan']['partitions']['fit']
            prepared,trace=prepare_words(s['raw'],dictionary,policy);features=word_features(s['raw'],trace,s['image_wh'])
            labels=training_word_labels(s['raw'],prepared,s['truth']);keep=np.logical_not(s['flags'])
            scenes[ds][sid]=dict(source=s,prepared=prepared,x=features)
            if fit:
                rgb=s['rgb_sha256']
                if rgb in used or rgb in protected:continue
                used.add(rgb);fit_ids[ds].append(sid);x.append(features[keep]);y.append(labels[keep]);domain.extend([ds]*int(keep.sum()))
            else:
                old=[r for r in s['light'] if r['word_probability']>=gate['admission_threshold']]
                baseline[ds].append(count(old,s['truth'],[s['flags'][int(r['source_row'])] for r in old]))
        base_triples[ds]=np.array([[v[k] for k in ('tp','fp','fn')] for v in baseline[ds]])
    xx,yy=np.concatenate(x),np.concatenate(y);weights=domain_weights(domain,np.ones(len(domain)))
    if len(np.unique(yy))!=2:raise ValueError('Both word-correctness classes required')
    baselines={ds:aggregate_counts(v) for ds,v in baseline.items()};trials=[]
    for j,recipe in enumerate(c['word_recipes']):
        model=HistGradientBoostingClassifier(max_iter=recipe['iterations'],max_leaf_nodes=recipe['leaves'],
            min_samples_leaf=recipe['minimum_leaf'],l2_regularization=recipe['l2'],early_stopping=False,random_state=c['seed']).fit(xx,yy,sample_weight=weights)
        name=f'pooled_word_{j}.joblib'
        if audit:
            saved=joblib.load(root/name);np.testing.assert_allclose(saved.predict_proba(xx),model.predict_proba(xx),rtol=0,atol=1e-12)
        else:joblib.dump(model,root/name)
        results={};triples={};grids={}
        for ds,v in domains.items():
            ids=v['plan']['partitions']['selection'];predicted={sid:(model.predict_proba(scenes[ds][sid]['x'])[:,1] if len(scenes[ds][sid]['x']) else []) for sid in ids};grid=[]
            for threshold in c['admission_thresholds']:
                values=[]
                for sid in ids:
                    s=scenes[ds][sid];raw=s['source'];rows=admit(s['prepared'],predicted[sid],threshold)
                    values.append(count(rows,raw['truth'],[raw['flags'][int(r['source_row'])] for r in rows]))
                grid.append(dict(threshold=threshold,counts=aggregate_counts(values),triples=[[v[k] for k in ('tp','fp','fn')] for v in values]))
            selected=select_threshold(grid,c['selection_tolerance']);triples[ds]=np.asarray(selected['triples']);grids[ds]=grid
            results[ds]=dict(threshold=selected['threshold'],counts=selected['counts'],deepsolo_images=0,total_images=len(ids),deepsolo_percentage=0.)
        lower=macro_lower_gain(base_triples,triples,c['seed'],c['bootstrap_draws'],c['bootstrap_alpha'])
        gains={ds:results[ds]['counts']['f1']-baselines[ds]['f1'] for ds in results}
        eligible=lower>0 and min(gains.values())>=-c['maximum_validation_drop']
        trials.append(dict(model=name,recipe=recipe,results=results,grids=grids,macro_lower_gain=lower,validation_gains=gains,eligible=eligible))
        print('WORD',j,{ds:round(v['counts']['f1'],5) for ds,v in results.items()},'lower',round(lower,5),'eligible',eligible,flush=True)
    eligible=[t for t in trials if t['eligible']];winner=max(eligible,key=lambda t:np.mean(list(t['validation_gains'].values()))) if eligible else None
    result=dict(status='FROZEN',mode='approximate',claim_label='approximate reproduction',baseline={ds:dict(counts=v,deepsolo_images=0,total_images=v['images'],deepsolo_percentage=0.) for ds,v in baselines.items()},
        trials=trials,selected_model=winner['model'] if winner else None,policies=policies,fit_ids=fit_ids,fit_words=len(yy),
        config_sha256=file_hash(path),source_sha256=file_hash(__file__),test_used_for_selection=False,actual_new_neural_calls=0,
        scope='Conditional decoder-heldout FIT word classifier; native validation; no new TEST output or annotation read. Upstream exposure/nonnested stacking and reused validation remain.')
    if audit:
        assert result==read(root/'selection_frozen.json')
        write(root/'independent_refit_audit.json',dict(status='PASS',models_refitted=len(trials),fit_words=len(yy),frozen_sha256=file_hash(root/'selection_frozen.json'),scope='Independent model refits and all validation counts; shared feature/count primitives.'))
    else:write(root/'selection_frozen.json',result)
    print('WORD ADMISSION','AUDIT PASS' if audit else 'FROZEN','selected',result['selected_model'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--audit',action='store_true');a=p.parse_args()
    with threadpool_limits(limits=2):run(a.config,a.audit)
