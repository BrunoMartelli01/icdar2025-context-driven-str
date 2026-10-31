"""Validation-only pooled image-router study with explicit hard fallback budgets."""

import argparse
from pathlib import Path
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from scripts.run_uniform_unet_training import read,write
from scripts.train_uniform_word_union import data
from scripts.audit_uniform_risk_fusion import count
from segmentation.uniform_unet_fusion import file_hash,aggregate_counts
from recognition.uniform_observation_bank import observation_sources,observation_map_path
from recognition.uniform_encoder_context import training_context
from recognition.uniform_utility_fusion import image_vector
from recognition.uniform_word_union import select
from recognition.uniform_risk_fusion import combine,route
from recognition.common_budget_pipeline import fit_count_head,predict_counts
from recognition.hard_fallback_budget import validate_policy,batch_quota,verify_route
from recognition.uniform_pooled_gate import domain_weights,macro_lower_gain


def checked(path):
    c=read(path)
    if path.resolve().parent.name!='approximate' or c['mode']!='approximate':
        raise ValueError('Explicit approximate pooled-gate study required')
    for name,digest in c['pinned_inputs'].items():
        if file_hash(name)!=digest:raise ValueError('Changed pooled-gate input: '+name)
    return c


def prepare(path):
    c=checked(path);out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False)
    write(out/'resolved_configuration.json',c)
    snapshot={}
    for package in ('recognition','segmentation','scripts','models','localize','pipeline','scoring'):
        for src in (Path(__file__).resolve().parents[1]/package).rglob('*.py'):
            relative=src.relative_to(Path(__file__).resolve().parents[1]);target=out/'code_snapshot'/relative
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(src.read_bytes());snapshot[str(relative)]=file_hash(target)
    write(out/'source_manifest.json',snapshot)
    source=Path(c['base_pipeline_config']);base=read(source);domains=data(base)
    ancestor=read(base['base_utility_config']); union=joblib.load(Path(base['output_root'])/'pooled_union_model.joblib')
    protected={r['rgb_sha256'] for v in domains.values() for r in v['plan']['test_records']}
    protected.update(v['scenes'][s]['rgb_sha256'] for v in domains.values() for s in v['plan']['partitions']['selection'])
    x,y,scales,metadata,used=[],[],[],[],set(); evidence={}
    for ds,v in domains.items():
        plan=v['plan'];root=Path(base['training_root'])/ds;scenes=v['scenes'];q=c['hard_budgets'][ds]['maximum_fallback_images']
        contract=validate_policy(c['hard_budgets'],ds,q)
        if contract['reference_images']!=len(plan['partitions']['test']):raise ValueError('Changed hard-budget population')
        frozen=read(Path(base['output_root'])/ds/'selection_frozen.json');gate=frozen['policies'][str(q)]
        ids=plan['partitions']['fit']+plan['partitions']['selection'];lookup={r['sample_id']:r for r in plan['train_records']}
        context=training_context(base,{s:lookup[s] for s in ids}); index,_=observation_sources(base,ds,root,plan)
        head=joblib.load(Path(base['output_root'])/ds/gate['model'])
        for sid in ids:
            s=scenes[sid];fit=sid in plan['partitions']['fit'];rgb=s['rgb_sha256']
            if fit and (rgb in protected or rgb in used):continue
            if fit:used.add(rgb)
            p=s['probabilities'];light=[r for r in s['light'] if r['word_probability']>=gate['admission_threshold']]
            kept={int(r['source_row']) for r in light}
            trace=[dict(t,accepted=i in kept,changed=i in kept and t['changed']) for i,t in enumerate(s['trace'])]
            observation=dict(rows=light,probabilities=p,trace=trace)
            sample=read(index[sid]['path']);mp=observation_map_path(root,sid,sample)
            if file_hash(mp)!=sample['maps_sha256']:raise ValueError('Changed actual predicted map')
            evidence[str(mp)]=sample['maps_sha256']
            with np.load(mp,allow_pickle=False) as z:
                vector=image_vector(s['raw'],s['image_wh'],z['probabilities'],observation,base['map_settings'],context[sid])
            def recount(rows):
                flags=[s['flags'][int(r['source_row'])] if r.get('fixed_light_complement') else r['ignored'] for r in rows]
                return count(rows,s['truth'],flags)
            lc=count(light,s['truth'],[s['flags'][int(r['source_row'])] for r in light])
            hc=count(s['heavy'],s['truth'],[r['ignored'] for r in s['heavy']])
            # Post-fallback fusion is unchanged; only validation counts enter selection.
            fused=None
            if not fit:
                if gate['learned_union_threshold'] is not None:
                    prob=union.predict_proba(s['x'])[:,1] if len(s['x']) else []
                    rows=select(s['rows'],prob,gate['learned_union_threshold'],base['learned_word_union']['conflict_iou'])
                else:rows=combine(observation,s['heavy'],gate['fusion'],gate['complement'])
                fused=recount(rows)
            scale=len(s['raw'])+1;pair=predict_counts(head,vector[None])[0]
            metadata.append(dict(dataset=ds,sample_id=sid,rgb_sha256=rgb,fit=fit,light=lc,fallback=fused,
                reference=gate['reference'],guard=gate['guard'],nonempty=bool(light),
                baseline_score=float((2*pair[0]-gate['reference']*pair[1])*scale)))
            x.append(vector);y.append([lc['tp']-hc['tp'],lc['detected_words']-hc['detected_words']]);scales.append(scale)
        print('Prepared',ds,'FIT/VAL',sum(m['fit'] and m['dataset']==ds for m in metadata),len(plan['partitions']['selection']),flush=True)
    np.savez_compressed(out/'features.npz',x=np.asarray(x),y=np.asarray(y),scales=np.asarray(scales))
    write(out/'samples.json',metadata);write(out/'feature_sources.json',evidence)
    write(out/'prepared.json',dict(status='COMPLETE',test_outputs_loaded=False,test_labels_used=False,
        features_sha256=file_hash(out/'features.npz'),samples_sha256=file_hash(out/'samples.json'),
        feature_sources_sha256=file_hash(out/'feature_sources.json'),source_sha256=file_hash(__file__),
        scope='Conditional decoder-heldout FIT; full-FIT validation; existing word probabilities and fusion. No unbiased end-to-end OOF claim.'))


def train(path,audit=False):
    started=time.monotonic();c=checked(path);root=Path(c['output_root']);ready=read(root/'prepared.json')
    if ready['source_sha256']!=file_hash(__file__):raise ValueError('Prepared study entrypoint changed')
    for name,field in [('features.npz','features_sha256'),('samples.json','samples_sha256')]:
        if file_hash(root/name)!=ready[field]:raise ValueError('Changed prepared pooled features')
    metadata=read(root/'samples.json')
    with np.load(root/'features.npz',allow_pickle=False) as z:x,y,scales=z['x'],z['y'],z['scales']
    fit=np.array([m['fit'] for m in metadata]);domains=np.array([m['dataset'] for m in metadata]);weights=domain_weights(domains[fit],scales[fit])
    def validation(scores):
        results={};triples={};decisions={}
        for ds,contract in c['hard_budgets'].items():
            ii=[i for i,m in enumerate(metadata) if not m['fit'] and m['dataset']==ds]
            ids=[metadata[i]['sample_id'] for i in ii];q=contract['maximum_fallback_images'];k=batch_quota(contract,q,contract['reference_images'],len(ids))
            observations={metadata[i]['sample_id']:dict(score=float(scores[i]),rows=[{}] if metadata[i]['nonempty'] else []) for i in ii}
            guards={metadata[i]['guard'] for i in ii}
            if len(guards)!=1:raise ValueError('Unchanged common guard required within each dataset')
            decision=route(observations,{metadata[i]['sample_id']:metadata[i]['rgb_sha256'] for i in ii},k,guards.pop())
            verify_route(decision,ids,contract,q);accepted=set(decision['accepted_ids'])
            values=[metadata[i]['light' if metadata[i]['sample_id'] in accepted else 'fallback'] for i in ii]
            triples[ds]=np.array([[v[t] for t in ('tp','fp','fn')] for v in values])
            results[ds]=dict(counts=aggregate_counts(values),deepsolo_images=k,total_images=len(ids),deepsolo_percentage=100*k/len(ids))
            decisions[ds]=decision
        return results,triples,decisions
    baseline,base,_=validation(np.array([m['baseline_score'] for m in metadata]));trials=[]
    for j,recipe in enumerate(c['count_recipes']):
        head=fit_count_head(x[fit],y[fit]/scales[fit,None],weights,recipe,c['seed'])
        name=f'pooled_count_{j}.joblib'
        if audit:
            saved=joblib.load(root/name);np.testing.assert_allclose(predict_counts(saved,x),predict_counts(head,x),rtol=0,atol=1e-12)
        else:joblib.dump(head,root/name)
        delta=predict_counts(head,x);scores=(2*delta[:,0]-np.array([m['reference'] for m in metadata])*delta[:,1])*scales
        result,triples,decisions=validation(scores)
        lower=macro_lower_gain(base,triples,c['seed'],c['bootstrap_draws'],c['bootstrap_alpha'])
        changes={ds:result[ds]['counts']['f1']-baseline[ds]['counts']['f1'] for ds in result}
        eligible=lower>0 and min(changes.values())>=-c['maximum_validation_drop']
        trials.append(dict(model=name,recipe=recipe,datasets=result,macro_lower_gain=lower,validation_changes=changes,eligible=eligible,decisions=decisions))
    eligible=[t for t in trials if t['eligible']]
    winner=max(eligible,key=lambda t:np.mean(list(t['validation_changes'].values()))) if eligible else None
    result=dict(status='FROZEN',mode='approximate',claim_label='approximate reproduction',baseline=baseline,trials=trials,
        selected_model=winner['model'] if winner else None,test_used_for_selection=False,
        hard_budgets=c['hard_budgets'],config_sha256=file_hash(path),source_sha256=file_hash(__file__),
        prepared_sha256=file_hash(root/'prepared.json'),fit_images=int(fit.sum()),validation_images=int((~fit).sum()),
        selected_policy='pooled' if winner else 'retain_existing_gate',additional_neural_models=0)
    if audit:
        assert result==read(root/'selection_frozen.json')
        write(root/'independent_refit_audit.json',dict(status='PASS',models_refitted=len(trials),
            frozen_sha256=file_hash(root/'selection_frozen.json'),scope='Full independent model refits and validation replay; shared feature/count primitives.',seconds=time.monotonic()-started))
    else:write(root/'selection_frozen.json',result)
    print('POOLED GATE', 'AUDIT PASS' if audit else 'FROZEN', 'selected',result['selected_policy'],
        [(t['model'],{d:round(v['counts']['f1'],5) for d,v in t['datasets'].items()},round(t['macro_lower_gain'],5),t['eligible']) for t in trials],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--stage',choices=['prepare','train','audit'],required=True);a=p.parse_args()
    with threadpool_limits(limits=2):
        prepare(a.config) if a.stage=='prepare' else train(a.config,audit=a.stage=='audit')
