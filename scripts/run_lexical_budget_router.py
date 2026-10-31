"""Select conservative light edits, fit target-utility heads and freeze budgets."""

import argparse
from fractions import Fraction
from pathlib import Path

import joblib
import numpy as np

from recognition.deepsolo_lexicon import DeepSoloLexicon
from recognition.lexical_budget_router import correct_rows,features,fit,predict
from recognition.icdar_budget_router import lexical_counts,calibration_cut,accept
from recognition.router_fallback_replay import replay
from scripts.icdar_budget_data import read,sha,key,annotation,predictions
from scripts.run_yolo_f1_quality_study import seal


def lexicon(c,ds,split):
    s=c['lexicons'][ds];v=s[split+'_vocabulary'];p=s[split+'_pairs']
    if split=='train' and s['train_pair_format']=='json':return DeepSoloLexicon(Path(v).read_text().splitlines(),read(p),distance_exclusive=1.5)
    return DeepSoloLexicon.from_files(v,p,distance_exclusive=1.5)


def corrected(scene,lex,policy):
    rows,trace=correct_rows(scene['rows'],lex,policy)
    counts=lexical_counts([dict(text=r['raw_text'],ignored=scene['ignored'][r['source_row']]) for r in rows],scene['ground_truth'])
    return rows,trace,counts


def fraction(p):return Fraction(2*p['tp'],p['detected_words']+p['gt_words'])


def order(scores,keys):return sorted(scores,key=lambda s:(-scores[s],keys[s],s))


def select(c):
    root=Path(c['study_root']);old=Path(c['original_router_root']);plan=read(old/'resolved_configuration.json');choices={};grids={}
    for ds in c['targets']:
        choices[ds]={};grids[ds]={};lex=lexicon(c,ds,'train')
        teacher=read(Path(c['teacher_study'])/'teacher'/ds/'counts.json')['samples']
        for arm in c.get('experts',('unet','yolo')):
            model=read(old/'frozen_choices.json')['choices'][ds][arm]['expert'];scenes={};pins={}
            for records in plan['records'][ds].values():
                for r in records:
                    p=old/'experts'/model/(key(r['sample_id'])+'.json');d=read(p)
                    assert d['image_sha256']==r['image_sha256'] and not d['ground_truth_used'];pins[str(p)]=sha(p)
                    gt,valid=annotation(r);pp=predictions(d['rows'],r,valid)
                    scenes[r['sample_id']]=dict(rows=d['rows'],ground_truth=gt,ignored=[p['ignored'] for p in pp],image_wh=r['image_wh'],tie_key=r['rgb_sha256'],teacher=teacher[r['sample_id']]['counts'])
            candidates=[dict(enabled=False,confidence_cap=0.,protect_numeric=c['protect_numeric'],unmatched_policy='keep_unmatched')]
            candidates += [dict(enabled=True,confidence_cap=cap,protect_numeric=c['protect_numeric'],unmatched_policy=unmatched) for cap in c['confidence_caps'] for unmatched in c['unmatched_policies']]
            if c.get('fixed_source_correction'):candidates=[p for p in candidates if p['enabled']]
            grid=[];ids=plan['partitions'][ds]['selection']
            for policy in candidates:
                counts=[];changes=0
                for sid in ids:
                    rows,trace,metrics=corrected(scenes[sid],lex,policy);counts.append(metrics);changes+=sum(t['changed'] or not t['accepted'] for t in trace)
                totals={k:sum(v[k] for v in counts) for k in counts[0]};totals['f1']=float(fraction(totals));grid.append(dict(policy=policy,metrics=totals,changes=changes))
            winner=max(grid,key=lambda x:(fraction(x['metrics']),-x['changes'],not x['policy']['enabled']))
            choices[ds][arm]=winner;grids[ds][arm]=grid
            seal(root/'prepared'/(ds+'__'+arm+'.json'),dict(scenes=scenes,input_sha256=pins,expert=model))
            print(ds,arm,'selected light correction',winner,flush=True)
    seal(root/'correction_grid.json',grids);seal(root/'correction_frozen.json',dict(choices=choices,test_used_for_selection=False,config_sha256=c['_source_config_sha256']))


def prepare_arrays(c,ds,arm):
    root=Path(c['study_root']);data=read(root/'prepared'/(ds+'__'+arm+'.json'));scenes=data['scenes'];policy=read(root/'correction_frozen.json')['choices'][ds][arm]['policy']
    lexicons={s:lexicon(c,ds,s) for s in ('train','test')};f={};branches={};scales={};keys={};observations={};extra=read(c['extra_feature_archive'])['values'] if c.get('extra_feature_archive') else None
    for sid,scene in scenes.items():
        rows,trace,counts=corrected(scene,lexicons['train' if ':train:' in sid else 'test'],policy)
        f[sid]=features(scene['rows'],rows,trace,scene['image_wh'],use_consensus=c.get('consensus_features',False));branches[sid]=dict(light=counts,deepsolo=scene['teacher']);scales[sid]=len(scene['rows'])+1;keys[sid]=scene['tie_key']
        if extra is not None:f[sid]['word_risk']=np.asarray(extra[sid],np.float64)
        observations[sid]=dict(rows=rows,trace=trace,counts=counts)
    return f,branches,scales,keys,observations


def train(c):
    root=Path(c['study_root']);plan=read(Path(c['original_router_root'])/'resolved_configuration.json');frozen={};trials={}
    for ds,t in c['targets'].items():
        frozen[ds]={};trials[ds]={};part=plan['partitions'][ds]
        for arm in c.get('experts',('unet','yolo')):
            f,b,scales,keys,obs=prepare_arrays(c,ds,arm);fitids=part['fit'];sel=part['selection'];cal=part['calibration'];trials[ds][arm]=[]
            def matrix(ids,groups):return np.stack([np.concatenate([f[s][g] for g in groups]) for s in ids])
            weights=np.array([scales[s] for s in fitids]);y=np.array([2*(b[s]['light']['tp']-b[s]['deepsolo']['tp'])-t['f1']*(b[s]['light']['detected_words']-b[s]['deepsolo']['detected_words']) for s in fitids])/weights
            acceptsel=len(sel)-round(len(sel)*t['fallback_images']/t['test_images'])
            for gi,groups in enumerate(c['group_sets']):
                x=matrix(fitids,groups)
                for ri,recipe in enumerate(c['recipes']):
                    bundle=fit(x,y,weights,recipe,c['seed']);bundle.update(groups=groups,fit_ids=fitids,target_f1=t['f1'],correction_policy=read(root/'correction_frozen.json')['choices'][ds][arm]['policy'])
                    values=predict(bundle,matrix(sel,groups),np.array([scales[s] for s in sel]));scores=dict(zip(sel,map(float,values)));selected=order(scores,keys)[:acceptsel];point=replay({s:b[s] for s in sel},selected)
                    path=root/'models'/(ds+'__'+arm+'__'+str(gi)+'__'+str(ri)+'.joblib');path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();joblib.dump(bundle,path)
                    trials[ds][arm].append(dict(path=str(path),sha256=sha(path),groups=groups,recipe=recipe,bytes=path.stat().st_size,selection=point))
            candidates=trials[ds][arm];best=max(p['selection']['f1'] for p in candidates);winner=min((p for p in candidates if p['selection']['f1']>=best-c['selection_tolerance']),key=lambda p:(p['bytes'],-p['selection']['f1']))
            bundle=joblib.load(winner['path']);values=predict(bundle,matrix(cal,bundle['groups']),np.array([scales[s] for s in cal]));scores=dict(zip(cal,map(float,values)))
            q=1-t['fallback_images']/t['test_images'];cut=calibration_cut(scores,cal,q,keys)
            frozen[ds][arm]=dict(winner=winner,calibration_cut=cut,calibration_scores=scores,calibration=replay({s:b[s] for s in cal},accept(scores,cut,keys)),fit_ids=fitids,selection_ids=sel,calibration_ids=cal)
            ids=list(f);np.savez_compressed(root/'models'/(ds+'__'+arm+'__features.npz'),ids=np.array(ids),scale=np.array([scales[s] for s in ids]),**{g:np.stack([f[s][g] for s in ids]) for g in f[ids[0]]})
            seal(root/'branches'/(ds+'__'+arm+'.json'),dict(branches=b,tie_keys=keys,observations=obs))
            print(ds,arm,'head frozen',winner['groups'],winner['recipe'],'selection F1',winner['selection']['f1'],flush=True)
    seal(root/'trials.json',trials);seal(root/'frozen.json',dict(choices=frozen,test_used_for_selection=False,no_refit=True,correction_frozen_sha256=sha(root/'correction_frozen.json')))


def evaluate(c):
    from recognition.budget_target_oracle import analyze_budget
    root=Path(c['study_root']);plan=read(Path(c['original_router_root'])/'resolved_configuration.json');frozen=read(root/'frozen.json');out={}
    assert not frozen['test_used_for_selection'] and frozen['no_refit']
    for ds,t in c['targets'].items():
        out[ds]={};ids=plan['partitions'][ds]['test']
        for arm in c.get('experts',('unet','yolo')):
            choice=frozen['choices'][ds][arm];assert sha(choice['winner']['path'])==choice['winner']['sha256'];bundle=joblib.load(choice['winner']['path']);data=read(root/'branches'/(ds+'__'+arm+'.json'));b={s:data['branches'][s] for s in ids};keys=data['tie_keys']
            with np.load(root/'models'/(ds+'__'+arm+'__features.npz'),allow_pickle=False) as z:
                positions={s:i for i,s in enumerate(z['ids'].tolist())};ix=[positions[s] for s in ids];x=np.concatenate([z[g][ix] for g in bundle['groups']],axis=1);scale=z['scale'][ix]
            scores=dict(zip(ids,map(float,predict(bundle,x,scale))));accepted=order(scores,keys)[:len(ids)-t['fallback_images']]
            primary=replay(b,accepted);scalar=replay(b,accept(scores,choice['calibration_cut'],keys));oracle=analyze_budget(b,list(ids),t['fallback_images'],str(t['f1']))
            out[ds][arm]=dict(all_light=replay(b,ids),all_deepsolo=replay(b,[]),primary=primary,validation_cut=scalar,oracle=oracle,target_met=primary['f1']>=t['f1'])
            seal(root/'test_scores'/(ds+'__'+arm+'.json'),dict(scores=scores,branches=b,tie_keys={s:keys[s] for s in ids}))
            print(ds,arm,'TEST F1',100*primary['f1'],'DeepSolo',primary['deepsolo_images'],'/',primary['total_images'],'oracle',100*oracle['exact_budget']['f1'],flush=True)
    seal(root/'results.json',dict(status='PASS',mode='approximate',claim_label='approximate reproduction',datasets=out,frozen_sha256=sha(root/'frozen.json'),actual_new_neural_inferences=0,scope='User budgets exact61/233 and228/500, logical cached routing; lexicon also applied to light outputs, an explicit extension.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--stage',choices=['select','train','evaluate'],required=True);a=p.parse_args();c=read(a.config)
    assert c['mode']=='approximate' and c['claim_label']=='approximate reproduction';c['_source_config_sha256']=sha(a.config)
    for path,h in c['pinned_inputs'].items():assert sha(path)==h,path
    {'select':select,'train':train,'evaluate':evaluate}[a.stage](c)
