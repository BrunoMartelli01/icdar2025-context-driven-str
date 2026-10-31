"""Independent lexical, copying, candidate, fold and calibration audit."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import pickle

import numpy as np
from PIL import Image


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(2**20),b''):h.update(part)
    return h.hexdigest()


def measure(texts,gt):
    values=Counter(v.casefold() for s in texts for v in (s.split() or ['']))
    matches=values & gt;t=sum(matches.values());n=sum(values.values());g=sum(gt.values())
    return dict(correct=t,predictions=n,gt_words=g,matched=matches)


def manual_selection(sample,values,threshold):
    chosen=[]
    for b,opts,vs in zip(sample['blocks'],sample['candidates'],values):
        ranked=sorted(enumerate(vs,1),key=lambda v:(-v[1],v[0]))
        j=ranked[0][0] if ranked and ranked[0][1]>=threshold else 0
        chosen.append(opts[j]['text'])
    return chosen


def audit(root):
    from scripts.evaluate_totaltext_v1_findcontours_parseq import _gt
    root=Path(root);p=read(root/'protocol.json');c=p['resolved_config'];data=read(root/'candidates.json')
    results=read(root/'results.json');predictions=read(root/'predictions.json');scores=read(root/'scores.json');selection=read(root/'selection_frozen.json')
    if sha(root/'protocol.json')!=results['protocol_sha256'] or sha(root/'selection_frozen.json')!=results['selection_sha256']:
        raise AssertionError('Resolved protocol or selection changed')
    counts=Counter();word_cache={};base_cache={};sample_index={s['sample_id']:s for ss in data.values() for s in ss}
    def expected(s):
        sid=s['sample_id']
        if sid not in word_cache:
            if sha(s['annotation_path'])!=s['annotation_sha256']:raise AssertionError('Changed annotation')
            words=_gt(Path(s['annotation_path']),tuple(reversed(s['image_hw'])))
            word_cache[sid]=Counter(w['text'].casefold() for w in words)
        return word_cache[sid]
    def metrics(samples,pred):
        total=Counter()
        for s in samples:
            sid=s['sample_id'];gt=expected(s);r=measure(pred[sid],gt)
            before=measure([b['text'] for b in s['blocks']],gt)
            r.update(recovered=sum((r['matched']-before['matched']).values()),damaged=sum((before['matched']-r['matched']).values()))
            for k in ('correct','predictions','gt_words','recovered','damaged'):total[k]+=r[k]
        total['f1']=2*total['correct']/(total['predictions']+total['gt_words'])
        return total
    # Independent source exclusion: exactly one held-out fold, no shared RGB.
    fit_by={s['sample_id']:s for s in data['fit']}
    for fold in range(c['oof']['folds']):
        pairs=read(root/f'oof{fold}_pairs.json');train=set(pairs['training_ids']);held=set(pairs['prediction_ids'])
        assert not train & held and train | held == set(fit_by)
        assert not {fit_by[s]['image_sha256'] for s in train} & {fit_by[s]['image_sha256'] for s in held}
        assert held=={sid for sid,value in p['folds'].items() if value==fold}
        assert sha(root/f'oof{fold}.pt')==read(root/'generator_memory.json')[fold]['checkpoint_sha256']
        counts['oof_folds']+=1
    with (root/'priors.pkl').open('rb') as stream:priors=pickle.load(stream)
    for fold in range(c['oof']['folds']):
        expected_counts=Counter()
        for sid,s in fit_by.items():
            if p['folds'][sid]!=fold:expected_counts.update(expected(s))
        assert priors[str(fold)].counts==expected_counts
        counts['fold_prior_checks']+=1
    final_counts=Counter()
    for s in fit_by.values():final_counts.update(expected(s))
    assert priors['final'].counts==final_counts
    for samples in data.values():
        for s in samples:
            assert len(s['blocks'])==len(s['candidates'])==len(s['contexts'])
            for b,opts,context in zip(s['blocks'],s['candidates'],s['contexts']):
                assert opts[0]['text']==b['text'] and len({v['text'] for v in opts})==len(opts)
                for v in opts[1:]:
                    assert ''.join(v['text'].split())==b['text']
                    assert 2<=len(v['text'].split())<=c['candidates']['maximum_words']
                    counts['candidate_copy_checks']+=1
                for side in context:
                    assert len(side)<=c['context']['neighbours_per_side']
                    assert all(s['blocks'][i]['id']!=b['id'] for i in side)
    for phase,arms in predictions.items():
        samples=data[phase]
        for name,by in arms.items():
            actual=metrics(samples,by);saved=results['results'][phase][name]['aggregate']
            for k,value in actual.items():assert abs(saved[k]-value)<1e-12,(phase,name,k)
            per={r['sample_id']:r for r in results['results'][phase][name]['per_image']}
            for s in samples:
                sid=s['sample_id'];gt=expected(s);r=measure(by[sid],gt)
                assert len(by[sid])==len(s['blocks'])
                for k in ('correct','predictions','gt_words'):assert per[sid][k]==r[k]
                for b,text in zip(s['blocks'],by[sid]):
                    assert ''.join(text.split())==''.join(b['text'].split())
                    counts['output_copy_checks']+=1
                counts['image_arm_checks']+=1
    # Independently reconstruct threshold policies and the selected calibration row.
    for name,by_phase in scores.items():
        for phase,samples in data.items():
            if phase=='fit':continue
            for cap in c['selector']['damage_limits']:
                threshold=selection['calibration'][name][str(cap)]['choice']['threshold']
                for s in samples:
                    assert manual_selection(s,by_phase[phase][s['sample_id']],threshold)==predictions[phase][name+'_cap'+str(cap)][s['sample_id']]
                    counts['frozen_policy_replay_checks']+=1
            if name.endswith(str(c['selector']['seeds'][0])):
                for s in samples:
                    assert manual_selection(s,by_phase[phase][s['sample_id']],0.)==predictions[phase][name+'_raw'][s['sample_id']]
        for cap,value in selection['calibration'][name].items():
            verified=[]
            for curve in value['curve']:
                threshold=curve['threshold'];pred={}
                for s in data['calibration']:
                    pred[s['sample_id']]=manual_selection(s,by_phase['calibration'][s['sample_id']],threshold)
                r=metrics(data['calibration'],pred)
                for k,v in r.items():assert abs(v-curve[k])<1e-12
                verified.append(curve);counts['calibration_curve_checks']+=1
            valid=[v for v in verified if v['damaged']<=int(cap)]
            choice=max(valid,key=lambda v:(v['correct'],v['f1'],v['threshold']))
            assert choice==value['choice']
    # Check the exact same RGB pixels were used by the weak visual ablation.
    restorations={r['sample_id']:r for r in read(c['paths']['restoration_manifest'])['restored']}
    for phase,samples in data.items():
        profiles=read(root/f'visual_{phase}.json')
        for s in samples:
            active=[i for i,v in enumerate(profiles[s['sample_id']]) if v is not None]
            if not active:continue
            source=read(Path(c['paths']['source_run'])/'exports'/phase/(s['sample_id'].replace(':','__')+'.json'))
            path=Path(source['image_path'])
            if sha(path)!=s['image_sha256']:path=Path(restorations[s['sample_id']]['isolated_copy'])
            assert sha(path)==s['image_sha256']
            with Image.open(path) as im:rgb=np.asarray(im.convert('RGB'))
            for i in active:
                b=s['blocks'][i];x1,y1,x2,y2=b['bbox'];crop=rgb[y1:y2,x1:x2]
                h=hashlib.sha256(np.asarray(crop.shape,dtype='<i8').tobytes());h.update(np.ascontiguousarray(crop).tobytes())
                assert h.hexdigest()==b['crop_sha256'],(s['sample_id'],b['id'])
                counts['original_crop_hashes']+=1
    # Independent exhaustive oracle checks for tractable per-image alternatives.
    oracle=read(root/'oracle.json')
    for phase in ('calibration','evaluation','test'):
        verified=0
        by={v['sample_id']:v for v in oracle[phase]}
        for s in data[phase]:
            cardinality=1
            for opts in s['candidates']:cardinality*=len(opts)
            if cardinality>50000 or cardinality==1:continue
            gt=expected(s)
            best=max(measure([v['text'] for v in choice],gt)['correct'] for choice in itertools.product(*s['candidates']))
            assert best==by[s['sample_id']]['correct']
            counts['exhaustive_oracle_checks']+=1;verified+=1
            if verified==12:break
    return dict(status='PASS',checks=dict(counts),audit_code_sha256=sha(__file__),results_sha256=sha(root/'results.json'),
        scope='Independent counting/copying/fold exclusion/calibration, source crop hashes and exhaustive oracle subset')


def audit_sensitivity(source,root):
    from scripts.evaluate_totaltext_v1_findcontours_parseq import _gt
    source=Path(source);root=Path(root);data=read(source/'candidates.json');scores=read(source/'scores.json')
    protocol=read(root/'protocol.json');result=read(root/'results.json');selection=read(root/'selection_frozen.json');predictions=read(root/'predictions.json')
    assert sha(source/'scores.json')==protocol['source_scores_sha256']
    assert sha(source/'candidates.json')==protocol['source_candidates_sha256']
    assert sha(root/'protocol.json')==result['protocol_sha256'] and sha(root/'selection_frozen.json')==result['selection_sha256']
    declared=read(source.parent/'threshold_sensitivity_predeclaration.json')
    assert not declared['primary_results_exists'] and not declared['primary_selection_exists']
    assert declared['config_sha256']==protocol['config_sha256'] and declared['code_sha256']==protocol['code_sha256']
    gt={};checks=Counter()
    for samples in data.values():
        for s in samples:
            if s['sample_id'] not in gt:
                assert sha(s['annotation_path'])==s['annotation_sha256']
                gt[s['sample_id']]=Counter(w['text'].casefold() for w in _gt(Path(s['annotation_path']),tuple(reversed(s['image_hw']))))
    def metrics(samples,pred):
        total=Counter()
        for s in samples:
            sid=s['sample_id'];g=gt[sid];r=measure(pred[sid],g);b=measure([v['text'] for v in s['blocks']],g)
            r['recovered']=sum((r['matched']-b['matched']).values());r['damaged']=sum((b['matched']-r['matched']).values())
            for k in ('correct','predictions','gt_words','recovered','damaged'):total[k]+=r[k]
        total['f1']=2*total['correct']/(total['predictions']+total['gt_words'])
        return total
    for arm,curve in selection['curves'].items():
        thresholds={1.01}
        for s in data['calibration']:thresholds.update(max(v) for v in scores[arm]['calibration'][s['sample_id']] if v)
        assert thresholds=={r['threshold'] for r in curve}
        for r in curve:
            p={s['sample_id']:manual_selection(s,scores[arm]['calibration'][s['sample_id']],r['threshold']) for s in data['calibration']}
            actual=metrics(data['calibration'],p)
            for k,v in actual.items():assert abs(r[k]-v)<1e-12
            checks['calibration_breakpoints']+=1
    for name,value in selection['choices'].items():
        eligible=[r for r in selection['curves'][value['source_arm']] if r['damaged']<=value['cap']]
        if value['objective']=='correct_then_f1':choice=max(eligible,key=lambda v:(v['correct'],v['f1'],v['threshold']))
        else:choice=max(eligible,key=lambda v:(v['f1'],v['correct'],v['threshold']))
        assert choice==value['choice']
        for phase in ('calibration','evaluation','test'):
            pred={s['sample_id']:manual_selection(s,scores[value['source_arm']][phase][s['sample_id']],choice['threshold']) for s in data[phase]}
            assert pred==predictions[phase][name]
            actual=metrics(data[phase],pred)
            for k,v in actual.items():assert abs(result['results'][phase][name]['aggregate'][k]-v)<1e-12
            checks['image_policy_checks']+=len(data[phase])
    return dict(status='PASS',checks=dict(checks),results_sha256=sha(root/'results.json'),audit_code_sha256=sha(__file__))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True,type=Path);parser.add_argument('--supplement',type=Path);args=parser.parse_args()
    result=audit(args.run);(args.run/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    if args.supplement:
        result=audit_sensitivity(args.run,args.supplement);(args.supplement/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
