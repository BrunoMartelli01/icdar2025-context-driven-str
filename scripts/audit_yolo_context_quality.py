"""Independent arithmetic/source audit of frozen contextual routing artifacts."""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def digest(v):
    return hashlib.sha256(json.dumps({k:x for k,x in v.items() if k!='payload_sha256'},
        sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def sealed(path,ph):
    v=read(path)
    assert v['plan_sha256']==ph and v['payload_sha256']==digest(v),str(path)
    return v


def good(counts):
    tp,fp,fn=(counts[k] for k in ('tp','fp','fn'))
    return 2*tp+fp+fn>0 and 10*tp>=4*(2*tp+fp+fn)


def counts_for(branches,ids,accepted,word_source=None):
    rows=[branches[s][word_source or ('light' if s in accepted else 'deepsolo')] for s in ids]
    counts={k:sum(r[k] for r in rows) for k in
            ('tp','fp','fn','detected_words','gt_words','all_outputs','ignored_outputs')}
    tp,fp,fn=(counts[k] for k in ('tp','fp','fn'))
    assert counts['detected_words']==tp+fp and counts['gt_words']==tp+fn
    return {**counts,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        'accuracy_tn0':tp/(tp+fp+fn) if tp+fp+fn else None}


def empirical_cut(scores,branches,ids,target,minimum):
    choices=[]
    for cut in sorted({scores[s] for s in ids if scores[s] is not None}):
        accepted=[s for s in ids if scores[s] is not None and scores[s]>=cut]
        g=sum(good(branches[s]['light']) for s in accepted)
        if len(accepted)>=minimum and Fraction(g,len(accepted))>=Fraction(str(target)):
            choices.append((len(accepted),g/len(accepted),cut))
    return max(choices) if choices else (0,0,None)


def rank_accept(scores,ids,n):
    return set(sorted((s for s in ids if scores[s] is not None),key=lambda s:(-scores[s],s))[:n])


def cdf(values,x):
    if x is None:return None
    return (sum(v<x for v in values)+sum(v<=x for v in values))/(2*len(values))


def expected_scores(feature,reference):
    names=feature['scores']
    if not feature['eligible']:return dict.fromkeys(names)
    s=feature['signals'];base=feature['source_scores']['all_checks_softcrop']
    scaled={k:cdf(v,s[k]) for k,v in reference['fields'].items()}
    mix=lambda x:(base*x)**.5 if x is not None else None
    return {'coverage_baseline':feature['source_scores']['coverage_baseline'],
        'all_checks_softcrop':base,'paper_caption_control':s['paper_caption_control'],
        'literal_only':s['literal_recall'],'base_literal':base*(s['literal_recall'] if s['literal_recall'] is not None else 1.),
        **scaled,'base_mpnet':mix(scaled['mpnet_contrast']),
        'base_clip_plain':mix(scaled['clip_plain_margin']),
        'base_clip_context':mix(scaled['clip_context_margin']),
        'base_clip_gain':mix(scaled['clip_context_gain']),
        'base_lm_literal':mix(scaled['lm_literal']),
        'base_lm_coherence':mix(scaled['lm_coherence'])}


def verify_point(point,branches,ids,accepted):
    assert point['accepted_ids']==sorted(accepted)
    assert point['accepted_images']==len(accepted)
    assert point['fallback_images']==len(ids)-len(accepted)
    g=sum(good(branches[s]['light']) for s in accepted)
    assert point['accepted_good_images']==g
    assert point['accepted_bad_images']==len(accepted)-g
    assert point['available_good_images']==sum(good(branches[s]['light']) for s in ids)
    assert point['good_fraction_among_accepted']==(g/len(accepted) if accepted else None)
    for k,value in counts_for(branches,ids,accepted).items():
        assert point['combined'][k]==value,(k,point['combined'][k],value)
    for k,value in counts_for(branches,sorted(accepted),accepted,'light').items():
        assert point['accepted_words'][k]==value


def audit(root):
    p=read(root/'resolved_configuration.json');ph=sha(root/'resolved_configuration.json');cfg=p['config']
    r=read(root/'results.json');frozen=sealed(root/'frozen_thresholds.json',ph)
    reference=sealed(root/'frozen_references.json',ph)
    assert r['status']=='SUCCESS' and r['plan_sha256']==ph and r['optimizer_steps']==0
    assert r['new_network_training'] is False and r['formal_risk_guarantee'] is False
    assert frozen['test_features_present_at_freeze']==0 and frozen['test_labels_used'] is False
    assert r['thresholds_sha256']==sha(root/'frozen_thresholds.json')
    assert r['references_sha256']==sha(root/'frozen_references.json')
    assert reference['reference_ids']==p['selection_ids']
    selection=p['selection_ids'];calibration=p['threshold_calibration_ids']
    assert not set(selection)&set(calibration)
    for name,d in p['code_sha256'].items():assert sha(root/'code_snapshot'/name)==d
    for name,d in p['input_sha256'].items():assert sha(name)==d
    coverage=Path(p['paths']['coverage_run']);source=read(coverage/'results.json')
    assert sha(coverage/'resolved_configuration.json')==cfg['coverage_plan_sha256']
    assert sha(coverage/'results.json')==cfg['coverage_results_sha256']
    branches=source['resolutions']['640']['branch_metrics']
    paired=Path(p['paths']['paired_run']);paired_plan=read(paired/'resolved_configuration.json')
    assert sha(paired/'resolved_configuration.json')==cfg['paired_plan_sha256']
    caps={s['sample_id']:s['caption']['text'] for rows in paired_plan['samples'].values() for s in rows}
    assert p['negative_caption_reference_pool']=={s:caps[s] for s in p['negative_caption_reference_ids']}
    features={};crop_count=0;states=0;point_count=0;diagnostics={}
    for split,samples in p['samples'].items():
        features[split]={};mp=sealed(root/split/'mpnet.json',ph);cl=sealed(root/split/'clip.json',ph)
        lm=sealed(root/split/'lm.json',ph);li={s:i for i,s in enumerate(lm['sample_ids'])}
        assert set(li)=={s['sample_id'] for s in samples}
        for receipt in (mp,cl):
            assert receipt['model_state_sha256_before']==receipt['model_state_sha256_after']
            assert receipt['optimizer_steps']==0 and receipt['ground_truth_used'] is False
            assert receipt['cuda_peak']['peak_allocated_bytes']>0 and receipt['cuda_peak']['peak_reserved_bytes']>0
            states+=1
        for key in ('exact_support','scene_coherence','wrong_caption_coherence'):
            q=lm[key];assert q['provenance']['model_state_sha256_before']==q['provenance']['model_state_sha256_after']
            assert q['provenance']['optimizer_steps']==0 and q['provenance']['ground_truth_in_scoring'] is False
            assert q['provenance']['truncation'] is False
            for row in q['rows']:
                if row['status']=='ok':
                    lp=np.array([row['yes_logprob'],row['no_logprob']]);prob=np.exp(lp-lp.max());prob/=prob.sum()
                    assert abs(row['score']-prob[0])<2e-7
                else:assert row['score'] is None and row['error']
            states+=1
        anchor_count=0;overflow_count=0
        for sample in samples:
            sid=sample['sample_id'];key=sid.replace(':','__')+'.json'
            assert sha(sample['image_path'])==sample['image_sha256']
            f=sealed(root/split/'features'/key,ph);features[split][sid]=f
            assert f['sample_id']==sid and f['image_sha256']==sample['image_sha256']
            assert f['ground_truth_used'] is False and f['models_trained'] is False and f['original_word_outputs_changed'] is False
            assert f['caption']['text']==caps[sid]
            assert len(f['controls'])==cfg['caption_negative_controls']
            for control in f['controls']:
                assert control['sample_id'] in p['negative_caption_reference_ids']
                assert control['sample_id']!=sid and control['text']==caps[control['sample_id']]
                assert control['text'].casefold()!=f['caption']['text'].casefold()
            cf=read(coverage/'640'/split/'features'/key)
            ordered=sorted(cf['rows'],key=lambda x:(x['bbox'][1],x['bbox'][0],x['instance_id']))
            assert f['transcript']==' '.join(x['raw_text'] for x in ordered)
            assert [w['candidate_id'] for w in cl['rows'][sid]['words']]==[w['candidate_id'] for w in cf['rows']]
            with Image.open(sample['image_path']) as im:rgb=np.array(im.convert('RGB'))
            for w,probe in zip(cf['rows'],cl['rows'][sid]['words']):
                x1,y1,x2,y2=w['bbox'];crop=np.ascontiguousarray(rgb[y1:y2,x1:x2])
                assert list(crop.shape)==w['crop_shape']
                assert hashlib.sha256(crop.tobytes()).hexdigest()==w['crop_rgb_sha256']
                assert probe['variants'][0]==w['raw_text']
                for condition,values in probe['cosines'].items():
                    if None not in values:
                        assert abs(probe['margins'][condition]-(values[0]-max(values[1:])))<1e-12
                    else:assert probe['margins'][condition] is None
                crop_count+=1
            words=cl['rows'][sid]['words'];q=cfg['word_margin_quantile']
            if cl['rows'][sid]['status']=='SCORED' and words:
                assert f['signals']['clip_plain_margin']==float(np.quantile([w['margins']['plain'] for w in words],q))
                assert f['signals']['clip_context_margin']==float(np.quantile([w['margins']['own'] for w in words],q))
                assert f['signals']['clip_context_gain']==float(np.quantile([w['margins']['own']-w['margins']['wrong'] for w in words],q))
            else:overflow_count+=cl['rows'][sid]['status']!='SCORED'
            assert f['signals']['mpnet_contrast']==mp['rows'][sid]['contrast']
            if mp['rows'][sid]['contrast'] is not None:
                own=mp['rows'][sid]['own_caption_cosine'];other=mp['rows'][sid]['control_cosines']
                assert abs(f['signals']['mpnet_contrast']-(own-sum(other)/len(other)))<1e-12
            assert f['signals']['lm_literal']==lm['exact_support']['scores'][li[sid]]
            assert f['signals']['lm_coherence']==lm['scene_coherence']['scores'][li[sid]]
            e=f['literal_evidence']['evidence'];tokens=[x['casefold'] for x in e['tokens']]
            matched=sum((Counter(tokens)&Counter(w['raw_text'].casefold() for w in cf['rows'])).values())
            assert f['signals']['literal_recall']==(matched/len(tokens) if tokens else None)
            for span in e['spans']:assert f['caption']['text'][span['start']:span['end']]==span['text']
            anchor_count+=bool(tokens)
            expected=expected_scores(f,reference)
            assert set(expected)==set(f['scores'])
            for name,val in expected.items():
                if val is None:assert f['scores'][name] is None
                else:assert abs(val-f['scores'][name])<1e-12
        diagnostics[split]={'images':len(samples),'literal_anchor_images':anchor_count,'clip_overflow_images':overflow_count,
            'good_light_images':sum(good(branches[split][s['sample_id']]['light']) for s in samples)}
    assert set(selection+calibration)==set(features['train'])
    assert not set(features['test'])&set(features['train'])
    for field,values in reference['fields'].items():
        assert values==sorted(features['train'][s]['signals'][field] for s in selection
                              if features['train'][s]['signals'][field] is not None)
    for family in cfg['families']:
        scores={s:f['scores'][family] for s,f in features['train'].items()}
        for target in cfg['target_good_image_fractions']:
            expected=empirical_cut(scores,branches['train'],calibration,target,cfg['minimum_calibration_accepted'])
            assert frozen['families'][family]['targets'][str(target)]['threshold']==expected[2]
    for target,selection_trials in frozen['selection_trials'].items():
        verified=[]
        for family in cfg['families']:
            scores={s:f['scores'][family] for s,f in features['train'].items()}
            count,frac,cut=empirical_cut(scores,branches['train'],selection,float(target),cfg['minimum_calibration_accepted'])
            verified.append((-count,-frac,cfg['families'].index(family),family))
        assert frozen['selected_families'][target]==min(verified)[3]
    for name,ids,split in [('train',list(features['train']),'train'),('selection',selection,'train'),
                           ('calibration',calibration,'train'),('test',list(features['test']),'test')]:
        for family,points in r['splits'][name]['families'].items():
            scores={s:features[split][s]['scores'][family] for s in ids}
            for point_name,point in points.items():
                if point_name.startswith('budget_'):
                    n=round(len(ids)*float(point_name.split('_')[1]));accepted=rank_accept(scores,ids,n)
                    assert point['unfilled_budget']==n-len(accepted)
                else:
                    cut=point['threshold'];accepted={s for s in ids if cut is not None and scores[s] is not None and scores[s]>=cut}
                verify_point(point,branches[split],ids,accepted);point_count+=1
        for source_name,key in [('light','light_only'),('deepsolo','deepsolo_only')]:
            for k,value in counts_for(branches[split],ids,set(),source_name).items():assert r['splits'][name][key][k]==value
    if p['result_class']!='SMOKE_TEST':
        assert diagnostics['train']['images']==125 and diagnostics['test']['images']==300
        assert r['splits']['train']['light_only']['gt_words']==983
        assert r['splits']['test']['light_only']['gt_words']==2204
    first_test=min((root/'test'/n).stat().st_mtime_ns for n in ('mpnet.json','clip.json','lm.json'))
    assert (root/'frozen_thresholds.json').stat().st_mtime_ns<first_test
    result={'status':'PASS','reproduction_mode':'approximate','claim_label':'approximate reproduction',
        'plan_sha256':ph,'results_sha256':sha(root/'results.json'),'audit_code_sha256':sha(__file__),
        'independent_count_arithmetic':True,'source_rgb_crops_verified':crop_count,'routing_points_verified':point_count,
        'features_verified':sum(len(v) for v in features.values()),'frozen_state_receipts_verified':states,
        'optimizer_steps':0,'train_only_references_thresholds':True,'test_labels_used_for_selection':False,
        'diagnostics':diagnostics}
    (root/'independent_audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run)
