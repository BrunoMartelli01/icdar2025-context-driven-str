"""Independent count/geometry/threshold audit of completed frozen consistency routing."""
from __future__ import annotations

import argparse
from bisect import bisect_left,bisect_right
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def shard(path,plan_sha):
    r=read(path);assert r['plan_sha256']==plan_sha
    payload={k:v for k,v in r.items() if k!='payload_sha256'}
    assert hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==r['payload_sha256']
    return r


def iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return intersection/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)


def match_count(a,b,threshold):
    if not a or not b:return 0
    graph=np.array([[bool(x['raw_text']) and x['raw_text'].casefold()==y['raw_text'].casefold()
        and iou(x['bbox'],y['bbox'])>=threshold for y in b] for x in a],dtype=np.int8)
    return int(np.count_nonzero(maximum_bipartite_matching(csr_matrix(graph),perm_type='column')>=0))


def edit_distance(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):current.append(min(prev[j]+1,current[-1]+1,prev[j-1]+(x!=y)))
        prev=current
    return prev[-1]


def good(m):
    den=2*m['tp']+m['fp']+m['fn']
    return den>0 and 10*m['tp']>=4*den


def calibrate(scores,branches,ids,target,minimum):
    candidates=[]
    for cut in sorted({scores[s] for s in ids if scores[s] is not None}):
        accepted=[s for s in ids if scores[s] is not None and scores[s]>=cut]
        n=len(accepted);g=sum(good(branches[s]['light']) for s in accepted)
        if n>=minimum and g/n>=target:candidates.append((n,g/n,cut,g))
    if not candidates:return None,None
    n,p,cut,g=max(candidates)
    return cut,{'threshold':cut,'accepted_images':n,'good_images':g,'good_fraction':p}


def run(args):
    if not __debug__:raise RuntimeError('Run audit without Python -O')
    root=args.run.resolve();plan=read(root/'resolved_configuration.json');result=read(root/'results.json')
    cfg=plan['config'];ph=sha(root/'resolved_configuration.json');source=Path(plan['paths']['source_run'])
    assert result['status']=='SUCCESS' and result['new_network_training'] is False
    assert sha(source/'results.json')==cfg['source_results_sha256']
    original=read(source/'results.json')['resolutions']['640'];source_sha=cfg['source_plan_sha256']
    for name,digest in plan['input_sha256'].items():assert sha(Path(name))==digest
    for name,digest in plan['code_sha256'].items():assert sha(root/'code_snapshot'/name)==digest
    receipt=read(root/'train/model_receipt.json')
    assert receipt['state_sha256_before']==receipt['state_sha256_after'] and receipt['optimizer_steps']==0
    assert receipt['device']=='cpu' and receipt['precision']=='float32'
    frozen=read(root/'frozen_thresholds.json');assert sha(root/'frozen_thresholds.json')==result['thresholds_sha256']
    assert frozen['test_features_present_at_freeze']==0 and frozen['test_labels_used'] is False
    assert set(plan['selection_ids']).isdisjoint(plan['threshold_calibration_ids'])
    ref=plan['source_selector']['confidence_reference'];winner=plan['source_selector']['winner']
    rank=lambda reference,x:(bisect_left(reference,x)+bisect_right(reference,x))/(2*len(reference))
    scores={};checked_views=0;score_records=0;features={};branches={}
    for split,samples in plan['samples'].items():
        scores[split]={};features[split]={};branches[split]={}
        for sample in samples:
            sid=sample['sample_id'];key=sid.replace(':','__')+'.json'
            a=shard(source/'640'/split/'features'/key,source_sha)
            b=shard(source/'960'/split/'features'/key,source_sha)
            view=shard(root/split/'views'/key,ph);f=shard(root/split/'features'/key,ph)
            assert f['ground_truth_used'] is False and f['models_trained'] is False and f['original_word_outputs_changed'] is False
            assert f['coverage']==a['coverage']
            assert sha(Path(sample['image_path']))==sample['image_sha256']
            with Image.open(sample['image_path']) as im:rgb=np.asarray(im.convert('RGB'))
            assert list(rgb.shape[:2][::-1])==sample['image_wh']
            bykey={(r['original_candidate_id'],r['fraction']):r for r in view['rows']}
            assert len(bykey)==len(view['rows'])==len(a['rows'])*len(cfg['crop_fractions'])
            exact=[];edits=[]
            for row in a['rows']:
                x1,y1,x2,y2=row['bbox'];w=x2-x1;h=y2-y1
                assert hashlib.sha256(np.ascontiguousarray(rgb[y1:y2,x1:x2]).tobytes()).hexdigest()==row['crop_rgb_sha256']
                texts=[]
                for fraction in cfg['crop_fractions']:
                    v=bykey[(row['candidate_id'],fraction)]
                    xx1=max(0,math.floor(x1-w*fraction));yy1=max(0,math.floor(y1-h*fraction))
                    xx2=min(sample['image_wh'][0],math.ceil(x2+w*fraction));yy2=min(sample['image_wh'][1],math.ceil(y2+h*fraction))
                    assert v['crop_bbox']==[xx1,yy1,xx2,yy2] and v['bbox']==row['bbox']
                    crop=np.ascontiguousarray(rgb[yy1:yy2,xx1:xx2])
                    assert v['crop_rgb_sha256']==hashlib.sha256(crop.tobytes()).hexdigest() and v['crop_shape']==list(crop.shape)
                    texts.append(v['raw_text'].casefold());checked_views+=1
                text=row['raw_text'].casefold();exact.append(bool(text) and all(t==text for t in texts))
                edits.append(min(1-edit_distance(text,t)/max(len(text),len(t)) if text and t else 0 for t in texts))
            scale=f['scale_agreement'];matches=match_count(a['rows'],b['rows'],cfg['scale_pair_iou'])
            assert len(scale['pairs'])==matches
            assert len({p['left'] for p in scale['pairs']})==len({p['right'] for p in scale['pairs']})==matches
            aa={r['candidate_id']:r for r in a['rows']};bb={r['candidate_id']:r for r in b['rows']}
            for pair in scale['pairs']:
                x,y=aa[pair['left']],bb[pair['right']]
                assert x['raw_text'].casefold()==y['raw_text'].casefold() and iou(x['bbox'],y['bbox'])>=cfg['scale_pair_iou']
            scale_score=2*matches/(len(a['rows'])+len(b['rows'])) if a['rows'] and b['rows'] else 0.
            crop_score=sum(exact)/len(exact) if exact else 0.;soft=sum(edits)/len(edits) if edits else 0.
            assert math.isclose(f['crop_stability']['exact_fraction'],crop_score,abs_tol=1e-15)
            assert math.isclose(f['crop_stability']['edit_agreement'],soft,abs_tol=1e-15)
            if a['rows']:
                native=min(rank(ref['native'],r['confidence']) for r in a['rows'])
                clip=min(rank(ref['word_advantage'],r['clip_word_advantage']) for r in a['rows'])
                conf=min(math.sqrt(rank(ref['native'],r['confidence'])*rank(ref['word_advantage'],r['clip_word_advantage'])) for r in a['rows'])
                previous=conf if not winner['weight'] else (1-winner['weight'])*conf+winner['weight']*rank(winner['reference'],a['coverage'][winner['coverage_field']])
                cov=min(a['coverage']['text_coverage'],a['coverage']['kernel_coverage'])
                values=[conf,previous,native,clip,scale_score,crop_score,cov,math.sqrt(conf*scale_score),
                    math.sqrt(conf*crop_score),math.sqrt(conf*cov),(conf*scale_score*crop_score*cov)**.25,
                    min(conf,scale_score,crop_score,cov),(conf*scale_score*soft*cov)**.25]
            else:values=[None]*len(cfg['families'])
            assert list(f['scores'].keys())==sorted(cfg['families'])
            for name,value in zip(cfg['families'],values):
                actual=f['scores'][name]
                assert actual is None if value is None else actual is not None and math.isclose(actual,value,rel_tol=0,abs_tol=1e-15)
            scores[split][sid]=dict(zip(cfg['families'],values));features[split][sid]=f
            branches[split][sid]=original['branch_metrics'][split][sid];score_records+=1
        print(f'[AUDIT probes {split}] {len(samples)} image feature/geometry/crop checks PASS',flush=True)
    for family in cfg['families']:
        values={s:features['train'][s]['scores'][family] for s in features['train']}
        for target in cfg['target_good_image_fractions']:
            for ids,stored in [(plan['selection_ids'],next(r for r in frozen['selection_trials'][str(target)] if r['family']==family)),
                    (plan['threshold_calibration_ids'],frozen['families'][family]['targets'][str(target)])]:
                cut,point=calibrate(values,branches['train'],ids,target,cfg['minimum_calibration_accepted'])
                assert cut==stored['threshold'] and point==stored['calibration']
        for fraction in cfg['bypass_fractions']:
            ids=plan['threshold_calibration_ids'];valid=[values[s] for s in ids if values[s] is not None]
            cut=min(set(valid),key=lambda t:(abs(sum(v>=t for v in valid)-len(ids)*fraction),-t)) if valid else None
            assert cut==frozen['families'][family]['coverage_cuts'][str(fraction)]
    for target,trials in frozen['selection_trials'].items():
        expected=min(trials,key=lambda t:(-t['calibration']['accepted_images'] if t['calibration'] else 0,
            -t['calibration']['good_fraction'] if t['calibration'] else 0,cfg['families'].index(t['family'])))['family']
        assert expected==frozen['selected_families'][target]==result['selected_families'][target]
    checked_points=0
    for split,ids,origin in [('train',list(features['train']),'train'),('selection',plan['selection_ids'],'train'),
            ('calibration',plan['threshold_calibration_ids'],'train'),('test',list(features['test']),'test')]:
        for family,points in result['splits'][split]['families'].items():
            vv={s:scores[origin][s][family] for s in ids}
            available={s for s in ids if good(branches[origin][s]['light'])}
            for name,point in points.items():
                if name.startswith('budget_'):
                    fraction=float(name.split('_')[1]);valid=sorted((s for s in ids if vv[s] is not None),key=lambda s:(-vv[s],s))
                    accepted=set(valid[:round(len(ids)*fraction)])
                else:
                    cut=point['threshold'];accepted={s for s in ids if vv[s] is not None and cut is not None and vv[s]>=cut}
                assert accepted==set(point['accepted_ids']) and point['accepted_good_images']==len(accepted & available)
                assert point['accepted_images']==len(accepted) and point['fallback_images']==len(ids)-len(accepted)
                assert point['accepted_bad_images']==len(accepted-available) and point['available_good_images']==len(available)
                c={k:sum(branches[origin][s]['light' if s in accepted else 'deepsolo'][k] for s in ids)
                    for k in ['tp','fp','fn','detected_words','gt_words','all_outputs','ignored_outputs']}
                assert all(point['combined'][k]==v for k,v in c.items())
                assert point['combined']['f1']==2*c['tp']/(2*c['tp']+c['fp']+c['fn'])
                assert point['combined']['accuracy_tn0']==c['tp']/(c['tp']+c['fp']+c['fn'])
                checked_points+=1
    for fraction,trials in frozen['budget_selection_trials'].items():
        ids=plan['selection_ids'];expected=[]
        for family in cfg['families']:
            vv={s:scores['train'][s][family] for s in ids};valid=sorted((s for s in ids if vv[s] is not None),key=lambda s:(-vv[s],s))
            acc=valid[:round(len(ids)*float(fraction))]
            expected.append({'family':family,'accepted_good_images':sum(good(branches['train'][s]['light']) for s in acc)})
        assert expected==trials
        family=min(expected,key=lambda t:(-t['accepted_good_images'],cfg['families'].index(t['family'])))['family']
        assert family==frozen['selected_budget_families'][fraction]==result['selected_budget_families'][fraction]
    verification={'status':'PASS','reproduction_mode':'approximate','claim_label':'approximate reproduction',
        'plan_sha256':ph,'results_sha256':sha(root/'results.json'),'audit_code_sha256':sha(Path(__file__)),
        'source_results_sha256':cfg['source_results_sha256'],'image_features_verified':score_records,
        'original_rgb_crop_views_verified':checked_views,'metric_routing_points_verified':checked_points,
        'family_threshold_selection_verified':True,'frozen_model_state_unchanged':True,'optimizer_steps':0,
        'test_labels_used_for_selection':False,'formal_risk_guarantee':False,
        'matching_audit':'independent SciPy sparse cardinality matching; original pixel and crop-boundary checks',
        'new_ocr_reads':sum(len(shard(p,ph)['rows']) for p in (root/'train/views').glob('*.json'))}
    path=root/'independent_audit.json'
    if path.exists():assert read(path)==verification
    else:path.write_text(json.dumps(verification,indent=2,sort_keys=True)+'\n')
    print(json.dumps(verification,sort_keys=True),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    run(parser.parse_args())
