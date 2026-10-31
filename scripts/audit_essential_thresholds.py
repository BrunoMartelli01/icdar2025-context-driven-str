"""Independent image matching, gate replay and rational threshold optimality audit."""
import argparse
from collections import Counter
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
FIELDS=('tp','fp','fn','detected_words','gt_words','all_outputs','ignored_outputs')
norm=lambda s:''.join(c for c in s.casefold() if c.isalnum())

def native_binary(image):
    width,height=image.size;side=max(width,height)
    canvas=Image.new('L',(side,side),0);canvas.paste(image.convert('L'),((side-width)//2,(side-height)//2))
    return np.asarray(canvas.resize((256,256),Image.Resampling.NEAREST))>0

def matched(rows,gt,flags):
    predictions=[r.get('raw_text',r.get('text')) for r,ignored in zip(rows,flags) if not ignored]
    valid=[not bool(r.get('invalid_envelope')) for r,ignored in zip(rows,flags) if not ignored]
    edges=np.asarray([[bool(norm(p)) and norm(p)==norm(g['text']) and good for g in gt] for p,good in zip(predictions,valid)],bool).reshape(len(predictions),len(gt))
    tp=int(np.count_nonzero(maximum_bipartite_matching(csr_matrix(edges),perm_type='column')>=0)) if len(predictions) and len(gt) else 0
    return dict(tp=tp,fp=len(predictions)-tp,fn=len(gt)-tp,detected_words=len(predictions),gt_words=len(gt),all_outputs=len(rows),ignored_outputs=sum(flags))

def independent_point(branches,scores,threshold):
    keep=set() if threshold is None else {s for s,v in scores.items() if v is not None and v>=threshold}
    c=Counter()
    for s,b in branches.items():c.update(b['light' if s in keep else 'deepsolo'])
    n=len(branches);return dict(c,f1=2*c['tp']/(c['detected_words']+c['gt_words']),
        accepted_images=len(keep),deepsolo_images=n-len(keep),total_images=n,deepsolo_percent=100*(n-len(keep))/n)

def compare(a,b):
    for k in FIELDS+('f1','accepted_images','deepsolo_images','total_images','deepsolo_percent'):
        if a[k]!=b[k]:raise ValueError('Independent route mismatch: '+k)

def rate(p):return Fraction(2*p['tp'],p['detected_words']+p['gt_words'])

def independent_bootstrap(curve,b,scores,draws,seed):
    """Use the exact F1-gain identity instead of subtracting two F1 estimates."""
    ids=sorted(b);index={s:i for i,s in enumerate(ids)};eligible=sorted((s for s in ids if scores[s] is not None),key=lambda s:(-scores[s],s))
    source=np.array([[b[s]['deepsolo'][k] for k in ['tp','detected_words','gt_words']] for s in ids],np.int64)
    delta=np.array([[b[s]['light'][k]-b[s]['deepsolo'][k] for k in ['tp','detected_words']] for s in eligible],np.int64).reshape(-1,2)
    ends=np.array([p['accepted_images'] for p in curve],int);positions=[index[s] for s in eligible];rng=np.random.default_rng(seed);maximum=[]
    estimate=np.array([p['f1']-curve[0]['f1'] for p in curve]);all_draws=[]
    for start in range(0,draws,100):
        weights=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=min(100,draws-start));base=weights@source
        increments=weights[:,positions,None]*delta[None];prefix=np.concatenate([np.zeros((len(weights),1,2),np.int64),increments.cumsum(axis=1)],axis=1)[:,ends]
        denominator=base[:,1]+base[:,2];dt=prefix[:,:,0];dn=prefix[:,:,1]
        difference=(2*dt*denominator[:,None]-2*base[:,None,0]*dn)/(denominator[:,None]*(denominator[:,None]+dn))
        maximum.extend((difference-estimate[None]).max(axis=1));all_draws.append(difference)
    return float(np.quantile(maximum,.95)),np.quantile(np.concatenate(all_draws),[.025,.975],axis=0).T

def verify_icdar_images(root,checks):
    plan=read(root/'resolved_configuration.json');records=read(root/'evaluation_records.json');saved=read(root/'branches.json');scores=read(root/'test_scores.json')
    deployment=read(plan['config']['paths']['router_deployment'])
    from recognition.lightweight_unet_router import ocr_summary,mask_summary
    for ds,samples in plan['population']['test'].items():
        for s in samples:
            sid=s['sample_id'];record=records[ds][sid]
            if sha(s['image_path'])!=s['image_sha256'] or sha(s['annotation_path'])!=s['annotation_sha256']:raise ValueError('Changed dataset image/GT')
            lines=list(csv.reader(Path(s['annotation_path']).read_text(encoding='utf-8-sig').splitlines(),skipinitialspace=True))
            gt=[]
            for fields in lines:
                if not fields:continue
                if ds=='icdar13':text=fields[4]
                else:
                    text=','.join(fields[8:]).strip()
                    if text=='###':continue
                gt.append(dict(text=text))
            if [g['text'] for g in gt]!=[g['text'] for g in record['ground_truth']]:raise ValueError('Independent GT transcription replay failed')
            if ds=='icdar13':valid=np.ones((256,256),bool)
            else:
                vp=Path(s['image_path']).parent.parent/'validity'/(s['image_path'].split('/')[-1].split('.')[0]+'.png')
                with Image.open(vp) as im:valid=native_binary(im)
            raw=root/'fallback'/(sid.replace(':','__')+'.json');value=read(raw)
            payload={k:v for k,v in value.items() if k!='payload_sha256'}
            if hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()!=value['payload_sha256']:raise ValueError('Changed native fallback payload')
            heavy=record['deepsolo'];flags=[]
            for p in heavy:
                if p['bbox'] is None:flag=False
                else:
                    support=np.zeros((s['image_wh'][1],s['image_wh'][0]),np.uint8);vertices=np.rint(np.clip(np.asarray(p['boundary_original']),-2**29,2**29)).astype(np.int32)
                    cv2.fillPoly(support,[vertices],255);mask=native_binary(Image.fromarray(support))
                    flag=bool(mask.any() and not np.any(mask&valid))
                if flag!=p['ignored']:raise ValueError('Independent polygon-ignore replay failed')
                flags.append(flag)
            hc=matched(heavy,gt,flags)
            u=read(root/'unet'/(sid.replace(':','__')+'.json'));mp=root/u['maps_path']
            if sha(mp)!=u['maps_sha256']:raise ValueError('Changed predicted mask')
            with np.load(mp,allow_pickle=False) as z:prob=z['probabilities']
            rgb=np.asarray(Image.open(s['image_path']).convert('RGB'))
            for expert,light in record['light'].items():
                rows=light['rows'];flags=[]
                for p in rows:
                    x,y,xx,yy=p['bbox'];image=Image.new('L',tuple(s['image_wh']),0);ImageDraw.Draw(image).rectangle((x,y,xx-1,yy-1),fill=255)
                    mask=native_binary(image);flags.append(bool(mask.any() and not np.any(mask&valid)))
                    box=p.get('crop_bbox',p['bbox']);a,b,c,d=box;crop=np.ascontiguousarray(rgb[b:d,a:c])
                    if expert=='unet':
                        h=hashlib.sha256(np.asarray(crop.shape,dtype='<i8').tobytes());h.update(crop.tobytes());digest=h.hexdigest()
                    else:digest=hashlib.sha256(crop.tobytes()).hexdigest()
                    if digest!=p['crop_rgb_sha256']:raise ValueError('Independent selected RGB crop replay failed')
                    checks['selected_crop_pixels']+=1
                actual_light=matched(rows,gt,flags)
                if flags!=light['ignored'] or actual_light!=saved[ds][expert][sid]['light'] or hc!=saved[ds][expert][sid]['deepsolo']:
                    raise ValueError('Independent lexical/ignore counts mismatch: '+json.dumps(dict(sample_id=sid,expert=expert,
                        flags_match=flags==light['ignored'],actual_light=actual_light,saved_light=saved[ds][expert][sid]['light'],actual_heavy=hc,saved_heavy=saved[ds][expert][sid]['deepsolo'])))
                x=np.r_[ocr_summary(rows,s['image_wh']),mask_summary(prob,s['image_wh'],rows,deployment['map_features'])]
                predicted=float(((x-deployment['mean'])/deployment['scale'])@np.asarray(deployment['coef'])+deployment['intercept'])*(len(rows)+1)
                expected=scores[ds][expert][sid]
                eligible=bool(rows) and all(p['raw_text'].strip() for p in rows)
                if (expected is not None)!=eligible or eligible and abs(predicted-expected)>1e-10:raise ValueError('Independent head score replay failed')
                checks['independent_expert_counts_and_head_replays']+=1
            checks['full_test_image_and_annotation_replays']+=1

def audit_analysis(path,checks):
    result=read(path/'results.json');frozen=read(path/'validation_choices_frozen.json');c=frozen['config']
    if sha(path/'validation_choices_frozen.json')!=result['frozen_choices_sha256'] or frozen['test_used_for_choices'] or not frozen['no_training']:raise ValueError('Changed validation choice binding')
    if sha(path/'results.json')!=read(path/'completion.json')['results_sha256']:raise ValueError('Changed final report')
    for source,digest in result['input_sha256'].items():
        if sha(source)!=digest:raise ValueError('Changed source input: '+source)
        checks['authenticated_analysis_inputs']+=1
    for role in ['validation','test']:
        curves=read(path/(role+'_curves.json'));branches=read(path/(role+'_branches.json'));scores=read(path/(role+'_scores.json'));bands=read(path/(role+'_bands.json'))
        for ds,experts in curves.items():
            for expert,curve in experts.items():
                b=branches[ds][expert];s=scores[ds][expert]
                candidates=[None]+sorted({v for v in s.values() if v is not None},reverse=True)
                if candidates!=[p['threshold'] for p in curve]:raise ValueError('Missing attainable score cutoff')
                for p in curve:
                    compare(independent_point(b,s,p['threshold']),p);checks['independent_'+role+'_cut_replays']+=1
                critical,point95=independent_bootstrap(curve,b,s,c['bootstrap_draws'],c['seed'])
                band=bands[ds][expert]
                if abs(critical-band['simultaneous_critical_value'])>1e-12 or not np.allclose(point95,band['pointwise_gain95'],rtol=0,atol=1e-12):raise ValueError('Independent bootstrap gain-identity replay failed')
                checks['independent_full_bootstrap_bands']+=1
                if role=='validation':
                    policies=frozen['policies'][ds][expert]
                    for loss in c['loss_budgets']:
                        valid=[p for p in curve if rate(p)>=rate(curve[0]) or loss and p['f1']>=curve[0]['f1']-loss]
                        chosen=min(valid,key=lambda p:(p['deepsolo_images'],-rate(p)))
                        if chosen['threshold']!=policies['empirical_loss_'+str(loss)]['threshold']:raise ValueError('Nonoptimal validation no-loss cutoff')
                        valid=[p for p,lower in zip(curve,band['lower_gain']) if lower>=-loss];chosen=min(valid,key=lambda p:(p['deepsolo_images'],-rate(p)))
                        if chosen['threshold']!=policies['bootstrap_band_loss_'+str(loss)]['threshold']:raise ValueError('Nonoptimal conservative validation cutoff')
                        checks['validation_objective_optima']+=2
                    chosen=max(curve,key=lambda p:(rate(p),-p['deepsolo_images']))
                    if chosen['threshold']!=policies['maximum_validation_f1']['threshold']:raise ValueError('Nonoptimal validation F1 cutoff')
                else:
                    for name,p in result['datasets'][ds][expert]['validation_selected'].items():
                        compare(independent_point(b,s,p['threshold']),p['test']);checks['frozen_validation_policy_test_replays']+=1
                    best=max(curve,key=lambda p:(rate(p),-p['deepsolo_images']))
                    if best!=result['datasets'][ds][expert]['test_descriptive_best']:raise ValueError('Nonoptimal descriptive TEST maximum')
                    candidates=[p for p in curve if rate(p)>=rate(curve[0])];chosen=min(candidates,key=lambda p:(p['deepsolo_images'],-rate(p)))
                    if chosen!=result['datasets'][ds][expert]['test_descriptive_minimum_deepsolo_at_no_loss']:raise ValueError('Nonoptimal descriptive no-loss maximum')
    checks['complete_analysis_families']+=1

def audit_matched_head(path,checks):
    deployment=read(path/'deployment.json');meta=read(path/'fit_metadata.json');c=meta['config'];sources=c['paths'];fit=meta['fit_ids']
    if meta['test_used'] or len(fit)!=1028 or sha(path/'deployment.json')!=meta['deployment_sha256']:raise ValueError('Invalid matched-head fitting scope')
    original=read(Path(sources['totaltext_features'])/'fit_feature_schema.json')
    if fit!=original['ids']:raise ValueError('Matched heads must share the original fitting population')
    from recognition.lightweight_unet_router import ocr_summary,mask_summary
    source=Path(sources['totaltext_features']);native=Path(sources['totaltext_native']);yolo=Path(sources['totaltext_yolo'])
    branches=read(native/'train/branches.json')['branches']['yolo'];trace=read(source/'train_feature_inputs.json');feature_plan=read(source/'resolved_configuration.json')
    records={s['sample_id']:s for group in feature_plan['records'].values() for s in group}
    total=Counter()
    for sid in fit:total.update(branches[sid]['deepsolo'])
    reference=2*total['tp']/(total['detected_words']+total['gt_words'])
    with np.load(path/'fit_arrays.npz',allow_pickle=False) as z:x=z['x'];y=z['y'];weights=z['weights']
    if reference!=meta['reference_f1']:raise ValueError('Matched-head reference includes non-fitting labels')
    expected_y=np.array([2*(branches[s]['light']['tp']-branches[s]['deepsolo']['tp'])-reference*(branches[s]['light']['detected_words']-branches[s]['deepsolo']['detected_words']) for s in fit])/weights
    np.testing.assert_array_equal(y,expected_y)
    for index,sid in enumerate(fit):
        p=yolo/'train/light'/(sid.replace(':','__')+'.json');rows=read(p)['rows'];mp=Path(trace[sid]['maps_path'])
        if sha(p)!=meta['input_sha256'][str(p)] or sha(mp)!=meta['input_sha256'][str(mp)]:raise ValueError('Changed matched-head FIT observations')
        with np.load(mp,allow_pickle=False) as z:prob=z['probabilities']
        observed=np.r_[ocr_summary(rows,records[sid]['image_wh']),mask_summary(prob,records[sid]['image_wh'],rows,deployment['map_features'])]
        np.testing.assert_array_equal(observed,x[index])
        if weights[index]!=len(rows)+1:raise ValueError('Incorrect matched-head sample weight')
        checks['matched_head_training_feature_and_target_replays']+=1
    np.testing.assert_allclose(x.mean(0),deployment['mean'],rtol=0,atol=1e-12)
    z=(x-np.asarray(deployment['mean']))/deployment['scale'];mu=np.average(z,axis=0,weights=weights);my=np.average(y,weights=weights);center=z-mu
    coefficient=np.linalg.solve((center*weights[:,None]).T@center+c['fit']['alpha']*np.eye(30),(center*weights[:,None]).T@(y-my))
    error=float(np.max(np.abs(coefficient-deployment['coef'])))
    if error>1e-8 or abs(my-mu@coefficient-deployment['intercept'])>1e-8:raise ValueError('Independent matched-head weighted Ridge solution differs')
    checks['independent_matched_head_normal_equation_solutions']+=1
    # All new-domain scores use exactly the already audited native YOLO rows/maps.
    root=Path(sources['icdar_transfer']);plan=read(root/'resolved_configuration.json')
    from scripts.run_essential_icdar_transfer import yolo_rows
    for role in ['validation','test']:
        scores=read(path/(role+'_scores.json'));phase='calibration' if role=='validation' else 'test'
        for ds,samples in plan['population'][phase].items():
            for s in samples:
                sid=s['sample_id'];u=read(root/'unet'/(sid.replace(':','__')+'.json'))
                with np.load(root/u['maps_path'],allow_pickle=False) as z:prob=z['probabilities']
                rows,_=yolo_rows(root,plan,phase,ds,s);a=np.r_[ocr_summary(rows,s['image_wh']),mask_summary(prob,s['image_wh'],rows,deployment['map_features'])]
                value=float(((a-deployment['mean'])/deployment['scale'])@np.asarray(deployment['coef'])+deployment['intercept'])*(len(rows)+1)
                expected=scores[ds]['yolo'][sid];eligible=bool(rows) and all(r['raw_text'].strip() for r in rows)
                if (expected is not None)!=eligible or eligible and abs(expected-value)>1e-10:raise ValueError('Matched-head source/target prediction replay failed')
                checks['matched_head_icdar_validation_and_test_scores']+=1
    return dict(max_coefficient_error=error,numeric_inference_bytes=deployment['numeric_inference_bytes'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--analysis',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();checks=Counter();verify_icdar_images(args.root,checks);matched=None
    if (args.root/'yolo_head_v1/deployment.json').exists():matched=audit_matched_head(args.root/'yolo_head_v1',checks)
    for path in args.analysis:audit_analysis(path,checks)
    report=dict(status='PASS',mode='approximate',checks=dict(checks),matched_head=matched,analysis_results_sha256={str(p):sha(p/'results.json') for p in args.analysis},
        source_transfer_results_sha256=sha(args.root/'results.json'),analysis_code_sha256=sha(__file__),
        scope='All733 ICDAR images/annotations, selected crop pixels, raw fallback payloads, independent SciPy lexical matching and ignore reconstruction, all attainable validation/TEST cuts, validation optima, frozen-policy TEST counts, independently derived F1-gain-identity bootstrap bands. No independent neural refit or untouched-dataset guarantee.')
    with args.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report))

if __name__=='__main__':main()
