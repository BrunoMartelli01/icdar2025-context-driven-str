"""Audit a completed approximate coverage/resolution experiment without model inference."""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from scripts import run_clip_router_e2e as source
from scripts import run_yolo_router_comparison as base
from scripts.crop_clip_runtime import array_sha, sha
from scripts.run_clip_high_recall_admission import seal

REPO = Path(__file__).resolve().parents[1]
COUNTS = ('tp','fp','fn','detected_words','gt_words','all_outputs','ignored_outputs')


def midcdf(reference, value):
    return (bisect_left(reference,value)+bisect_right(reference,value))/(2*len(reference))


def project(image_wh, boxes, side):
    w,h = image_wh
    im = Image.new('L',(w,h),0)
    for x1,y1,x2,y2 in boxes:
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
        ImageDraw.Draw(im).rectangle((x1,y1,x2-1,y2-1),fill=255)
    canvas = Image.new('L',(max(w,h),)*2,0)
    canvas.paste(im,((max(w,h)-w)//2,(max(w,h)-h)//2))
    return np.asarray(canvas.resize((side,side),Image.Resampling.NEAREST)) != 0


def counts(rows, ground_truth, validity, image_wh):
    care=[]
    for r in rows:
        support=project(image_wh,[r['bbox']],validity.shape[0])
        if not (support.any() and not (support & validity).any()):care.append(r)
    pc=Counter(r['raw_text'].casefold() for r in care if r['raw_text'])
    gc=Counter(g['text'].casefold() for g in ground_truth if g['text'])
    tp=sum((pc & gc).values())
    return {'tp':tp,'fp':len(care)-tp,'fn':len(ground_truth)-tp,
        'detected_words':len(care),'gt_words':len(ground_truth),
        'all_outputs':len(rows),'ignored_outputs':len(rows)-len(care)}


def combine(branches, ids, accepted):
    c={k:sum(branches[s]['light' if s in accepted else 'deepsolo'][k] for s in ids) for k in COUNTS}
    tp,fp,fn=(c[k] for k in ('tp','fp','fn'))
    return {**c,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        'accuracy_tn0':tp/(tp+fp+fn) if tp+fp+fn else None}


def check_counts(actual,expected):
    for k in COUNTS:assert actual[k] == expected[k], (k,actual,expected)


def bootstrap(branches, ids, left, right, iterations, seed):
    a=np.array([[branches[s]['light' if s in left else 'deepsolo'][k] for k in ('tp','fp','fn')] for s in ids])
    b=np.array([[branches[s]['light' if s in right else 'deepsolo'][k] for k in ('tp','fp','fn')] for s in ids])
    rng=np.random.default_rng(seed);delta_f1=[];delta_tn=[]
    for start in range(0,iterations,500):
        index=rng.integers(0,len(ids),size=(min(500,iterations-start),len(ids)))
        aa=a[index].sum(axis=1);bb=b[index].sum(axis=1)
        delta_f1.extend(2*aa[:,0]/(2*aa[:,0]+aa[:,1]+aa[:,2])-2*bb[:,0]/(2*bb[:,0]+bb[:,1]+bb[:,2]))
        delta_tn.extend(aa[:,0]/aa.sum(axis=1)-bb[:,0]/bb.sum(axis=1))
    la=combine(branches,ids,left);ra=combine(branches,ids,right)
    return {'iterations':iterations,'seed':seed,'unit':'paired images; fixed fitted gates; exploratory; no multiplicity correction',
        'delta_f1':la['f1']-ra['f1'],'f1_ci95':np.quantile(delta_f1,[.025,.975]).tolist(),
        'delta_tn0':la['accuracy_tn0']-ra['accuracy_tn0'],'tn0_ci95':np.quantile(delta_tn,[.025,.975]).tolist()}


def run(args):
    if not __debug__:
        raise RuntimeError('Run this independent audit without Python -O; all assertions are required')
    root=args.run.resolve();plan=base.read(root/'resolved_configuration.json');result=base.read(root/'results.json')
    assert result['status']=='SUCCESS' and plan['result_class']=='APPROXIMATE_EXPLORATORY_COMPARISON'
    cfg=plan['config'];ph=sha(root/'resolved_configuration.json')
    for name,digest in plan['code_sha256'].items():
        assert sha(root/'code_snapshot'/name)==digest
    for name,digest in plan['cached_input_sha256'].items():assert sha(Path(name))==digest
    qualified=source.qualify(SimpleNamespace(mode='approximate',config=REPO/plan['paths']['qualification_config'],
        paths=REPO/plan['paths']['qualification_paths'],output_root=None))[-1]
    data={r['observed']['sample_id']:r for rows in qualified.values() for r in rows}
    fallback=base.checked(Path(plan['paths']['vitae_root'])/'results.json',cfg['vitae_results_sha256'])
    fallback={split:{r['sample_id']:r['word'] for r in fallback['splits'][split]['per_image']} for split in ('train','test')}
    summary={'status':'PASS','claim_label':'approximate reproduction','run':str(root),'plan_sha256':ph,
        'results_sha256':sha(root/'results.json'),'analysis_code_sha256':sha(Path(__file__)),
        'metric':result['metric'],'test_labels_used_for_fit':False,'images':{'train':125,'test':300},
        'resolutions':{}}
    table=[];checked_features=checked_crops=0
    summary['original_resolution_audit']={split:{
        'images':len(samples),'larger_than_640':sum(max(s['image_wh'])>640 for s in samples),
        'larger_than_960':sum(max(s['image_wh'])>960 for s in samples),
        'larger_than_1280':sum(max(s['image_wh'])>1280 for s in samples)}
        for split,samples in plan['samples'].items()}
    for resolution in cfg['resolutions']:
        fit=base.read(root/str(resolution)/'frozen_train_selector.json')
        assert fit['fit_ids']==plan['calibration_ids'] and fit['test_labels_used'] is False
        assert fit['current_test_features_present_at_freeze']==0
        assert set(fit['fit_ids']).isdisjoint(plan['validation_ids'])
        item=result['resolutions'][str(resolution)];features={};branches={};values={};shapes=set();peaks=[]
        for split,samples in plan['samples'].items():
            assert len(samples)==summary['images'][split]
            features[split]={};branches[split]={}
            for s in samples:
                sid=s['sample_id'];key=base.key(sid)+'.json'
                f=base.checked_shard(root/str(resolution)/split/'features'/key,ph)
                o=base.checked_shard(root/str(resolution)/split/'observations'/key,ph)
                assert f['sample_id']==sid and f['ground_truth_used'] is False
                assert f['rows']==o['rows'] if resolution==640 else all(
                    {k:v for k,v in r.items() if k!='clip_word_advantage'}==old for r,old in zip(f['rows'],o['rows']))
                assert len(f['rows'])==len(o['rows'])
                assert sha(Path(s['image_path']))==s['image_sha256']
                with Image.open(s['image_path']) as im:rgb=np.asarray(im.convert('RGB'))
                for r in f['rows']:
                    x1,y1,x2,y2=r['bbox'];assert array_sha(rgb[y1:y2,x1:x2])==r['crop_rgb_sha256']
                    checked_crops+=1
                mp=Path(s['map_path']);assert sha(mp)==f['map_sha256']
                with np.load(mp,allow_pickle=False) as z:p=z['probabilities']
                side=p.shape[1];w,h=s['image_wh'];canvas=project((w,h),[[0,0,w,h]],side)
                union=project((w,h),[r['bbox'] for r in f['rows']],side)
                for channel,name in enumerate(['text','kernel']):
                    weights=np.where((p[channel]>=cfg['coverage_probability_threshold']) & canvas,p[channel],0).astype(np.float64)
                    cov=float(weights[union].sum()/weights.sum()) if weights.sum() else 1.
                    assert cov==f['coverage'][name+'_coverage']
                b={'light':counts(f['rows'],data[sid]['ground_truth'],data[sid]['validity'],(w,h)),
                   'deepsolo':fallback[split][sid]}
                for branch in b:check_counts(b[branch],item['branch_metrics'][split][sid][branch])
                features[split][sid]=f;branches[split][sid]=b;checked_features+=1
                receipt=o['execution']
                shapes.update(tuple(v) for v in receipt.get('yolo_input_shapes',[]))
                for peak in (receipt.get('detector_peak'),f.get('clip_peak')):
                    if peak:peaks.append(peak)
        train=features['train'];reference=fit['confidence_reference']
        fitrows=[r for s in fit['fit_ids'] for r in train[s]['rows']]
        assert reference=={'native':sorted(r['confidence'] for r in fitrows),
            'word_advantage':sorted(r['clip_word_advantage'] for r in fitrows)}
        win=fit['winner'];assert win['reference']==sorted(train[s]['coverage'][win['coverage_field']] for s in fit['fit_ids'])
        assert (item['selected_weight'],item['selected_coverage_field'])==(win['weight'],win['coverage_field'])
        for split,ff in features.items():
            values[split]={}
            for s,f in ff.items():
                rows=f['rows']
                confidence=min((midcdf(reference['native'],r['confidence'])*midcdf(reference['word_advantage'],r['clip_word_advantage']))**.5
                    for r in rows) if rows else None
                coverage=(confidence if win['weight']==0 else (1-win['weight'])*confidence+win['weight']*
                    midcdf(win['reference'],f['coverage'][win['coverage_field']])) if confidence is not None else None
                values[split][s]={'confidence':confidence,'coverage':coverage}
        for split,ids in [('train',list(train)),('calibration',fit['fit_ids']),('validation',plan['validation_ids']),
                ('test',list(features['test']))]:
            origin='test' if split=='test' else 'train';bb=branches[origin]
            for name,point in item[split]['points'].items():
                family,rule=name.split('/');mode,fraction=rule.split('_');fraction=float(fraction)
                vv={s:values[origin][s][family] for s in ids}
                ranked=sorted((s for s in ids if vv[s] is not None),key=lambda s:(-vv[s],s))
                accepted=set(ranked[:round(len(ids)*fraction)]) if mode=='budget' else {
                    s for s in ids if vv[s] is not None and point['threshold'] is not None and vv[s]>=point['threshold']}
                assert accepted==set(point['accepted_ids'])
                actual=combine(bb,ids,accepted);check_counts(actual,point['combined'])
                assert actual['accuracy_tn0']==point['combined']['accuracy_tn0'] and actual['f1']==point['combined']['f1']
                table.append({'resolution':resolution,'split':split,'gate':family,'rule':mode,'target_bypass':fraction,
                    'accepted_images':len(accepted),'fallback_images':len(ids)-len(accepted),
                    'threshold':point['threshold'],**actual})
        ids=list(features['test']);points=item['test']['points']
        ci=bootstrap(branches['test'],ids,set(points['coverage/budget_0.5']['accepted_ids']),
            set(points['confidence/budget_0.5']['accepted_ids']),args.bootstrap_iterations,args.seed)
        summary['resolutions'][str(resolution)]={'selected_weight':win['weight'],'selected_field':win['coverage_field'],
            'bootstrap_vs_confidence_at_half':ci,'yolo_input_shapes':sorted(shapes),
            'peak_allocated_bytes':max((p['peak_allocated_bytes'] for p in peaks),default=None),
            'peak_reserved_bytes':max((p['peak_reserved_bytes'] for p in peaks),default=None)}
        print(f'[AUDIT {resolution}] images=425, word counts, rank budgets, scalar cuts and original crop hashes PASS',flush=True)
    summary['checked_feature_images']=checked_features;summary['checked_original_crops']=checked_crops
    destination=root/'analysis';seal(destination/'verification.json',summary)
    csvpath=destination/'comparison.csv'
    if not csvpath.exists():
        with csvpath.open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    else:
        with csvpath.open(newline='') as stream:
            assert list(csv.DictReader(stream))==[{k:str(v) if v is not None else '' for k,v in r.items()} for r in table]
    lines=['# Coverage and YOLO resolution: completed exploratory comparison','',
        '**Approximate reproduction.** All125 TRAIN and300 TEST images; no omitted sample. ',
        'Accuracy TN0 = TP/(TP+FP+FN); casefold-exact within-image word multiset, punctuation retained, no IoU.',
        'The common ViTAEv2-S+TextOCR DeepSolo branch and evaluation-only native256 ignore remain fixed.',
        'These are local diagnostic scores, not the official paper benchmark protocol.','',
        '## Primary comparison: exactly50% images bypass DeepSolo','',
        '| Resolution / router | TRAIN TN0 | TRAIN45 validation TN0 | TEST TN0 | TEST F1 | TEST predicted words |',
        '|---|---:|---:|---:|---:|---:|']
    for res,item in result['resolutions'].items():
        for family in ['confidence','coverage']:
            rows=[item[s]['points'][family+'/budget_0.5']['combined'] for s in ['train','validation','test']]
            lines.append(f'| {res} / {family} | {rows[0]["accuracy_tn0"]*100:.4f}% | {rows[1]["accuracy_tn0"]*100:.4f}% | '
                f'{rows[2]["accuracy_tn0"]*100:.4f}% | {rows[2]["f1"]*100:.4f}% | {rows[2]["detected_words"]} |')
    lines.extend(['','TRAIN125 contains the80 calibration images;45 are separately held out. Upstream DeepSolo saw TotalTextTRAIN.',
        'Field/weight were chosen only on TRAIN80, before TEST features; all populations are exploratory.','',
        '## Bypass and scalar threshold controls, primary640','',
        '| Policy | Target bypass | Threshold | Actual TEST bypass | TEST TN0 | TEST F1 | Predicted words |',
        '|---|---:|---:|---:|---:|---:|---:|'])
    for name,point in result['resolutions']['640']['test']['points'].items():
        if not name.startswith('coverage/'):continue
        c=point['combined'];threshold='batch ranking' if point['threshold'] is None else f'{point["threshold"]:.9f}'
        lines.append(f'| {name} | {name.rsplit("_",1)[-1]} | {threshold} | {point["accepted_images"]}/300 | '
            f'{c["accuracy_tn0"]*100:.4f}% | {c["f1"]*100:.4f}% | {c["detected_words"]} |')
    ci=summary['resolutions']['640']['bootstrap_vs_confidence_at_half']
    lines.extend(['','## Uncertainty and audit','',
        f'At640 and50% bypass: coverage-minus-confidence TN0 {ci["delta_tn0"]*100:+.4f}pp, '
        f'paired-image95% interval [{ci["tn0_ci95"][0]*100:+.4f}, {ci["tn0_ci95"][1]*100:+.4f}]pp; '
        f'F1 {ci["delta_f1"]*100:+.4f}pp, interval [{ci["f1_ci95"][0]*100:+.4f}, {ci["f1_ci95"][1]*100:+.4f}]pp. ',
        f'{args.bootstrap_iterations} bootstrap draws; gates fixed; no multiplicity correction. An interval crossing zero does not establish superiority.','',
        f'Independent checks: {checked_features} image-feature records, {checked_crops} original RGB crop hashes, '
        'per-image word counts, predicted-map coverage, TRAIN references, rank-budget decisions and transferred scalar cuts.',
        'GPU stage peaks/shapes and source/result/code hashes are in verification.json. Runtime sessions preserve co-resident processes.',
        'New larger-resolution YOLO uses CUDA; cached960 TEST used CPU. No isolated resolution-causality or8GB qualification claim.',
        'Full runtime of the new coverage gate, including generating U-Net maps, was not benchmarked.','',
        f'Run: `{root}`. Full curves: comparison.csv; source/selector/features/results remain sealed.'])
    report='\n'.join(lines)+'\n';report_path=destination/'report.md'
    if report_path.exists():assert report_path.read_text()==report
    else:report_path.write_text(report)
    print(json.dumps(summary,sort_keys=True),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--bootstrap-iterations',type=int,default=10000)
    parser.add_argument('--seed',type=int,default=20261021)
    run(parser.parse_args())
