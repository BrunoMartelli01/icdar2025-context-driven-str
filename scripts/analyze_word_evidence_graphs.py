"""Independent word-F1 audit and failure accounting for evidence graph experiments."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from recognition.selective_word_repair import exact_matching
from recognition.text_reconstruction import box_iou
from scripts.run_word_evidence_graphs import load,log
from scripts.run_text_reconstruction_tests import read,write,sha


def paired(samples,left,right,c,iterations):
    arrays=[];recovered=[];lost=[];improved=worsened=0
    for s in samples:
        sid=s['sample_id'];a=exact_matching(left[sid],s['words'],c['matching_iou'])
        b=exact_matching(right[sid],s['words'],c['matching_iou'])
        x,y=set(a['matched_gt_indices']),set(b['matched_gt_indices'])
        recovered.extend(dict(sample_id=sid,gt_index=k,text=s['words'][k]['text']) for k in sorted(y-x))
        lost.extend(dict(sample_id=sid,gt_index=k,text=s['words'][k]['text']) for k in sorted(x-y))
        n=len(s['words']);fa=2*a['correct']/max(1,len(left[sid])+n);fb=2*b['correct']/max(1,len(right[sid])+n)
        improved+=fb>fa;worsened+=fb<fa
        arrays.append([a['correct'],len(left[sid])+n,b['correct'],len(right[sid])+n])
    arrays=np.asarray(arrays);total=arrays.sum(0);delta=2*total[2]/total[3]-2*total[0]/total[1]
    rng=np.random.default_rng(c['seed']);draws=rng.integers(0,len(arrays),(iterations,len(arrays)))
    sums=arrays[draws].sum(1);distribution=2*sums[:,2]/sums[:,3]-2*sums[:,0]/sums[:,1]
    return dict(delta_f1_points=float(delta*100),conditional_image_bootstrap_95_points=(np.quantile(distribution,[.025,.975])*100).tolist(),
        recovered=len(recovered),lost=len(lost),recovered_words=recovered,lost_words=lost,
        images_improved=int(improved),images_worsened=int(worsened),images_unchanged=len(samples)-int(improved)-int(worsened))


def analyze(root,c,smoke):
    data=load(root);results={};checks=0;errors=[]
    for phase in ('evaluation','test'):
        baseline=read(root/f'{phase}_baseline_filter_predictions.json')
        graph=read(root/f'{phase}_baseline_gnn_predictions.json')
        results[phase]={}
        for path in sorted(root.glob(phase+'_*_results.json')):
            name=path.name[len(phase)+1:-len('_results.json')]
            predictions=read(root/f'{phase}_{name}_predictions.json');saved=read(path)
            rows={r['sample_id']:r for r in saved['per_image']};counts=[0,0,0]
            for s in data[phase]:
                sid=s['sample_id'];pool=predictions[sid];words=s['words']
                matrix=np.array([[box_iou(p['bbox'],w['bbox'])>=c['matching_iou'] and
                    p['text'].casefold()==w['text'].casefold() for w in words] for p in pool],bool).reshape(len(pool),len(words))
                a,b=linear_sum_assignment(matrix.astype(np.int8),maximize=True);correct=int(matrix[a,b].sum())
                row=rows[sid]['spatial']
                if (correct,len(pool),len(words))!=(row['correct'],row['predictions'],row['gt_words']):
                    errors.append([phase,name,sid,'independent matching differs'])
                counts=np.asarray(counts)+[correct,len(pool),len(words)];checks+=1
                if not name.startswith('baseline'):
                    prefix=graph[sid] if name.startswith(('consensus','whole_reading','active','fixed_queries')) else baseline[sid]
                    if pool[:len(prefix)]!=prefix:errors.append([phase,name,sid,'baseline prefix changed'])
            value=saved['aggregate']['spatial']
            if counts.tolist()!=[value['correct'],value['predictions'],value['gt_words']]:errors.append([phase,name,'aggregate differs'])
            results[phase][name]=dict(metric=value,versus_filter=paired(data[phase],baseline,predictions,c,c['analysis']['bootstrap_iterations']),
                versus_gnn=paired(data[phase],graph,predictions,c,c['analysis']['bootstrap_iterations']))
    for path in (root/'active').glob('*trace.json'):
        for sid,rows in read(path).items():
            if len(rows)>c['active']['query_budget']:errors.append([str(path),sid,'query budget'])
    for path in (root/'active').glob('*/*trace.json'):
        for sid,rows in read(path).items():
            if len(rows)>c['active']['query_budget']:errors.append([str(path),sid,'query budget'])
    write(root/'independent_audit.json',dict(status='FAIL' if errors else 'PASS',image_arm_checks=checks,errors=errors))
    if errors:raise ValueError('Independent scientific audit failed')
    # Attribution for actual accepted character reconstructions, not oracle F1.
    candidates=read(root/'consensus/consensus/candidates.json');diagnostics={}
    for phase in ('evaluation','test'):
        selected=read(root/f'{phase}_consensus_predictions.json');rows=[]
        for s in data[phase]:
            sid=s['sample_id'];details={p['id']:p for p in candidates[phase][sid]}
            for p in selected[sid][len(s['baseline_gnn']):]:
                if p['id'] not in details:raise ValueError('Consensus provenance missing')
                entry=details[p['id']];matches=[k for k,w in enumerate(s['words']) if box_iou(p['bbox'],w['bbox'])>=.5 and p['text'].casefold()==w['text'].casefold()]
                absent=all(not any(box_iou(q['bbox'],s['words'][k]['bbox'])>=.5 and q['text'].casefold()==s['words'][k]['text'].casefold()
                    for q in s['pool']) for k in matches) if matches else False
                rows.append(dict(sample_id=sid,**entry,exact_gt_indices=matches,correct_word_absent_from_all_original_readings=absent))
        diagnostics[phase]=dict(accepted=len(rows),accepted_changed=sum(x['changed_characters']>0 for x in rows),
            correct_novel_reconstructions=sum(bool(x['exact_gt_indices']) and x['correct_word_absent_from_all_original_readings'] for x in rows),per_candidate=rows)
    write(root/'consensus_attribution.json',diagnostics)
    write(root/'results.json',dict(status='SUCCESS',claim_label='approximate reproduction',smoke=smoke,
        metric='word F1, bbox IoU>=.5 + exact Unicode casefold; punctuation retained',results=results,
        inherited_calibration_exposure=True,historically_explored_test=True,independent_generalization=False,
        bootstrap_scope='conditional image sampling only; no seed/selection variability'))
    lines=['# Word evidence graphs: completed comparisons','','Approximate reproduction; exploratory historical populations. No automatic promotion.',
        '', '| Arm | TEST word F1 % | Delta vs filter, points | Delta vs prior GNN, points | Recovered / lost vs filter | FP |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,x in results['test'].items():
        a=x['metric'];d=x['versus_filter'];g=x['versus_gnn']
        lines.append(f"| {name} | {100*a['f1']:.4f} | {d['delta_f1_points']:+.4f} | {g['delta_f1_points']:+.4f} | {d['recovered']} / {d['lost']} | {a['fp']} |")
    lines += ['', 'Selected and raw-final endpoints are separate. KEEP means the original prefix was selected on calibration.',
        'Typed/ownership use the <=20 original crops. Character alignment adds teacher-forcing decoder computation but no new RGB crops.',
        'Active observation permits <=2 additional crop reads per image; compare its fixed-query and self controls, including all rejected reads.',
        'Per-image outputs, exact word changes, conditional intervals and provenance are in results.json and the individual reports.']
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    artifacts={str(p.relative_to(root)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(root.rglob('*'))
        if p.is_file() and p.name not in ('status.json','artifact_manifest.json')}
    write(root/'artifact_manifest.json',dict(status='PASS',artifacts=artifacts))
    log(root,'ANALYSIS_COMPLETE',checks=checks,artifacts=len(artifacts))
