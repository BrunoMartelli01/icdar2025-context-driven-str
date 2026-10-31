"""Independent matching, paired word attribution, costs and artifact hashes."""
from __future__ import annotations
import json
from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from recognition.text_reconstruction import box_iou
from scripts.run_grounded_word_recovery import load
from scripts.analyze_word_evidence_graphs import paired
from scripts.run_text_reconstruction_tests import read,write,sha


def analyze(root,c):
    data=load(root);results={};checks=0
    for phase in ('evaluation','test'):
        reference=read(root/f'{phase}_baseline_gnn_predictions.json')
        verified=read(root/f'{phase}_verifier_replacement_predictions.json')
        results[phase]={}
        for path in sorted(root.glob(phase+'_*_results.json')):
            name=path.name[len(phase)+1:-len('_results.json')]
            predictions=read(root/f'{phase}_{name}_predictions.json');saved=read(path)
            per_image={row['sample_id']:row['spatial'] for row in saved['per_image']}
            aggregate=np.zeros(3,np.int64)
            if set(predictions)!={s['sample_id'] for s in data[phase]}:raise ValueError('Prediction population changed')
            for s in data[phase]:
                rows=predictions[s['sample_id']];words=s['words']
                matrix=np.asarray([[box_iou(p['bbox'],w['bbox'])>=c['matching_iou'] and
                    p['text'].casefold()==w['text'].casefold() for w in words] for p in rows],bool).reshape(len(rows),len(words))
                a,b=linear_sum_assignment(matrix.astype(np.int8),maximize=True)
                value=np.asarray([int(matrix[a,b].sum()),len(rows),len(words)])
                original=per_image[s['sample_id']]
                if value.tolist()!=[original['correct'],original['predictions'],original['gt_words']]:
                    raise ValueError('Independent word matching differs: '+phase+'/'+name+'/'+s['sample_id'])
                aggregate+=value;checks+=1
            actual=saved['aggregate']['spatial']
            if aggregate.tolist()!=[actual['correct'],actual['predictions'],actual['gt_words']]:raise ValueError('Aggregate differs')
            results[phase][name]=dict(metric=actual,versus_gnn=paired(data[phase],reference,predictions,c,c['bootstrap_iterations']),
                versus_verified=paired(data[phase],verified,predictions,c,c['bootstrap_iterations']))
    query_checks=0;costs={}
    for path in sorted((root/'search').rglob('*trace.json')):
        count=Counter();seen=set()
        for sid,trace in read(path).items():
            if len(trace)>c['search']['query_budget']:raise ValueError('Query cap exceeded')
            if len({r['action']['id'] for r in trace})!=len(trace):raise ValueError('An action was repeated')
            query_checks+=1
            for row in trace:
                count['queries']+=1;count[row['action']['kind']+'_queries']+=1
                count['changed_decisions']+=row['changed']
                p=row['outcome']['proposal']
                if p is None:count['no_novel_geometry']+=1;continue
                count['native_reads']+=1;seen.add((sid,p['crop_array_sha256']))
        costs[str(path.relative_to(root))]=dict(**count,distinct_native_reads=len(seen))
    write(root/'query_costs.json',costs)
    matched={}
    for phase in ('evaluation','test'):
        matched[phase]={}
        for method,control in [('verifier_replacement','confidence_replacement'),('verifier_replacement','verifier_addonly'),
            ('search_gnn_matched','search_fixed_matched'),('search_gnn_selected','search_self_selected'),
            ('editor_visual_selected','editor_text_selected'),('editor_visual_final','editor_text_final')]:
            matched[phase][method+'_versus_'+control]=paired(data[phase],read(root/f'{phase}_{control}_predictions.json'),
                read(root/f'{phase}_{method}_predictions.json'),c,c['bootstrap_iterations'])
    write(root/'matched_comparisons.json',matched)
    write(root/'results.json',dict(label='approximate reproduction / historically explored evaluation',results=results))
    write(root/'independent_audit.json',dict(status='PASS',image_arm_checks=checks,query_trace_checks=query_checks,
        criterion='Independent SciPy maximum assignment counts; all care words/outputs; query cap/uniqueness'))
    lines=['# Three grounded word recovery experiments','','Approximate reproduction; exploratory historical splits; one seed.',
        '', 'Primary reference: previous calibrated GNN, not official paper recognition accuracy.',
        '', '| TEST arm | Word F1 % | Delta vs GNN, points | Recovered/lost | FP delta |',
        '| --- | ---: | ---: | ---: | ---: |']
    baseline=results['test']['baseline_gnn']['metric']
    for name,item in results['test'].items():
        m=item['metric'];d=item['versus_gnn']
        lines.append(f"| {name} | {100*m['f1']:.4f} | {d['delta_f1_points']:+.4f} | {d['recovered']}/{d['lost']} | {m['fp']-baseline['fp']:+d} |")
    lines+=['','## Matched effects','']
    for name,item in matched['test'].items():
        lo,hi=item['conditional_image_bootstrap_95_points']
        lines.append(f"- {name}: {item['delta_f1_points']:+.4f} points; conditional95% [{lo:+.4f}, {hi:+.4f}].")
    lines+=['','## Scope and costs','','Selected and trained-final endpoints are distinct. Identity/KEEP is eligible on calibration only.',
        'Editors begin from the frozen verified-native output; query methods share that same prefix and admission model.',
        'Word recovery/loss IDs, per-image results, all query attempts and byte edit scripts are saved separately.',
        'Native OCR crop budgets exclude neither rejected queries nor old-run cache hits outside the20-crop source pool.',
        'Encoder-only passes, tile segmentation calls and total all-policy preparation costs are separately recorded.',
        'Conditional bootstrap intervals omit training/search variability. Known identical TRAIN/TEST RGBs and inherited calibration exposure remain.',
        'No strict reproduction, independent validation, total latency advantage, or novelty-priority claim is made.',
        '',f'Independent audit: {checks} image/arm counts and {query_checks} image query traces PASS.']
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    artifacts={str(p.relative_to(root)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(root.rglob('*'))
               if p.is_file() and p.name not in ('artifact_manifest.json','status.json')}
    write(root/'artifact_manifest.json',dict(status='PASS',artifacts=artifacts))
