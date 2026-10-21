"""APPROXIMATE image-word evaluation for unlocalized T1 and frozen T3/DeepSolo.

Casefold-exact one-to-one word matching within each image, without bbox IoU.
The same criterion is used for every engine and for replayed CLIP decisions.
Frozen ignore flags and invalid-envelope false positives remain in force.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from recognition.clip_router_evaluation import evaluate_predictions
from recognition.clip_router_threshold_sweep import COUNT_KEYS
from scripts.crop_clip_runtime import sha
from scripts.run_clip_high_recall_admission import payload_digest

PROTOCOL='image_word_multiset_casefold_exact; punctuation retained; no bbox IoU'


def word_metrics(predictions,ground_truth):
    if (any(not isinstance(p['text'],str) or type(p['ignored']) is not bool for p in predictions)
            or any(not isinstance(g['text'],str) for g in ground_truth)):
        raise ValueError('Invalid word-evaluation text/ignore flag')
    care=[p for p in predictions if not p['ignored']]
    remaining=Counter(p['text'].casefold() for p in care
        if p['text'] and p.get('invalid_envelope') is None)
    matched=[]
    for index,g in enumerate(ground_truth):
        key=g['text'].casefold()
        if key and remaining[key]:
            matched.append(index);remaining[key]-=1
    tp=len(matched)
    return {'tp':tp,'fp':len(care)-tp,'fn':len(ground_truth)-tp,'tn':0,
        'detected_words':len(care),'gt_words':len(ground_truth),'all_outputs':len(predictions),
        'ignored_outputs':len(predictions)-len(care),
        'invalid_envelopes':sum(p.get('invalid_envelope') is not None for p in predictions),
        'matched_gt_indices':matched}


def read_word_branches(cfg,root,observed,frozen,t1):
    """Read sealed evaluation artifacts after routing; never run DeepSolo."""
    expected=cfg['word_evaluation_sources'];inputs={}
    def read(name):
        path=root/name
        if sha(path)!=expected[name]:raise ValueError('Frozen word-evaluation source changed: '+name)
        inputs[str(path)]=expected[name]
        return json.loads(path.read_text())
    audit=read('test_independent_audit.json')
    reports={s:read(s+'/results.json') for s in ('train','test')}
    if (audit['status']!='PASS' or audit['plan_sha256']!=cfg['source_plan_sha256']
            or audit['result_sha256']!=expected['test/results.json']):
        raise ValueError('Frozen independent TEST word audit changed')
    ds={r['sample_id']:r for r in audit['deepsolo_only_per_image']}
    if len(ds)!=len(audit['deepsolo_only_per_image']) or set(ds)!={s['sample_id'] for s in observed['test']}:
        raise ValueError('Frozen TEST DeepSolo word population changed')
    splits={}
    for split,samples in observed.items():
        report=reports[split];by_id={r['sample_id']:r for r in report['per_image']}
        if (report['status']!='SUCCESS' or report['plan_sha256']!=cfg['source_plan_sha256']
                or len(by_id)!=len(report['per_image']) or set(by_id)!={s['sample_id'] for s in samples}):
            raise ValueError('Frozen word-evaluation image population changed')
        metrics={r['sample_id']:r for r in frozen[split]};splits[split]=[]
        for sample in samples:
            sid=sample['sample_id'];result=by_id[sid];gt=result['ground_truth']
            path=root/split/'routes'/(sid.replace(':','__')+'.json');route=json.loads(path.read_text())
            if (route['payload_sha256']!=payload_digest(route) or route['observed']!=sample
                    or route['contract']!={'plan_sha256':cfg['source_plan_sha256'],'sample_id':sid,
                        'image_sha256':sample['image_sha256']}):
                raise ValueError('Frozen word-evaluation route changed')
            inputs[str(path)]=sha(path)
            predictions=[{'text':r['raw_text'],'bbox':r['bbox'],
                'ignored':r['instance_id'] in route['ignored_ids']} for r in sample['rows']]
            d_predictions=ds[sid]['predictions'] if split=='test' else (
                result['predictions_all'] if result['action']=='deepsolo' else None)
            if (d_predictions is None)!=(metrics[sid]['deepsolo'] is None):
                raise ValueError('Frozen DeepSolo word availability changed')
            for engine,preds in [('parseq',predictions),('deepsolo',d_predictions)]:
                if preds is None:continue
                geometric=evaluate_predictions(preds,gt,iou_threshold=cfg['geometry_matching_iou'])
                if any(geometric[k]!=metrics[sid][engine][k] for k in COUNT_KEYS):
                    raise ValueError('Frozen geometric branch replay changed: '+sid+'/'+engine)
            raw_t1=t1[sid]['result']['raw_text']
            splits[split].append({'sample_id':sid,'parseq':word_metrics(predictions,gt),
                't1':word_metrics([{'text':raw_t1,'ignored':False,'unlocalized':True}],gt),
                'deepsolo':word_metrics(d_predictions,gt) if d_predictions is not None else None,
                'ground_truth':gt,'T1_raw_text':raw_t1,'metric_protocol':PROTOCOL})
    return {'splits':splits,'input_sha256':inputs,'metric_protocol':PROTOCOL,
        'ground_truth_in_routing':False,'claim_label':'approximate reproduction',
        't1_output_rule':'one raw sequence; no whitespace splitting or fabricated bbox'}


def chosen_branches(rows,features,gate):
    """Choose the metrics of the candidate before thresholding any confidence."""
    return [{**r,'parseq':r['t1'] if gate=='paper_selected'
        and features[r['sample_id']]['selected_source']=='T1' else r['parseq']} for r in rows]


def annotate_candidate_counts(point,features,gate):
    sources={sid:features[sid]['selected_source'] if gate=='paper_selected' else 'T3'
        for sid in point['accepted_sample_ids']}
    point.update(t1_images=sum(s=='T1' for s in sources.values()),
        t3_images=sum(s=='T3' for s in sources.values()),accepted_sample_sources=sources,
        metric_protocol=PROTOCOL)
    if point['combined_model'] is not None:
        point['combined_model'].update(t1_images=point['t1_images'],t3_images=point['t3_images'],
            metric_protocol=PROTOCOL)
    return point
