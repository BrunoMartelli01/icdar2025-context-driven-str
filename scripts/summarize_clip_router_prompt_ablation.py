"""Summarize sealed prompt experiments; choose model only by TRAIN45 evidence."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.crop_clip_runtime import sha, write_json

REPO=Path(__file__).resolve().parents[1]


def summary(paths, model_names, output):
    profile=json.loads(paths.read_text());root=Path(profile['output_root'])
    loaded={};rows=[];selection=[]
    for name in model_names:
        pointer=json.loads((root/name/'latest.json').read_text());path=Path(pointer['results_path'])
        if sha(path)!=pointer['results_sha256']:
            raise ValueError('Experiment result checksum differs')
        result=json.loads(path.read_text());chosen_path=path.parent/'train_selection.json'
        plan_path=path.parent/'resolved_configuration.json'
        if (result['model']!=name or result['status']!='SUCCESS'
                or result['result_class']!='approximate reproduction'
                or result['reproduction_mode']!='approximate'
                or result['selection_uses_test_labels'] is not False
                or sha(chosen_path)!=result['train_selection_sha256']
                or sha(plan_path)!=result['plan_sha256']):
            raise ValueError('A complete sealed approximate experiment is required')
        chosen=json.loads(chosen_path.read_text());plan=json.loads(plan_path.read_text())
        loaded[name]={'results_path':str(path),'results_sha256':sha(path),'plan':plan,'selection':chosen,'result':result}
        winner=chosen['winner'];val=chosen['families'][winner]['validation']
        selection.append({'model':name,'family':winner,'validation_precision':val['precision_retained_parseq'],
            'validation_words':val['words_retained_parseq']})
        for family, splits in result['comparisons'].items():
            for split, protocols in splits.items():
                for protocol,p in protocols.items():
                    m=p['combined_model']
                    rows.append({'model':name,'family':family,'split':split,'protocol':protocol,
                        'threshold':p['threshold'] if protocol=='train_threshold' else None,
                        'parseq_images':p['parseq_images'],'parseq_words':p['words_retained_parseq'],
                        'parseq_tp':p['correct_words_retained_parseq'],
                        'parseq_precision':p['precision_retained_parseq'],'safe_images':p['safe_images'],
                        'complete_images':p['complete_images'],
                        'missing_deepsolo':len(p['missing_deepsolo_sample_ids']),
                        'final_words':m['detected_words'] if m else None,
                        'final_precision':m['accuracy_on_detected_words'] if m else None,
                        'final_accuracy_tn0':m['accuracy_tn0'] if m else None,
                        'final_recall':m['recall'] if m else None,'final_f1':m['word_f1'] if m else None})
    # Require an identical scientific protocol/source on every model/device.
    first=next(iter(loaded.values()))['plan']
    for entry in loaded.values():
        for field in ('config','source_sha256','branch_metrics_sha256','code_sha256'):
            if entry['plan'][field]!=first[field]:
                raise ValueError('Models did not use the same frozen protocol: '+field)
    winner=max(selection,key=lambda s:(s['validation_precision'] or -1,s['validation_words'],s['model']))
    output.mkdir(parents=True,exist_ok=True)
    report={'status':'SUCCESS','reproduction_mode':'approximate','claim_label':'approximate reproduction',
        'table_5_claim':False,'winner_train':winner,'selection_uses_test_labels':False,
        'selection_is_exploratory':True,'models':{n:{k:v for k,v in x.items() if k in ('results_path','results_sha256')} for n,x in loaded.items()},
        'same_budget_is_descriptive':True,'rows':rows,
        'code_sha256':{'scripts/summarize_clip_router_prompt_ablation.py':sha(Path(__file__))}}
    write_json(output/'summary.json',report)
    with (output/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# CLIP router prompts — approximate reproduction','',
        'Model/family chosen on TRAIN45 only: '+winner['model']+' / '+winner['family']+'.',
        'TRAIN125 is a previously analysed held-out subset; TEST300 was previously inspected with the old router. This is an exploratory comparison, not a new blind benchmark.',
        '', 'Matched image-budget comparisons (62 TRAIN / 149 TEST; GT-free score ranking with ID tie breaks).',
        'Final precision = TP/(TP+FP); accuracy TN0 = TP/(TP+FP+FN). Missing fallback keeps final metrics unavailable.', '',
        '| Model | Prompt/selector | Split | Accepted images | PARSeq words | PARSeq precision % | Final words | Final precision % | Final TN0 % |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    fmt=lambda v:'—' if v is None else f'{100*v:.2f}'
    for r in rows:
        if r['protocol']!='same_budget_descriptive':continue
        lines.append(f"| {r['model']} | {r['family']} | {r['split'].upper()} | {r['parseq_images']} | {r['parseq_words']} | {fmt(r['parseq_precision'])} | {r['final_words'] if r['final_words'] is not None else '—'} | {fmt(r['final_precision'])} | {fmt(r['final_accuracy_tn0'])} |")
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'winner_train':winner,'summary':str(output/'summary.md'),'csv':str(output/'summary.csv')}),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paths',type=Path,default=REPO/'configs/approximate/clip_router_prompt_ablation_local_paths_v2.json')
    p.add_argument('--models',nargs='+',required=True);p.add_argument('--output',type=Path,default=Path('/tmp/clip-router-prompt-ablation-v2/summary'))
    a=p.parse_args();summary(a.paths,a.models,a.output)


if __name__=='__main__':main()
