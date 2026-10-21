"""Cached, exploratory TEST validation of exact reading agreement across640/960."""
import argparse
from pathlib import Path
from collections import Counter
import json
from recognition.yolo_scale_agreement import agreement_controls
from scripts.evaluate_totaltext_yolo_direct_parseq import metrics
from scripts.evaluate_totaltext_yolo_improvements import DEFAULT_CONFIG,DEFAULT_PATHS,REPO
from scripts.crop_clip_runtime import sha
from scripts.analyze_totaltext_crop_word_groups import freeze
from scripts.evaluate_totaltext_yolo_word_slots import read_json
from scripts.run_crop_group_word_selection import evaluate_predictions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--paths',type=Path,default=DEFAULT_PATHS)
    parser.add_argument('--agreement-config',type=Path,default=REPO/'configs/approximate/totaltext_yolo_scale_agreement_v1.json')
    parser.add_argument('--device',choices=('cpu','cuda:0'),default='cuda:0')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--stage',choices=('cached','all'),default='all')
    args=parser.parse_args()
    if args.stage=='cached':return
    cfg=read_json(args.agreement_config)
    if (args.agreement_config.resolve().parent!=REPO/'configs/approximate' or cfg['reproduction_mode']!='approximate'
        or cfg['exploratory_test_analysis'] is not True or not 0<cfg['pair_iou']<=1
        or not 0<cfg['text_nms_iou']<1 or any(not 0<t<1 for t in cfg['joint_controls'])):
        raise ValueError('isolated approximate agreement config required')
    paths=read_json(args.paths)
    run=args.output.resolve() if args.output else Path(paths['output_root'])/args.device.replace(':','_')/('full' if args.limit is None else f'limit{args.limit}')
    plan=read_json(run/'resolved_configuration.json');result=read_json(run/'results.json')
    if result['plan_sha256']!=sha(run/'resolved_configuration.json'):raise RuntimeError('source plan changed')
    baseline=Path(plan['baseline_run']);bp=read_json(baseline/'resolved_configuration.json');gt_source=Path(bp['gt_source'])
    ids=plan['sample_ids'];output=run/'scale_agreement'
    receipt={'schema_version':1,'reproduction_mode':'approximate','exploratory_test_analysis':True,
             'config':cfg,'source_result_sha256':sha(run/'results.json'),'sample_ids':ids,
             'code_sha256':{name:sha(REPO/name) for name in ('scripts/evaluate_totaltext_yolo_scale_agreement.py','recognition/yolo_scale_agreement.py','recognition/yolo_word_refinement.py')},
             'ground_truth_use':'only after selections frozen; no new neural inference'}
    freeze(output/'resolved_configuration.json',receipt)
    for sid in ids:
        key=sid.replace(':','__');a=read_json(baseline/'ocr'/f'{key}.json')['rows'];b=read_json(run/'ocr/high960'/f'{key}.json')['rows']
        freeze(output/'selection'/f'{key}.json',{'sample_id':sid,'methods':agreement_controls(a,b,cfg),
              'base_sha256':sha(baseline/'ocr'/f'{key}.json'),'high_sha256':sha(run/'ocr/high960'/f'{key}.json'),'ground_truth_use':'none'})
    totals={};gt_count=0
    for sid in ids:
        key=sid.replace(':','__');gt=read_json(gt_source/'evaluation'/f'{key}.json')['ground_truth'];gt_count+=len(gt)
        for name,rows in read_json(output/'selection'/f'{key}.json')['methods'].items():
            score=evaluate_predictions(rows,gt,plan['config']['matching_iou']);totals.setdefault(name,Counter()).update(outputs=len(rows),correct=score['correct'])
    report={'schema_version':1,'reproduction_mode':'approximate','exploratory_test_analysis':True,
            'images':len(ids),'gt_words':gt_count,'plan_sha256':sha(output/'resolved_configuration.json'),
            'summary':{name:metrics(t['correct'],t['outputs'],gt_count) for name,t in totals.items()},
            'limitation':'local bbox proxy; exploratory TEST method, not independent validation'}
    freeze(output/'results.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
