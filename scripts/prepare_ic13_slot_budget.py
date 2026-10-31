"""Validation-only raw-slot admission and post-correction canonical text NMS."""

import argparse
import math
from pathlib import Path

from localize.independent_word_slots import overlap_iou
from recognition.deepsolo_lexicon import DeepSoloLexicon
from recognition.lexical_budget_router import correct_rows
from scripts.icdar_budget_data import read,sha,key,annotation,predictions
from scripts.run_yolo_f1_quality_study import seal
from scripts.retrain_icdar_lexicon_router import independent


def canonical_slots(rows,lex,threshold,iou):
    policy=dict(enabled=True,confidence_cap=1.01,protect_numeric=False,unmatched_policy='source_reject')
    edited,_=correct_rows(rows,lex,policy)
    selected=[]
    def score(r):return math.sqrt(r['detector_score']*r['confidence'])
    for row in sorted(edited,key=lambda r:(-score(r),r['candidate_id'])):
        if score(row)<threshold:continue
        norm=lambda s:''.join(c for c in s.casefold() if c.isalnum())
        if any(norm(row['raw_text'])==norm(old['raw_text']) and overlap_iou(row['bbox'],old['bbox'])>iou for old in selected):continue
        selected.append(row)
    return selected


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);a=p.parse_args();c=read(a.config);root=Path(c['study_root']);old=Path(c['old_router']);plan=read(old/'resolved_configuration.json');records={r['sample_id']:r for rows in plan['records']['icdar13'].values() for r in rows};base=Path(c['generic_train_root'])
    lextrain=DeepSoloLexicon((base/'train_generic_vocabulary.txt').read_text().splitlines(),read(base/'train_generic_pairs.json'),distance_exclusive=1.5)
    test=Path(c['generic_test_root']);lextest=DeepSoloLexicon.from_files(test/'GenericVocabulary_new.txt',test/'GenericVocabulary_pair_list.txt',distance_exclusive=1.5)
    grid=[]
    for threshold in c['joint_thresholds']:
        totals={k:0 for k in ('tp','fp','fn','detected_words','gt_words','all_outputs','ignored_outputs')}
        for sid in plan['partitions']['icdar13']['selection']:
            r=records[sid];raw=read(root/'raw_ocr'/(key(sid)+'.json'));rows=canonical_slots(raw['rows'],lextrain,threshold,c['same_text_nms_iou']);gt,valid=annotation(r);pp=predictions(rows,r,valid);v=independent(pp,gt)
            for k in totals:totals[k]+=v[k]
        grid.append(dict(threshold=threshold,counts=totals,f1=2*totals['tp']/(totals['detected_words']+totals['gt_words'])))
    chosen=max(grid,key=lambda p:(p['f1'],p['threshold']));seal(root/'admission_frozen.json',dict(status='FROZEN',grid=grid,selected=chosen,test_used_for_selection=False,selection_ids=plan['partitions']['icdar13']['selection']))
    staging=root/'staging';seal(staging/'resolved_configuration.json',dict(plan,light_expert_extension='Raw low-floor YOLO640 slots plus source Generic correction and post-canonical-word NMS; admission frozen on original selection23',admission_sha256=sha(root/'admission_frozen.json')));seal(staging/'frozen_choices.json',read(old/'frozen_choices.json'))
    for sid,r in records.items():
        source=root/'raw_ocr'/(key(sid)+'.json');raw=read(source);lex=lextrain if ':train:' in sid else lextest;edited=canonical_slots(raw['rows'],lex,chosen['threshold'],c['same_text_nms_iou']);rows=[raw['rows'][p['source_row']] for p in edited]
        seal(staging/'experts/yolo'/(key(sid)+'.json'),dict(model='yolo',image_sha256=r['image_sha256'],rows=rows,ground_truth_used=False,raw_source_path=str(source),raw_source_sha256=sha(source),canonical_nms_after_edit=True))
    cfg=read(Path('configs/approximate/ic13_split_generic_budget_v1.json'));cfg.update(study_root=str(root/'router'),original_router_root=str(staging),experts=['yolo'],fixed_source_correction=True)
    cfg['pinned_inputs']=dict(cfg['pinned_inputs'],**{str(root/'admission_frozen.json'):sha(root/'admission_frozen.json'),str(root/'raw_ocr_receipt.json'):sha(root/'raw_ocr_receipt.json')});cfg['protocol']='New light expert from original low-floor YOLO boxes, all PARSeq readings and post-canonical-word NMS. Joint threshold chosen only on original selection23. Fixed source Generic editing; unchanged183/23/23 gate roles and exact61/233 budget. Prior TEST exposure remains.'
    cfg['scientific_choices']['ASSUMPTION']='Validation-selected finite raw-slot joint cut, source Generic correction before same-canonical-word NMS, TRAIN-only Generic preparation and target utility head. Changes the light expert, no TEST-selected operational policy.'
    dest=Path('configs/approximate/ic13_recovered_slot_router_v1.json');seal(dest,cfg);Path(cfg['study_root']).mkdir();seal(Path(cfg['study_root'])/'resolved_configuration.json',cfg)
    print('Selected joint cut',chosen,'TEST labels unused',flush=True)


if __name__=='__main__':main()
