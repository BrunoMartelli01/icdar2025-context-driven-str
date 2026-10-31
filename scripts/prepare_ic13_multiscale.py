"""Validation-only canonical-word fusion of three frozen YOLO/PARSeq resolutions."""

import argparse
import math
from pathlib import Path

from localize.independent_word_slots import overlap_iou
from recognition.deepsolo_lexicon import DeepSoloLexicon
from recognition.lexical_budget_router import correct_rows
from scripts.icdar_budget_data import read,sha,key,annotation,predictions
from scripts.retrain_icdar_lexicon_router import independent
from scripts.run_yolo_f1_quality_study import seal


def fused(rows,lex,threshold,minimum_support,same_iou,conflict_iou):
    edited,_=correct_rows(rows,lex,dict(enabled=True,confidence_cap=1.01,protect_numeric=False,unmatched_policy='source_reject'))
    score=lambda r:math.sqrt(r['detector_score']*r['confidence'])
    norm=lambda s:''.join(c for c in s.casefold() if c.isalnum())
    groups=[]
    for row in sorted(edited,key=lambda r:(-score(r),r['candidate_id'])):
        if score(row)<threshold:continue
        found=next((g for g in groups if norm(g[0]['raw_text'])==norm(row['raw_text']) and overlap_iou(g[0]['bbox'],row['bbox'])>same_iou),None)
        if found is None:groups.append([row])
        else:found.append(row)
    representatives=[]
    for group in groups:
        support=sorted({r['scale'] for r in group})
        if len(support)<minimum_support:continue
        rep=group[0];conflicts=sum(norm(g[0]['raw_text'])!=norm(rep['raw_text']) and overlap_iou(g[0]['bbox'],rep['bbox'])>conflict_iou for g in groups)
        rep=dict(rep,scale_support=len(support),scale_joints=[max(score(r) for r in group if r['scale']==s) for s in support],scale_text_conflicts=conflicts)
        if any(overlap_iou(rep['bbox'],r['bbox'])>conflict_iou for r in representatives):continue
        representatives.append(rep)
    return representatives


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);a=p.parse_args();c=read(a.config);root=Path(c['study_root']);old=Path(c['old_router']);plan=read(old/'resolved_configuration.json');records={r['sample_id']:r for rows in plan['records']['icdar13'].values() for r in rows};base=Path(c['generic_train_root']);trainlex=DeepSoloLexicon((base/'train_generic_vocabulary.txt').read_text().splitlines(),read(base/'train_generic_pairs.json'),distance_exclusive=1.5)
    def raw(sid):
        result=[]
        for size in c['sizes']:
            source=root/str(size)/'ocr'/(key(sid)+'.json');d=read(source);assert d['image_sha256']==records[sid]['image_sha256'];result.extend(dict(r,scale=size) for r in d['rows'])
        return result
    grid=[]
    for threshold in c['fusion_joint_thresholds']:
        for support in c['fusion_minimum_supports']:
            total={k:0 for k in ('tp','fp','fn','detected_words','gt_words','all_outputs','ignored_outputs')}
            for sid in plan['partitions']['icdar13']['selection']:
                rows=fused(raw(sid),trainlex,threshold,support,c['canonical_nms_iou'],c['conflict_nms_iou']);gt,valid=annotation(records[sid]);v=independent(predictions(rows,records[sid],valid),gt)
                for k in total:total[k]+=v[k]
            grid.append(dict(threshold=threshold,minimum_support=support,counts=total,f1=2*total['tp']/(total['detected_words']+total['gt_words'])))
    winner=max(grid,key=lambda x:(x['f1'],x['minimum_support'],x['threshold']));seal(root/'fusion_frozen.json',dict(status='FROZEN',selected=winner,grid=grid,selection_ids=plan['partitions']['icdar13']['selection'],test_used_for_selection=False,config_sha256=sha(a.config)))
    test=Path(c['generic_test_root']);testlex=DeepSoloLexicon.from_files(test/'GenericVocabulary_new.txt',test/'GenericVocabulary_pair_list.txt',distance_exclusive=1.5);staging=root/'fusion_staging'
    seal(staging/'resolved_configuration.json',dict(plan,light_expert_extension='Three-resolution canonical text fusion and observable scale agreement',fusion_frozen_sha256=sha(root/'fusion_frozen.json')));seal(staging/'frozen_choices.json',read(old/'frozen_choices.json'))
    for sid,r in records.items():
        source=raw(sid);rows=fused(source,trainlex if ':train:' in sid else testlex,winner['threshold'],winner['minimum_support'],c['canonical_nms_iou'],c['conflict_nms_iou'])
        # Retain raw native readings/tokens for confidence features; the fixed
        # downstream Generic correction reproduces the canonical strings.
        rows=[dict(source[row['source_row']],scale_support=row['scale_support'],scale_joints=row['scale_joints'],scale_text_conflicts=row['scale_text_conflicts']) for row in rows]
        seal(staging/'experts/yolo'/(key(sid)+'.json'),dict(model='yolo',image_sha256=r['image_sha256'],rows=rows,ground_truth_used=False,sizes=c['sizes']))
    cfg=read(Path('configs/approximate/ic13_recovered_slot_router_v1.json'));cfg.update(study_root=str(root/'fusion_router'),original_router_root=str(staging),consensus_features=True);cfg['group_sets']=[['raw_ocr','lexical','consensus'],['raw_ocr','tokens','lexical','consensus'],['raw_ocr','post_ocr','lexical','consensus'],['raw_ocr','tokens','post_ocr','lexical','consensus']];cfg['pinned_inputs']={p:h for p,h in cfg['pinned_inputs'].items() if '/ic13_raw_slot_recovery/' not in p};cfg['pinned_inputs'][str(root/'fusion_frozen.json')]=sha(root/'fusion_frozen.json');cfg['protocol']='Three-resolution light fusion selected on original selection23; FIT183 and CAL23 remain separate; consensus is prediction-only. Fixed source Generic and exact61/233 DeepSolo; three light detector/reader passes, no new heavy encoder or fallback outputs.'
    cfg['scientific_choices']['ASSUMPTION']='Multi-resolution canonical-word geometry fusion and scale-support descriptors; selection-only threshold/support, TRAIN-only Generic and target utility head.'
    seal(Path('configs/approximate/ic13_multiscale_router_v1.json'),cfg);Path(cfg['study_root']).mkdir();seal(Path(cfg['study_root'])/'resolved_configuration.json',cfg);print('Multi-scale fusion chosen on selection only',winner,flush=True)


if __name__=='__main__':main()
