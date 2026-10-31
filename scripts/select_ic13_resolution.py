"""Freeze selection-only resolution/admission, then stage the stronger light expert."""

import argparse
from pathlib import Path

from recognition.deepsolo_lexicon import DeepSoloLexicon
from scripts.icdar_budget_data import read,sha,key,annotation,predictions
from scripts.retrain_icdar_lexicon_router import independent
from scripts.prepare_ic13_slot_budget import canonical_slots
from scripts.run_yolo_f1_quality_study import seal


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--stage',choices=['freeze','staging'],required=True);a=p.parse_args();c=read(a.config);root=Path(c['study_root']);old=Path(c['old_router']);plan=read(old/'resolved_configuration.json');records={r['sample_id']:r for rows in plan['records']['icdar13'].values() for r in rows};base=Path(c['generic_train_root']);trainlex=DeepSoloLexicon((base/'train_generic_vocabulary.txt').read_text().splitlines(),read(base/'train_generic_pairs.json'),distance_exclusive=1.5)
    if a.stage=='freeze':
        grid=[];pins={};ids=plan['partitions']['icdar13']['selection']
        for size in c['sizes']:
            for threshold in c['joint_thresholds']:
                totals={k:0 for k in ('tp','fp','fn','detected_words','gt_words','all_outputs','ignored_outputs')}
                for sid in ids:
                    path=root/str(size)/'ocr'/(key(sid)+'.json');raw=read(path);pins[str(path)]=sha(path);assert raw['image_sha256']==records[sid]['image_sha256']
                    rows=canonical_slots(raw['rows'],trainlex,threshold,c['canonical_nms_iou']);gt,valid=annotation(records[sid]);value=independent(predictions(rows,records[sid],valid),gt)
                    for k in totals:totals[k]+=value[k]
                grid.append(dict(size=size,threshold=threshold,counts=totals,f1=2*totals['tp']/(totals['detected_words']+totals['gt_words'])))
        winner=max(grid,key=lambda p:(p['f1'],-p['size'],p['threshold']));seal(root/'resolution_frozen.json',dict(status='FROZEN',selected=winner,grid=grid,selection_ids=ids,test_used_for_selection=False,input_sha256=pins,config_sha256=sha(a.config),code_sha256=sha(__file__)));print('Resolution chosen on selection23',winner,flush=True);return
    frozen=read(root/'resolution_frozen.json');winner=frozen['selected'];size=winner['size'];test=Path(c['generic_test_root']);testlex=DeepSoloLexicon.from_files(test/'GenericVocabulary_new.txt',test/'GenericVocabulary_pair_list.txt',distance_exclusive=1.5)
    staging=root/'staging';seal(staging/'resolved_configuration.json',dict(plan,light_expert_extension='Selection-only input-size and joint-admission adaptation with source Generic and canonical NMS',resolution_frozen_sha256=sha(root/'resolution_frozen.json')));seal(staging/'frozen_choices.json',read(old/'frozen_choices.json'))
    for sid,r in records.items():
        path=root/str(size)/'ocr'/(key(sid)+'.json');raw=read(path);lex=trainlex if ':train:' in sid else testlex;edited=canonical_slots(raw['rows'],lex,winner['threshold'],c['canonical_nms_iou']);rows=[raw['rows'][p['source_row']] for p in edited]
        seal(staging/'experts/yolo'/(key(sid)+'.json'),dict(model='yolo',image_sha256=r['image_sha256'],rows=rows,ground_truth_used=False,source_ocr_path=str(path),source_ocr_sha256=sha(path),input_size=size))
    config=read(Path('configs/approximate/ic13_recovered_slot_router_v1.json'));config.update(study_root=str(root/'router'),original_router_root=str(staging));config['pinned_inputs']={p:h for p,h in config['pinned_inputs'].items() if '/ic13_raw_slot_recovery/' not in p};config['pinned_inputs'][str(root/'resolution_frozen.json')]=sha(root/'resolution_frozen.json');config['protocol']='YOLO resolution and joint confidence selected on original23 selection images only; source Generic and canonical NMS fixed. Fit183, select23, CAL23 unchanged; exact61/233 budget, no TEST-based operational choice.'
    config['scientific_choices']['ASSUMPTION']='Selection-only finite YOLO input-resolution and admission adaptation, TRAIN-only Generic, canonical NMS and target utility head. Explicitly changes light expert; historical TEST exposure remains.'
    seal(Path('configs/approximate/ic13_resolution_router_v1.json'),config);Path(config['study_root']).mkdir();seal(Path(config['study_root'])/'resolved_configuration.json',config);print('Stronger light expert staged',len(records),'images; frozen resolution',size,flush=True)


if __name__=='__main__':main()
