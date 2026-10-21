"""CPU-only post-hoc TRAIN V1 OCR/mask/geometry failure audit; no inference."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from rapidfuzz.fuzz import ratio
import torch

from diagnostics.gt_mask_localization_oracle import checked_native, save_json, sha, verify_file
from localize.grouped_word_blocks import GroupedWordBlockParams, group_word_blocks
from localize.practical_remap_failure_analysis import classify_failure, _box_union
from pipeline.frozen_segmentation import backproject_binary_mask
from scripts.diagnose_train_gtbox_vs_v1_parseq import _area, _inter, _iou, _wordness
from segmentation.data import load_segmentation_manifest, icdar13_task22_rgb_to_target_and_validity
from segmentation.evaluate import resize_binary_target_to_native_logits


def native_box(box, hw):
    h, w = hw; x1,y1,x2,y2 = box
    return (max(0, x1*128//w), max(0, y1*128//h),
            min(128, (x2*128+w-1)//w), min(128, (y2*128+h-1)//h))


def pixel_metrics(target, pred):
    target, pred = np.asarray(target, bool), np.asarray(pred, bool)
    if target.shape != pred.shape or target.ndim != 2 or not target.size:
        raise ValueError("aligned nonempty regions required")
    tp = int((target & pred).sum()); fp = int((~target & pred).sum()); fn = int((target & ~pred).sum())
    return {"gt_pixels": tp+fn, "pred_pixels": tp+fp, "tp": tp, "fp": fp, "fn": fn,
            "recall": tp/(tp+fn) if tp+fn else None,
            "precision": tp/(tp+fp) if tp+fp else None,
            "iou": tp/(tp+fp+fn) if tp+fp+fn else None}


def mask_bin(metrics, cfg):
    if not metrics['gt_pixels']: return "NO_NATIVE_TARGET_FOREGROUND"
    if metrics['recall'] < cfg['mask_recall_bins'][0]: return "LOW_RECALL_LT_50"
    if metrics['recall'] < cfg['mask_recall_bins'][1]: return "PARTIAL_RECALL_50_80"
    if metrics['precision'] < cfg['high_mask_precision']: return "HIGH_RECALL_LOW_PRECISION"
    return "HIGH_RECALL_HIGH_PRECISION"


def crop_failure(exact, available, multi, iou, good_iou):
    if exact: return "OCR_CORRECT"
    if not available: return "NO_OVERLAPPING_CROP"
    if multi: return "MULTIWORD_CROP"
    if iou >= good_iou: return "GOOD_BOX_OCR_ERROR"
    return "LOW_IOU_CROP"


def scalar_summary(rows):
    wrong = [r for r in rows if not r['exact']]
    return {"words": len(rows), "available": sum(r['available'] for r in rows),
        "exact": sum(r['exact'] for r in rows), "fuzzy_ge_0_8": sum((r['fuzzy'] or 0) >= .8 for r in rows),
        "failures": len(wrong), "crop_failure_counts": dict(Counter(r['crop_failure'] for r in wrong)),
        "mask_bins_failures": dict(Counter(r['mask_bin'] for r in wrong)),
        "mask_bins_correct": dict(Counter(r['mask_bin'] for r in rows if r['exact'])),
        "geometry_categories_failures": dict(Counter(r['geometry_category'] for r in wrong)),
        "wrong_but_another_existing_crop_exact_any_overlap": sum(r['any_overlap_exact'] for r in wrong),
        "wrong_but_another_existing_crop_exact_iou_ge_0_50": sum(r['any_good_iou_exact'] for r in wrong),
        "no_prediction_foreground_failures": sum(r['native_metrics']['pred_pixels']==0 for r in wrong),
        "high_quality_mask_poor_box_failures": sum(r['mask_bin']=='HIGH_RECALL_HIGH_PRECISION' and r['best_iou'] < .5 for r in wrong),
        "failure_cross_tab": {c: dict(Counter(r['mask_bin'] for r in wrong if r['crop_failure']==c)) for c in sorted({r['crop_failure'] for r in wrong})}}


def visual_panel(row, image_path, pred, gt, destination):
    with Image.open(image_path) as source: rgb = source.convert('RGB')
    gt_box = row['gt_bbox']; box = row['crop_bbox'] if row['available'] else None
    union = _box_union([gt_box] + ([box] if box else [])); x1,y1,x2,y2 = union
    region = (max(0,x1-20),max(0,y1-20),min(rgb.width,x2+20),min(rgb.height,y2+20))
    context = rgb.crop(region); d = ImageDraw.Draw(context)
    for b,color in [(gt_box,'red')] + ([(box,'lime')] if box else []):
        d.rectangle((b[0]-region[0],b[1]-region[1],b[2]-region[0]-1,b[3]-region[1]-1),outline=color,width=2)
    canvas = Image.new('RGB',(1000,570),'white'); draw = ImageDraw.Draw(canvas)
    labels = [f"sample={row['sample_id']} word={row['gt_index']} {row['crop_failure']}",
        f"GT: {row['gt_text']} | cached V1 PARSeq: {row['parseq_text']!r}",
        f"IoU={row['best_iou']:.3f} | {row['mask_bin']} | geometry={row['geometry_category']}"]
    for i,text in enumerate(labels): draw.text((8,8+18*i),text,fill='black')
    def paste(img, bounds, title):
        x,y,w,h = bounds; shown=img.copy();shown.thumbnail((w,h-22));canvas.paste(shown,(x,y+22));draw.text((x,y),title,fill='black')
    paste(context,(8,72,490,225),'RGB context: red GT, green selected crop')
    paste(rgb.crop(gt_box),(510,72,480,105),'GT box RGB (no GT_BOX OCR available)')
    if box: paste(rgb.crop(box),(510,185,480,112),'Actual selected V1 crop')
    nb=native_box(gt_box,(rgb.height,rgb.width))
    for x,array,title in [(8,gt,'GT native128'),(300,pred,'PRED native128')]:
        im=Image.fromarray((array*255).astype(np.uint8)).convert('RGB');d=ImageDraw.Draw(im)
        d.rectangle((nb[0],nb[1],nb[2]-1,nb[3]-1),outline='red',width=1)
        paste(im.resize((230,230),Image.Resampling.NEAREST),(x,315,270,250),title+' (display zoom only)')
    x1,y1,x2,y2=nb
    paste(Image.fromarray((gt[y1:y2,x1:x2]*255).astype(np.uint8)).resize((180,180),Image.Resampling.NEAREST).convert('RGB'),(600,315,190,240),'GT word region')
    paste(Image.fromarray((pred[y1:y2,x1:x2]*255).astype(np.uint8)).resize((180,180),Image.Resampling.NEAREST).convert('RGB'),(800,315,190,240),'PRED word region')
    canvas.save(destination)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','oracle','output-dir'): p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output_dir.exists(): raise FileExistsError(a.output_dir)
    cfg=json.loads(a.config.read_text())
    if cfg['reproduction_mode']!='approximate' or cfg['scope']!={'split':'TRAIN','images':229,'gt_words':849}: raise ValueError('TRAIN229/849 only')
    verify_file(a.oracle,cfg['oracle_sha256']);oracle=json.loads(a.oracle.read_text())
    for path,expected in oracle['protocol']['code_sha256'].items(): verify_file(path,expected)
    paths=oracle['protocol']['source_paths'];root=Path(paths['data_root']);source_path=Path(paths['source_report'])
    verify_file(source_path,oracle['protocol']['source_report_sha256']);source=json.loads(source_path.read_text())
    verify_file(paths['train_manifest'],oracle['protocol']['train_manifest_sha256'])
    manifest=load_segmentation_manifest(paths['train_manifest'],require_checksums=False)
    if manifest.split_role!='train' or len(manifest.samples)!=229: raise ValueError('TRAIN only')
    samples={s.sample_id:s for s in manifest.samples};refs={s['sample_id']:s for s in oracle['samples']}
    v1=GroupedWordBlockParams(**oracle['protocol']['parameters']['v1'])
    word_rows=defaultdict(list);crops=defaultdict(list)
    for w in source['per_word']: word_rows[w['sample_id']].append(w)
    for c in source['per_crop']:
        if c['method']=='PRACTICAL_REMAP_V1': crops[c['sample_id']].append(c)
    protocol={**cfg,'config_sha256':sha(a.config),'oracle_path':str(a.oracle.resolve()),'source_report':str(source_path),
        'source_report_sha256':sha(source_path),'checkpoint_sha256':oracle['protocol']['checkpoint_sha256'],
        'gtbox_per_word_status':'UNAVAILABLE_NOT_REINFERRED','device':'cpu','models_loaded':False,
        'code_sha256':{path:sha(path) for path in [__file__,'localize/practical_remap_failure_analysis.py','localize/grouped_word_blocks.py']}}
    a.output_dir.mkdir(parents=True);save_json(a.output_dir/'protocol.json',protocol)
    torch.set_num_threads(1);cv2.setNumThreads(1);rows=[]
    for index,sample in enumerate(manifest.samples,1):
        sid=sample.sample_id
        if index==1 or index%25==0 or index==229: print(f'[TRAIN OCR audit CPU] {index}/229 sample={sid}',flush=True)
        for rel in (sample.image_path,sample.mask_path,sample.validity_annotation_path): verify_file(root/rel,source['protocol']['dataset_files'][str(rel)]['sha256'])
        pred,meta=checked_native(source_path.parent/'masks'/f'{sid}.json',source['contract_sha256'],sid,
            source['protocol']['dataset_files'][str(sample.image_path)]['sha256'],protocol['checkpoint_sha256'])
        with Image.open(root/sample.mask_path) as opened:
            target,_=icdar13_task22_rgb_to_target_and_validity(opened,(root/sample.validity_annotation_path).read_text(encoding='utf-8-sig'))
        full=(np.asarray(target)>0).astype(np.uint8);gt=resize_binary_target_to_native_logits(target).numpy().astype(np.uint8)
        if hashlib.sha256(gt.tobytes()).hexdigest()!=refs[sid]['gt_native_sha256']: raise RuntimeError('GT128 changed')
        projected=backproject_binary_mask(pred,full.shape)
        if hashlib.sha256(projected.tobytes()).hexdigest()!=meta['segmentation_metadata']['output_mask_sha256']: raise RuntimeError('prediction changed')
        grouped=group_word_blocks(projected,projected,v1)
        if [list(b.padded_bbox) for b in grouped.retained]!=[c['bbox'] for c in crops[sid]]: raise RuntimeError('V1 crops changed')
        _,labels,_,_=cv2.connectedComponentsWithStats(projected,connectivity=v1.connectivity)
        raw={c.component_id:c for c in grouped.raw_components};groups=grouped.groups_before_cap
        words=[{'bbox':w['bbox'],'text':w['text']} for w in word_rows[sid]]
        raw_ids=[];component_words=defaultdict(set)
        for wi,w in enumerate(words):
            x1,y1,x2,y2=w['bbox'];ids=[int(i) for i in np.unique(labels[y1:y2,x1:x2]) if i]
            raw_ids.append(ids)
            for i in ids: component_words[i].add(wi)
        for wi,w in enumerate(word_rows[sid]):
            old=w['methods']['PRACTICAL_REMAP_V1'];box=old['bbox'];available=old['text'] is not None
            exact=available and old['text'].casefold()==w['text'].casefold()
            fuzzy=ratio(old['text'].casefold(),w['text'].casefold())/100 if available else None
            if exact!=old['exact'] or fuzzy!=old['fuzzy']: raise RuntimeError('OCR control mismatch')
            best=max(crops[sid],key=lambda c:_iou(c['bbox'],w['bbox']),default=None)
            if (best['bbox'] if best else None)!=box or available!=(best is not None and _inter(best['bbox'],w['bbox'])>0): raise RuntimeError('matching drift')
            if available and best['text']!=old['text']: raise RuntimeError('cached transcription mismatch')
            significant=sum(_inter(box,v['bbox'])/max(1,_area(v['bbox']))>=cfg['significant_gt_coverage'] for v in words) if available else 0
            nb=native_box(w['bbox'],full.shape);nx1,ny1,nx2,ny2=nb
            metrics=pixel_metrics(gt[ny1:ny2,nx1:nx2],pred[ny1:ny2,nx1:nx2])
            x1,y1,x2,y2=w['bbox'];fm=pixel_metrics(full[y1:y2,x1:x2],projected[y1:y2,x1:x2])
            ids=set(raw_ids[wi]);union=_box_union([raw[i].bbox for i in ids]);union_iou=_iou(union,w['bbox']) if union else 0
            associated=[g for g in groups if ids.intersection(g['raw_component_ids'])]
            best_group=max(groups[:v1.nmax],key=lambda g:_iou(g['padded_bbox'],w['bbox']),default=None)
            group_words=set().union(*(component_words[i] for i in best_group['raw_component_ids'])) if best_group else set()
            precap=max((_iou(g['padded_bbox'],w['bbox']) for g in groups),default=0)
            geom='SUCCESS' if old['iou']>=.5 else classify_failure(foreground_recall=fm['recall'] or 0,
                best_retained_group_iou=old['iou'],pre_cap_has_valid_group=precap>=.5,
                oracle_union_iou=union_iou,associated_group_count=len(associated),best_group_gt_word_count=len(group_words))
            alternatives=[c for c in crops[sid] if _inter(c['bbox'],w['bbox'])>0 and c['text'].casefold()==w['text'].casefold()]
            rows.append({'sample_id':sid,'gt_index':wi,'gt_text':w['text'],'gt_bbox':w['bbox'],
                'parseq_text':old['text'],'available':available,'exact':exact,'fuzzy':fuzzy,'crop_bbox':box,
                'best_iou':old['iou'],'best_precap_iou':precap,'significant_gt_words_in_crop':significant,
                'gt_coverage':_inter(box,w['bbox'])/_area(w['bbox']) if available else 0,
                'crop_purity_bbox':_inter(box,w['bbox'])/_area(box) if available else 0,
                'crop_failure':crop_failure(exact,available,significant>=2,old['iou'],cfg['good_iou']),
                'native_metrics':metrics,'mask_bin':mask_bin(metrics,cfg),'original_metrics':fm,
                'oracle_component_union_bbox':union,'oracle_component_union_iou':union_iou,
                'associated_raw_components':len(ids),'associated_v1_groups':len(associated),
                'geometry_category':geom,'any_overlap_exact':bool(alternatives),
                'any_good_iou_exact':any(_iou(c['bbox'],w['bbox'])>=.5 for c in alternatives),
                'alternative_exact_crops':alternatives,'gtbox_parseq_text':None})
    summary=scalar_summary(rows)
    if (summary['words'],summary['exact'],summary['fuzzy_ge_0_8'],sum(map(len,crops.values())))!=(849,cfg['reference_exact'],cfg['reference_fuzzy'],cfg['reference_crops']): raise RuntimeError('frozen aggregate mismatch')
    geometry_counts=dict(Counter(r['geometry_category'] for r in rows));summary['geometry_counts_all_words']=geometry_counts
    (a.output_dir/'examples').mkdir();examples=[]
    for category in sorted(summary['crop_failure_counts']):
        selected=[];seen=set()
        for row in sorted(rows,key=lambda r:(int(r['sample_id']),r['gt_index'])):
            if row['crop_failure']==category and row['sample_id'] not in seen:
                selected.append(row);seen.add(row['sample_id'])
                if len(selected)==cfg['example_count_per_failure_category']:break
        for row in selected:
            sid=row['sample_id'];sample=samples[sid]
            pred,_=checked_native(source_path.parent/'masks'/f'{sid}.json',source['contract_sha256'],sid,source['protocol']['dataset_files'][str(sample.image_path)]['sha256'],protocol['checkpoint_sha256'])
            with Image.open(root/sample.mask_path) as opened:
                target,_=icdar13_task22_rgb_to_target_and_validity(opened,(root/sample.validity_annotation_path).read_text(encoding='utf-8-sig'))
            gt=resize_binary_target_to_native_logits(target).numpy().astype(np.uint8)
            out=a.output_dir/'examples'/f'{sid}-{row["gt_index"]}-{category}.png'
            visual_panel(row,root/sample.image_path,pred,gt,out)
            examples.append({'sample_id':sid,'gt_index':row['gt_index'],'category':category,'path':str(out),'sha256':sha(out)})
    with (a.output_dir/'per-word.csv').open('w',newline='') as stream:
        fields=['sample_id','gt_index','gt_text','parseq_text','exact','fuzzy','best_iou','gt_coverage','crop_failure','mask_bin','geometry_category','associated_raw_components','associated_v1_groups','oracle_component_union_iou','any_overlap_exact','any_good_iou_exact']
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    result={'protocol':protocol,'summary':summary,'per_word':rows,'examples':examples,'csv_sha256':sha(a.output_dir/'per-word.csv'),
        'validation':{'images':229,'gt_words':849,'exact_and_fuzzy_match_frozen':True,'all_v1_crop_boxes_identical':True,'native_input_hashes_verified':True}}
    save_json(a.output_dir/'results.json',result)
    print(json.dumps(summary,indent=2));print('JSON',a.output_dir/'results.json','SHA256',sha(a.output_dir/'results.json'))


if __name__=='__main__':main()
