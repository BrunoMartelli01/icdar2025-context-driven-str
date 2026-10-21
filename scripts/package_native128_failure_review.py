"""Render a few measured diagnostic figures, not an exported OCR crop dataset."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rapidfuzz.distance import Levenshtein

from scripts.analyze_native128_failures import augment, exact_edges, box_matrices, write
from segmentation.data import file_sha256, load_segmentation_manifest
from segmentation.tighter_word_gt import native_training_owners


def font(size):
    return ImageFont.truetype('DejaVuSans.ttf', size)


def correct(row):
    return row['category'] in ['correct_top10','correct_only_beyond_cap']


def run(directory):
    protocol=json.loads((directory/'full/protocol.json').read_text())
    c=protocol['resolved_config'];source=Path(c['source_directory'])/'full'
    rows=json.loads((directory/'full/test_words.json').read_text())
    targets=json.loads((directory/'full/train_words.json').read_text())
    gt=json.loads((source/'ground_truth.json').read_text())['ground_truth']
    ocr={r['crop_array_sha256']:r for line in (source/'ocr.jsonl').read_text().splitlines() if (r:=json.loads(line))}
    details={}
    for model in c['models']:
        a=[r for r in rows if r['model']==model];totals=Counter()
        for path in sorted((source/model/'proposals').glob('*.json')):
            shard=json.loads(path.read_text());ps=shard['variants']['1']['proposals'][:10];words=gt[shard['sample_id']]
            totals['top10_padded_geometry']+=len(augment(box_matrices(ps,words,'padded_bbox_original')[0]>=c['iou_threshold']))
            totals['top10_padded_exact']+=len(augment(exact_edges(ps,words,ocr,'padded_bbox_original',c['iou_threshold'])))
        details[model]={'unlimited_padded_metric_gained_gt':sum(r['correct_with_padded_box_metric'] and not correct(r) for r in a),
            'unlimited_padded_metric_lost_gt':sum(not r['correct_with_padded_box_metric'] and correct(r) for r in a),**totals}
    errors=[r for r in rows if r['model']=='compact_361' and r['category']=='localized_but_ocr_or_crop_error']
    alnum=lambda s:''.join(x for x in s.casefold() if x.isalnum())
    details['compact_localized_ocr_errors']={'count':len(errors),
        'empty_best_iou_text':sum(not r['best_ocr_text'].strip() for r in errors),
        'best_iou_crop_one_edit_away':sum(Levenshtein.distance(r['gt_text'].casefold(),r['best_ocr_text'].casefold())==1 for r in errors),
        'nonalphanumeric_only_best_iou_text':sum(alnum(r['gt_text'])==alnum(r['best_ocr_text']) for r in errors),
        'note':'Diagnostic best-IoU crop only; official GT and primary text normalization unchanged'}
    sets={m:{(r['sample_id'],r['gt_index']) for r in rows if r['model']==m and correct(r)} for m in ['sep_447','compact_361']}
    details['unlimited_v2_vs_v1_exact_gt_sets']={'common':len(sets['compact_361']&sets['sep_447']),
        'gained':len(sets['compact_361']-sets['sep_447']),'lost':len(sets['sep_447']-sets['compact_361']),
        'note':'Membership under documented matching; comparison is not causal GT-only evidence'}
    write(directory/'full/failure_details.json',details)

    out=directory/'figures';out.mkdir(exist_ok=True)
    cfg=json.loads((Path(c['review_directory'])/'source/configs/approximate/native128_gt_review_v1.json').read_text())
    versions={v['id']:v for v in cfg['gt_versions']}
    manifests={f:load_segmentation_manifest(Path(versions[f]['manifest']),require_checksums=True) for f in ['separated','compact']}
    samples={f:{s.sample_id:s for s in m.samples} for f,m in manifests.items()}
    stats={(r['family'],r['size'],r['sample_id'],r['owner']):r for r in targets}
    conditions=[('Resize breaks the compact core',lambda a,b,c:a['components']==1 and b['components']>=3 and c['components']==1),
        ('Fragmentation persists at 256',lambda a,b,c:a['components']==1 and b['components']>=3 and c['components']>=3),
        ('Word vanishes at 128',lambda a,b,c:b['components']==0 and c['components']==1)]
    selected=[];seen=set()
    for label,condition in conditions:
        candidates=[]
        for r in targets:
            if r['family']!='compact' or r['size']!=128 or r['sample_id'] in seen:
                continue
            sid,owner=r['sample_id'],r['owner']
            a=stats['separated',128,sid,owner];b=stats['compact',128,sid,owner];cc=stats['compact',256,sid,owner]
            if condition(a,b,cc):candidates.append((sid,owner,a,b,cc))
        sid,owner,a,b,cc=sorted(candidates,key=lambda x:(x[0],x[1]))[0]
        selected.append((label,sid,owner,a,b,cc));seen.add(sid)
    canvas=Image.new('RGB',(1280,930),'white');draw=ImageDraw.Draw(canvas)
    draw.text((16,10),'TRAIN GT audit | measured native targets | approximate reproduction',font=font(23),fill='black')
    draw.text((16,43),'Cyan: selected word; grey: other words. 256 = target-only counterfactual, no new model.',font=font(17),fill='black')
    provenance=[]
    for row,(label,sid,owner,a,b,cc) in enumerate(selected):
        y=82+row*280;native={};sources=[]
        for family in ['separated','compact']:
            s=samples[family][sid];arrays=[];root=Path(versions[family]['data_root'])
            for rel,digest,mode in [(s.word_owner_annotation_path,s.word_owner_annotation_sha256,'I'),
                (s.validity_annotation_path,s.validity_annotation_sha256,'L'),(s.separation_annotation_path,s.separation_annotation_sha256,'L')]:
                path=root/rel
                if file_sha256(path)!=digest:raise ValueError('Figure source changed')
                with Image.open(path) as im:arrays.append(np.asarray(im.convert(mode)).copy())
                sources.append({'path':str(path),'sha256':digest})
            native[family,128]=native_training_owners(*arrays,(128,128))
            if family=='compact':native[family,256]=native_training_owners(*arrays,(256,256))
        s=samples['compact'][sid];root=Path(versions['compact']['data_root'])
        annotation_path=(root/s.word_owner_annotation_path).parent.parent/'annotations'/f'{Path(s.word_owner_annotation_path).stem}.json'
        annotation=json.loads(annotation_path.read_text())
        assert annotation['owners_sha256']==s.word_owner_annotation_sha256
        instance=next(x for x in annotation['instances'] if x['owner_id']==owner)
        xy=np.asarray(instance['polygon_xy']);h,w=annotation['size_wh'][1],annotation['size_wh'][0]
        roi=[max(0,int(np.floor(xy[:,0].min()*128/w))-3),max(0,int(np.floor(xy[:,1].min()*128/h))-3),
            min(128,int(np.ceil(xy[:,0].max()*128/w))+4),min(128,int(np.ceil(xy[:,1].max()*128/h))+4)]
        text=instance['transcription'];draw.text((16,y),f'{label} | {sid}, owner {owner}, GT: {text}',font=font(18),fill='black')
        image_path=root/s.image_path
        assert file_sha256(image_path)==s.image_sha256
        with Image.open(image_path) as im:im=im.convert('RGB')
        im.thumbnail((294,214),Image.Resampling.LANCZOS);left=16+(294-im.width)//2;top=y+42+(214-im.height)//2
        canvas.paste(im,(left,top));bounds=[left+xy[:,0].min()*im.width/w,top+xy[:,1].min()*im.height/h,
            left+xy[:,0].max()*im.width/w,top+xy[:,1].max()*im.height/h]
        draw.rectangle(bounds,outline=(240,40,40),width=3)
        for col,(family,size,values) in enumerate([('separated',128,a),('compact',128,b),('compact',256,cc)],1):
            x=16+col*316;draw.text((x,y+28),f'{family} {size}: {values["components"]} components',font=font(16),fill='black')
            mask=native[family,size];scale=size//128;x0,y0,x1,y1=[v*scale for v in roi]
            patch=mask[y0:y1,x0:x1];rgb=np.zeros((*patch.shape,3),np.uint8);rgb[patch>0]=(75,75,75);rgb[patch==owner]=(0,210,230)
            panel=Image.fromarray(rgb);ratio=min(294/panel.width,204/panel.height)
            panel=panel.resize((max(1,round(panel.width*ratio)),max(1,round(panel.height*ratio))),Image.Resampling.NEAREST)
            canvas.paste(panel,(x+(294-panel.width)//2,y+54+(204-panel.height)//2))
        provenance.append({'scope':'TRAIN visual diagnostic, selected failure examples not representative sample','label':label,
            'sample_id':sid,'owner':owner,'transcription':text,'native_roi':roi,'counts':[a['components'],b['components'],cc['components']],
            'image_sha256':s.image_sha256,'annotation_sha256':file_sha256(annotation_path),'sources':sources})
    canvas.save(out/'gt_connectivity_examples.png')

    test_manifest=load_segmentation_manifest(Path(cfg['test_manifest']),require_checksums=True)
    test_samples={s.sample_id:s for s in test_manifest.samples}
    model_rows=[r for r in rows if r['model']=='compact_361']
    conditions=[('Small core: correct OCR, IoU below 0.5',lambda r:r['category']=='not_localized_iou50' and r['correct_with_padded_box_metric']
        and r['best_ocr_text'].casefold()==r['gt_text'].casefold() and .2<r['best_iou']<.5),
        ('Several candidate boxes inside one word (proxy)',lambda r:r['fragment_bbox_proxy'] and len(r['gt_text'])>=5 and r['native_bbox_short_side']>4),
        ('Difference only in punctuation / spacing',lambda r:r['category']=='localized_but_ocr_or_crop_error'
            and alnum(r['gt_text'])==alnum(r['best_ocr_text']) and len(r['gt_text'])>=4),
        ('Correct word excluded by top10',lambda r:r['category']=='correct_only_beyond_cap'
            and r['best_ocr_text'].casefold()==r['gt_text'].casefold() and r['best_proposal']['rank']>10)]
    canvas=Image.new('RGB',(1200,960),'white');draw=ImageDraw.Draw(canvas)
    draw.text((16,12),'TEST examples | V2/361, native padding1 | approximate reproduction',font=font(22),fill='black')
    draw.text((16,43),'Red: annotated box. Blue: predicted component box. Green: actual padded crop.',font=font(17),fill='black')
    seen=set()
    for k,(label,condition) in enumerate(conditions):
        candidates=sorted((r for r in model_rows if condition(r) and r['sample_id'] not in seen),key=lambda r:(r['sample_id'],r['gt_index']))
        r=candidates[0];seen.add(r['sample_id']);s=test_samples[r['sample_id']];path=Path(cfg['test_data_root'])/s.image_path
        assert file_sha256(path)==s.image_sha256
        with Image.open(path) as im:im=im.convert('RGB')
        w,h=im.size;im.thumbnail((566,310),Image.Resampling.LANCZOS)
        x=16+(k%2)*596;y=90+(k//2)*424
        draw.text((x,y),label,font=font(16),fill='black')
        draw.text((x,y+24),f"{r['sample_id']}  GT={r['gt_text']!r}",font=font(17),fill='black')
        draw.text((x,y+48),f"OCR={r['best_ocr_text']!r} | IoU={r['best_iou']:.3f} | rank={r['best_proposal']['rank']}",font=font(15),fill='black')
        left=x+(566-im.width)//2;top=y+78+(310-im.height)//2;canvas.paste(im,(left,top))
        for box,color in [(r['gt_bbox'],(235,40,40)),(r['best_proposal']['bbox'],(20,80,245)),
            (r['best_proposal']['padded_bbox_original'],(20,180,40))]:
            mapped=[left+box[0]*im.width/w,top+box[1]*im.height/h,left+box[2]*im.width/w,top+box[3]*im.height/h]
            draw.rectangle(mapped,outline=color,width=2)
        if r['fragment_bbox_proxy']:
            shard=json.loads((source/'compact_361/proposals'/f"{r['sample_id'].replace(':','__')}.json").read_text())
            proposals=shard['variants']['1']['proposals'];_,cov_gt,cov_pred=box_matrices(proposals,[{'bbox':r['gt_bbox']}])
            for index in np.flatnonzero((cov_gt[:,0]>=c['fragment_min_word_fraction'])&(cov_pred[:,0]>=c['proxy_coverage'])):
                box=proposals[index]['bbox'];draw.rectangle([left+box[0]*im.width/w,top+box[1]*im.height/h,
                    left+box[2]*im.width/w,top+box[3]*im.height/h],outline=(20,80,245),width=2)
        provenance.append({'scope':'TEST illustrative frozen-bbox diagnostic','label':label,'word':r,'image_sha256':s.image_sha256})
    canvas.save(out/'prediction_failure_examples.png')
    write(out/'provenance.json',{'claim_label':c['claim_label'],'selection':'first lexicographic qualifying distinct-image example per named failure class; diagnostic extremes, not representative sampling',
        'package_code_sha256':file_sha256(Path(__file__)),'examples':provenance})
    print(json.dumps({'status':'SUCCESS','figures':2,'saved_individual_crops_or_weights':0,'details':details}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    run(p.parse_args().directory)
