"""Explicit image populations and annotation-only metric boundaries."""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image,ImageDraw
import yaml

from recognition.icdar13_annotations import load_icdar13_recognition_samples
from recognition.scene_text_datasets import parse_icdar2015_annotation
from segmentation.separated_word_gt import parse_icdar13_character_groups
from segmentation.scene_text_preparation import clip_instances_to_pixel_extent
from scripts.crop_clip_runtime import array_sha
from recognition.clip_router_evaluation import native_mask

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def key(s):return s.replace(':','__')


def image_record(path,sid,annotation,split):
    path=Path(path)
    with Image.open(path) as im:
        rgb=np.asarray(im.convert('RGB'));wh=list(im.size);orientation=im.getexif().get(274,1) or 1
    return dict(sample_id=sid,image_path=str(path),image_sha256=sha(path),rgb_sha256=array_sha(rgb),image_wh=wh,
                orientation=orientation,annotation_path=str(annotation),annotation_sha256=sha(annotation),split=split)


def prepare_population(paths,cfg):
    records={};partitions={};metadata={}
    ic13=Path(paths['icdar13_root']);ic15=Path(paths['segmentation_root'])
    raw=yaml.safe_load(Path(paths['icdar13_train_manifest']).read_text())['samples'];train=[]
    # The train and validation segmentation manifests form the complete229 population.
    raw+=yaml.safe_load(Path(paths['icdar13_validation_manifest']).read_text())['samples']
    for row in raw:
        sid=str(row['id']);record=image_record(ic13/row['image'],'icdar13:train:'+sid,ic13/'raw/train_segmentation_gt'/(sid+'_GT.txt'),'train')
        if record['image_sha256']!=row['image_sha256']:raise ValueError('ICDAR13 training manifest RGB changed')
        train.append(record)
    samples,_=load_icdar13_recognition_samples(Path(paths['icdar13_test_manifest']),ic13)
    test=[image_record(s.image_path,'icdar13:'+s.sample_id,s.annotation_path,'test') for s in samples]
    records['icdar13']=dict(train=sorted(train,key=lambda r:r['sample_id']),test=sorted(test,key=lambda r:r['sample_id']))
    for split in ('train','test'):
        manifest=yaml.safe_load(Path(paths['icdar2015_'+split+'_manifest']).read_text())['samples'];values=[]
        for row in manifest:
            number=row['id'].split(':')[-1];annotation=ic15/'datasets/icdar2015'/split/'annotations'/paths['icdar2015_annotation_template'].format(number=number)
            record=image_record(ic15/row['image'],'icdar2015:'+('train:' if split=='train' else '')+number,annotation,split)
            if record['image_sha256']!=row['image_sha256']:raise ValueError('ICDAR2015 prepared raster changed')
            values.append(record)
        records.setdefault('icdar2015',{})[split]=sorted(values,key=lambda r:r['sample_id'])
    native=read(paths['icdar2015_selection']);native_val={s.split(':')[-1] for s in native['validation_ids']}
    for ds,data in records.items():
        test_rgb={r['rgb_sha256'] for r in data['test']};excluded=[r['sample_id'] for r in data['train'] if r['rgb_sha256'] in test_rgb]
        data['train']=[r for r in data['train'] if r['sample_id'] not in excluded]
        ids=[r['sample_id'] for r in data['train']]
        if ds=='icdar13':
            # Both eligible TotalText expert variants have never fitted IC13 images.
            grouped={}
            for r in data['train']:grouped.setdefault(r['rgb_sha256'],[]).append(r['sample_id'])
            order=sorted(grouped,key=lambda g:hashlib.sha256((str(cfg['seed'])+':'+g).encode()).digest())
            held=[]
            for g in order:
                if len(held)>=cfg['icdar13_validation_images']:break
                held+=grouped[g]
        else:held=[s for s in ids if s.split(':')[-1] in native_val]
        fit=sorted(set(ids)-set(held));ordered=sorted(held,key=lambda s:hashlib.sha256((str(cfg['seed'])+'|'+s).encode()).digest())
        n=len(ordered)//2;select=sorted(ordered[:n]);cal=sorted(ordered[n:])
        groups=[{r['rgb_sha256'] for r in data['train'] if r['sample_id'] in part} for part in (fit,select,cal)]+[test_rgb]
        if any(groups[a]&groups[b] for a in range(4) for b in range(a+1,4)):raise ValueError('RGB overlap across router populations')
        partitions[ds]=dict(fit=fit,selection=select,calibration=cal,test=[r['sample_id'] for r in data['test']])
        metadata[ds]=dict(excluded_train_test_RGB_copies=excluded,original_training_images=len(raw) if ds=='icdar13' else 1000,
            test_images=len(data['test']),expert_fitting_exposure='Both IC13 candidates are TotalText-only experts' if ds=='icdar13' else 'Native IC15 FIT900 seen by expert; held-out100 unseen by expert gradients; historically used for upstream selection')
    return records,partitions,metadata


def annotation(record):
    p=Path(record['annotation_path']);wh=tuple(record['image_wh'])
    if sha(p)!=record['annotation_sha256']:raise ValueError('Annotation bytes changed')
    if record['sample_id'].startswith('icdar13:'):
        if record['split']=='train':instances=parse_icdar13_character_groups(p)
        else:
            from recognition.icdar13_annotations import parse_task1_annotation
            words=parse_task1_annotation(p,image_size_wh=wh)
            return [dict(text=w.transcription,bbox=[w.bbox_xyxy[0],w.bbox_xyxy[1],w.bbox_xyxy[2]+1,w.bbox_xyxy[3]+1]) for w in words],native_mask(Image.new('L',wh,255),256)
    else:instances=parse_icdar2015_annotation(p)
    instances,_=clip_instances_to_pixel_extent(instances,wh)
    validity=Image.new('L',wh,255);draw=ImageDraw.Draw(validity)
    for ignored,fill in [(True,0),(False,255)]:
        for word in instances:
            if word.ignore==ignored:draw.polygon(list(word.polygon_xy),fill=fill)
    words=[dict(text=w.transcription,bbox=[w.bbox_xyxy[0],w.bbox_xyxy[1],w.bbox_xyxy[2]+1,w.bbox_xyxy[3]+1]) for w in instances if not w.ignore]
    return words,native_mask(validity,256)


def predictions(rows,record,validity):
    result=[]
    for r in rows:
        a,b,c,d=r['bbox'];mask=Image.new('L',tuple(record['image_wh']),0);ImageDraw.Draw(mask).rectangle([a,b,c-1,d-1],fill=255)
        footprint=native_mask(mask,256)
        result.append(dict(text=r['raw_text'],bbox=r['bbox'],ignored=bool(footprint.any() and not np.any(footprint&validity))))
    return result
