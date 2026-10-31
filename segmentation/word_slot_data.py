"""Checksum-bound word/owner alignment and explicit letterbox coordinates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from recognition.scene_text_datasets import parse_totaltext_annotation
from segmentation.scene_text_preparation import clip_instances_to_pixel_extent
from segmentation.data import load_segmentation_manifest
from segmentation.totaltext_image_frequency_augment import letterbox_geometry


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n')
    temporary.replace(path)


def sample_key(sid):return sid.replace(':','__')


def original_to_canvas(boxes, image_wh):
    width,height=image_wh;side=max(width,height)
    offset=torch.tensor([(side-width)//2,(side-height)//2]*2,dtype=boxes.dtype,device=boxes.device)
    return (boxes+offset)/side


def canvas_to_original(boxes,image_wh):
    width,height=image_wh;side=max(width,height)
    offset=torch.tensor([(side-width)//2,(side-height)//2]*2,dtype=boxes.dtype,device=boxes.device)
    maximum=torch.tensor([width,height,width,height],dtype=boxes.dtype,device=boxes.device)
    return torch.minimum((boxes*side-offset).clamp_min(0),maximum)


def build_index(config, limit=None):
    paths=config['paths'];protocol=read(paths['split_protocol'])
    result={};inputs={};samples_by_phase={}
    for label in ('train','test'):
        manifest=load_segmentation_manifest(Path(paths[label+'_manifest']),require_checksums=True)
        samples_by_phase[label]={s.sample_id:s for s in manifest.samples}
    for phase in ('fit','calibration','evaluation','test'):
        label='test' if phase=='test' else 'train'
        sample_dict=samples_by_phase[label];root=Path(paths['test_root'] if label=='test' else paths['word_root'])
        ids=sorted(sample_dict) if phase=='test' else protocol[phase+'_ids']
        if limit:ids=ids[:limit]
        records=[]
        for sid in ids:
            sample=sample_dict[sid]
            paths_to_check=[(root/sample.image_path,sample.image_sha256)]
            image_path=paths_to_check[0][0]
            with Image.open(image_path) as image:
                wh=image.size;raw_orientation=image.getexif().get(274,1)
                orientation=raw_orientation or 1
            annotation=root/('datasets/total_text/'+label+'/annotations')/(sid.split(':')[1]+'.txt')
            instances,_=clip_instances_to_pixel_extent(parse_totaltext_annotation(annotation),wh)
            gt=[{'bbox':[item.bbox_xyxy[0],item.bbox_xyxy[1],item.bbox_xyxy[2]+1,item.bbox_xyxy[3]+1],
                 'text':item.transcription} for item in instances if not item.ignore]
            record={'sample_id':sid,'image_path':str(image_path),'image_sha256':sample.image_sha256,
                    'image_wh':list(wh),'orientation':orientation,'raw_exif_orientation':raw_orientation,
                    'pixel_frame':'stored PIL RGB; inverse EXIF mapping required for OpenCV DeepSolo boundaries',
                    'annotation_path':str(annotation),'annotation_sha256':sha(annotation),
                    'ground_truth':gt,'phase':phase}
            if label=='train':
                for field in ('word_owner_annotation','validity_annotation'):
                    p=root/getattr(sample,field+'_path');expected=getattr(sample,field+'_sha256')
                    paths_to_check.append((p,expected));record[field+'_path']=str(p)
            for path,expected in paths_to_check:
                actual=sha(path)
                if actual!=expected:raise ValueError('Dataset bytes changed: '+str(path))
                inputs[str(path)]=actual
            inputs[str(annotation)]=record['annotation_sha256'];records.append(record)
        result[phase]=records
    for i,a in enumerate(('fit','calibration','evaluation')):
        for b in ('fit','calibration','evaluation')[i+1:]:
            if {r['sample_id'] for r in result[a]} & {r['sample_id'] for r in result[b]}:
                raise ValueError('TRAIN data roles overlap')
    return result,inputs


def load_rgb(record):
    if sha(record['image_path'])!=record['image_sha256']:raise ValueError('RGB SHA256 changed')
    with Image.open(record['image_path']) as image:return image.convert('RGB')


def image_tensor(image):
    image=letterbox_geometry(image,fill=(0,0,0))
    image=TF.resize(image,[512,512],interpolation=InterpolationMode.BILINEAR,antialias=True)
    return TF.normalize(TF.to_tensor(image),(.485,.456,.406),(.229,.224,.225))[None]


def target_for(record,size):
    boxes=torch.tensor([g['bbox'] for g in record['ground_truth']],dtype=torch.float32).reshape(-1,4)
    target={'boxes':original_to_canvas(boxes,record['image_wh'])}
    with Image.open(record['word_owner_annotation_path']) as image:
        owners_original=np.asarray(image).copy()
        if int(owners_original.max())!=len(boxes):raise ValueError('Owner/annotation word order contract failed')
        owner=letterbox_geometry(image.convert('I'),fill=0).resize((size,size),Image.Resampling.NEAREST)
        owner=torch.from_numpy(np.asarray(owner,dtype=np.int64).copy())
    target['masks']=torch.stack([owner==i+1 for i in range(len(boxes))]).float() if len(boxes) else torch.empty(0,size,size)
    with Image.open(record['validity_annotation_path']) as image:
        validity=letterbox_geometry(image.convert('L'),fill=0).resize((size,size),Image.Resampling.NEAREST)
        target['validity']=torch.from_numpy((np.asarray(validity)>0).copy()).float()
    target['empty_mask_indices']=torch.where(target['masks'].flatten(1).sum(1)==0)[0]
    return target
