"""Audited synthetic target cache; original labels and RGB bytes are preserved."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from segmentation.data import file_sha256
from segmentation.synthtext_words import synthtext_word_targets
from segmentation.totaltext_word_kernel import native_owner_map


def prepare_sample(row,image_root,cache_root,target_policy,excluded_hashes):
    image_root,cache_root=Path(image_root),Path(cache_root)
    relative=Path(row['image_relative'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe synthetic RGB path')
    path=image_root/relative
    if file_sha256(path)!=row['image_sha256']:
        raise ValueError('Synthetic RGB bytes changed: '+row['sample_id'])
    if row['image_sha256'] in excluded_hashes:
        raise ValueError('Synthetic RGB overlaps an excluded exposure population')
    index=int(row['sample_id'].split(':')[1]);directory=cache_root/'targets'/str(index//1000)
    target=directory/(str(index)+'.npz');receipt=target.with_suffix('.json')
    words_sha=sha256(json.dumps(row['words'],sort_keys=True).encode()).hexdigest()
    if receipt.exists():
        audit=json.loads(receipt.read_text())
        if (audit['words_sha256']!=words_sha or audit['image_sha256']!=row['image_sha256']
                or audit['target_policy']!=target_policy or file_sha256(target)!=audit['target_sha256']):
            raise ValueError('Published synthetic target cache changed')
    else:
        with Image.open(path) as image:
            size=image.size;image.verify()
        mask,valid,separator,owners,audit=synthtext_word_targets(size,row['words'],policy=target_policy,sample_id=row['sample_id'])
        owner=np.asarray(owners)
        if owner.min()<0 or owner.max()>65535:
            raise ValueError('Owner IDs do not fit the explicitly declared uint16 cache')
        directory.mkdir(parents=True,exist_ok=True);temporary=target.with_suffix('.npz.tmp')
        with temporary.open('wb') as f:
            np.savez_compressed(f,mask=np.asarray(mask,dtype=np.uint8),validity=np.asarray(valid,dtype=np.uint8),
                separation=np.asarray(separator,dtype=np.uint8),owners=owner.astype(np.uint16))
        temporary.replace(target)
        audit=dict(target_audit=audit,words_sha256=words_sha,image_sha256=row['image_sha256'],
            image_wh=list(size),target_policy=target_policy,target_sha256=file_sha256(target))
        temporary=receipt.with_suffix('.json.tmp');temporary.write_text(json.dumps(audit,sort_keys=True)+'\n');temporary.replace(receipt)
    return dict(row,target_file=str(target.relative_to(cache_root)),target_sha256=audit['target_sha256'],
        image_wh=audit['image_wh'],target_audit=audit['target_audit'])


class CachedSynthTextBCE(Dataset):
    def __init__(self,manifest,*,image_root,cache_root,transform):
        if manifest['target_policy']!='preserve_visible_words_v1':
            raise ValueError('Explicit qualified synthetic target policy required')
        self.manifest=manifest;self.image_root=Path(image_root);self.cache_root=Path(cache_root);self.transform=transform

    def __len__(self):return len(self.manifest['samples'])

    def __getitem__(self,index):
        row=self.manifest['samples'][index];path=self.image_root/row['image_relative'];target=self.cache_root/row['target_file']
        if file_sha256(path)!=row['image_sha256'] or file_sha256(target)!=row['target_sha256']:
            raise ValueError('Synthetic RGB or cached target checksum changed')
        with Image.open(path) as opened:image=opened.convert('RGB')
        with np.load(target,allow_pickle=False) as a:
            mask=Image.fromarray(a['mask']);valid=Image.fromarray(a['validity']);separator=Image.fromarray(a['separation'])
            owners=Image.fromarray(a['owners'].astype(np.int32))
        x,y,v,sep=self.transform.transform_with_validity_and_separation(image,mask,valid,separator,owners,index)
        native=native_owner_map(owners,self.transform,index,y,v)
        visible=row['target_audit']['visible_words'];lost=visible-len(np.unique(native[native>0]))
        if lost<0:raise RuntimeError('Native owner population exceeds original visible words')
        return dict(sample_id=row['sample_id'],image_path=str(path),image=x,mask=y,validity=v,separation=sep,
            ignored_negative=torch.zeros_like(v),native_words_lost=lost,visible_words=visible,
            outside_image_words=len(row['target_audit']['outside_image_word_ids']))
