"""Independent full-artifact verification using OpenCV binary components."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import find_objects
import yaml


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path, manifest_path: Path, v4_root: Path):
    data=json.loads(manifest_path.read_text())
    config_path=manifest_path.parent/'resolved_config.json'
    config=json.loads(config_path.read_text())
    if data['claim_label']!='approximate reproduction' or config['reproduction_mode']!='approximate':
        raise ValueError('Scientific mode/label differs')
    if sha256(config_path)!=data['resolved_config_sha256']:
        raise ValueError('Resolved config checksum mismatch')
    samples=data['samples']
    if len(samples)!=data['images'] or len({s['id'] for s in samples})!=len(samples):
        raise ValueError('Sample count or duplicate sample IDs')
    v4_rows={}
    if data['dataset']=='total_text' and data['split']=='train':
        m=yaml.safe_load((v4_root/'manifests/total_text_train.yaml').read_text())
        v4_rows={r['id']:r for r in m['samples']}
    words=fragmented=protected=removed=0
    for index,sample in enumerate(samples,1):
        image=Path(sample['image'])
        if sha256(image)!=sample['image_sha256']:
            raise ValueError(f"Source image drift: {sample['id']}")
        with Image.open(image) as im:
            size=im.size
        arrays={}
        for name in ['masks','owners','validity','separators']:
            path=root/sample[name]
            if sha256(path)!=sample[name+'_sha256']:
                raise ValueError(f'Checksum mismatch: {path}')
            with Image.open(path) as im:
                if im.size!=size:
                    raise ValueError(f'Image/mask coordinate mismatch: {path}')
                arrays[name]=np.asarray(im).copy()
        mask=arrays['masks'];own=arrays['owners'];valid=arrays['validity'];sep=arrays['separators']
        if not set(np.unique(mask)).issubset({0,255}) or not set(np.unique(valid)).issubset({0,255}):
            raise ValueError('Nonbinary target or validity')
        if not np.array_equal(mask>0,own>0) or np.any((own>0)&(valid==0)) or np.any((mask>0)&(sep>0)):
            raise ValueError(f"Mask/owner/validity/separator inconsistency: {sample['id']}")
        expected=len(sample['instances'])
        if set(np.unique(own[own>0]))!=set(range(1,expected+1)):
            raise ValueError(f"A word disappeared or an owner was invented: {sample['id']}")
        n,components=cv2.connectedComponents((mask>0).astype(np.uint8),connectivity=8)
        component_owners={};counts={}
        for word_id,window in enumerate(find_objects(own,max_label=expected),1):
            ids=np.unique(components[window][own[window]==word_id])
            counts[str(word_id)]=len(ids)
            for component in ids:
                component_owners.setdefault(int(component),set()).add(word_id)
        if any(len(v)>1 for v in component_owners.values()) or len(component_owners)!=n-1:
            raise ValueError(f"Different words are joined in an8-connected component: {sample['id']}")
        if counts!=sample['gap_audit']['components_per_word']:
            raise ValueError('Logged component counts differ from independent OpenCV audit')
        if 'source_annotation' in sample and sha256(Path(sample['source_annotation']))!=sample['source_annotation_sha256']:
            raise ValueError('Source annotation drift')
        if sample.get('protected_pixel_gt'):
            if sha256(Path(sample['protected_pixel_gt']))!=sample['protected_pixel_gt_sha256']:
                raise ValueError('Protected GT drift')
        if v4_rows:
            old=v4_rows[sample['id']]
            with Image.open(v4_root/old['word_owner_annotation']) as im:
                before=np.asarray(im)
            if np.any((own>0)&(own!=before)):
                raise ValueError('New V4 target added or reassigned pixels')
            with Image.open(sample['protected_pixel_gt']) as im:
                seeds=np.asarray(im)>0
            protected_seed=seeds&(before>0)
            if np.any(protected_seed&(own!=before)):
                raise ValueError('V4-derived mask lost a source stroke seed')
            protected+=int(protected_seed.sum())
            removed+=int(np.count_nonzero(before)-np.count_nonzero(own))
        words+=expected
        fragmented+=sum(count>1 for count in counts.values())
        if index%100==0 or index==len(samples):
            print(f"[independent audit] {data['dataset']}/{data['split']}: {index}/{len(samples)}",flush=True)
    if words!=data['words'] or fragmented!=data['fragmented_words']:
        raise ValueError('Manifest word/fragmentation aggregate mismatch')
    if v4_rows and removed!=data['gap_pixels_removed']:
        raise ValueError('V4 foreground reduction differs from report')
    return {'dataset':data['dataset'],'split':data['split'],'images':len(samples),'words':words,
            'missing_words':0,'merged_word_components':0,'fragmented_words':fragmented,
            'v4_protected_stroke_pixels_retained':protected,'v4_removed_pixels':removed,
            'manifest_sha256':sha256(manifest_path)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--v4-root',type=Path,required=True)
    parser.add_argument('--dataset',nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    reports=[]
    for dataset in args.dataset:
        manifests=sorted((args.root/dataset).glob('*/manifest.json'))
        if not manifests:
            raise ValueError(f'No completed export for {dataset}')
        for manifest in manifests:
            reports.append(verify(args.root,manifest,args.v4_root))
    args.output.write_text(json.dumps({'reproduction_mode':'approximate',
        'claim_label':'approximate reproduction','verifier':'OpenCV8-connected binary components',
        'verifier_sha256':sha256(Path(__file__)),'datasets':reports},indent=2)+'\n')


if __name__=='__main__':
    main()
