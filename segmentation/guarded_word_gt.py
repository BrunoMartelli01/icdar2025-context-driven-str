"""Opt-in APPROXIMATE native word-target repair, after the real paired transform.

V1/V2 files and the legacy transform remain unchanged. New pixels may only
restore the SAME V1 owner in the native reference raster and valid region.
Failures are retained and reported, never hidden by dropping fragments/words.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, distance_transform_edt, find_objects, label
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from segmentation.data import SegmentationDataset, file_sha256, load_segmentation_manifest

NEIGHBOURS = np.ones((3, 3), dtype=np.uint8)


@dataclass(frozen=True)
class WordGuardConfig:
    max_growth_native: float
    growth_steps: int
    min_area_fraction: float
    min_span_fraction: float
    crop_padding_native: int
    thin_bbox_side_native: int
    unresolved_policy: str

    def __post_init__(self):
        if (self.max_growth_native <= 0 or type(self.growth_steps) is not int or self.growth_steps < 1
                or not 0 < self.min_area_fraction <= 1 or not 0 < self.min_span_fraction <= 1
                or type(self.crop_padding_native) is not int or self.crop_padding_native < 0
                or type(self.thin_bbox_side_native) is not int or self.thin_bbox_side_native < 1
                or self.unresolved_policy != 'retain_all_original_fragments_and_report'):
            raise ValueError('Invalid explicit native word guard configuration')


def describe(mask):
    ys, xs = np.where(mask)
    if not len(ys):
        return {'components':0, 'pixels':0, 'bbox':None}
    box=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
    roi=mask[box[1]:box[3],box[0]:box[2]]
    return {'components':int(label(roi,structure=NEIGHBOURS)[1]),'pixels':len(ys),'bbox':box}


def defects(candidate, reference, config):
    c, r=describe(candidate), describe(reference)
    failed=[]
    if not r['pixels']:
        return ['no_native_reference_support']
    if c['components']!=1:
        failed.append('missing' if not c['pixels'] else 'fragmented')
    if not c['pixels']:
        return failed
    cb, rb=c['bbox'],r['bbox'];cw,ch=cb[2]-cb[0],cb[3]-cb[1];rw,rh=rb[2]-rb[0],rb[3]-rb[1]
    if c['pixels'] < config.min_area_fraction*r['pixels']:
        failed.append('area')
    if cw < config.min_span_fraction*rw or ch < config.min_span_fraction*rh:
        failed.append('span')
    p=config.crop_padding_native
    if cb[0]-p>rb[0] or cb[1]-p>rb[1] or cb[2]+p<rb[2] or cb[3]+p<rb[3]:
        failed.append('crop_coverage')
    if min(rw,rh)<=config.thin_bbox_side_native and not np.array_equal(candidate,reference):
        failed.append('thin_word_protection')
    return failed


def contact_pairs(owners):
    h,w=owners.shape;contacts=0
    for dy,dx in [(0,1),(1,-1),(1,0),(1,1)]:
        a=owners[max(0,-dy):h-max(0,dy),max(0,-dx):w-max(0,dx)]
        b=owners[max(0,dy):h-max(0,-dy),max(0,dx):w-max(0,-dx)]
        contacts+=int(np.count_nonzero((a>0)&(b>0)&(a!=b)))
    return contacts


def guard_native_owners(compact, reference, valid, expected_ids, config):
    compact=np.asarray(compact);reference=np.asarray(reference);valid=np.asarray(valid,dtype=bool)
    if compact.ndim!=2 or compact.shape!=reference.shape or valid.shape!=compact.shape:
        raise ValueError('Native guard geometry mismatch')
    if not np.issubdtype(compact.dtype,np.integer) or not np.issubdtype(reference.dtype,np.integer):
        raise ValueError('Native owner rasters must be integer')
    if np.any(compact<0) or np.any(reference<0) or np.any((compact>0)&(compact!=reference)):
        raise ValueError('Compact owners are not a same-ID subset of V1 reference')
    result=compact.astype(np.int32,copy=True);result[~valid]=0
    if contact_pairs(result):
        raise ValueError('Frozen compact target already has cross-owner contacts')
    rows=[];windows=find_objects(reference,max_label=max(expected_ids,default=0))
    for owner in expected_ids:
        window=windows[owner-1]
        if window is None:
            rows.append({'owner':owner,'before_components':0,'after_components':0,
                'before_pixels':0,'after_pixels':0,'changed':False,'accepted_source':'unchanged',
                'initial_defects':['no_native_reference_support'],'unresolved':['no_native_reference_support']})
            continue
        # Include one neighbour cell when constraining source restoration.
        ys,xs=window;y0,y1=max(0,ys.start-1),min(result.shape[0],ys.stop+1)
        x0,x1=max(0,xs.start-1),min(result.shape[1],xs.stop+1)
        local=result[y0:y1,x0:x1];support=(reference[y0:y1,x0:x1]==owner)&valid[y0:y1,x0:x1]
        current=local==owner;before=describe(current);initial=defects(current,support,config)
        chosen=current;accepted='unchanged'
        if initial:
            other=(local>0)&(local!=owner)
            allowed=support&~binary_dilation(other,structure=NEIGHBOURS)
            candidates=[]
            if current.any():
                distance=distance_transform_edt(~current)
                candidates=[(f'growth_{float(radius):.6g}',current|((distance<=radius)&allowed))
                    for radius in np.linspace(config.max_growth_native/config.growth_steps,
                                             config.max_growth_native,config.growth_steps)]
            candidates.append(('native_owner_reference',allowed))
            for name,candidate in candidates:
                if not defects(candidate,support,config):
                    chosen=candidate;accepted=name;break
            if accepted!='unchanged':
                local[current]=0;local[chosen]=owner
        after=describe(chosen)
        rows.append({'owner':owner,'before_components':before['components'],'after_components':after['components'],
            'before_pixels':before['pixels'],'after_pixels':after['pixels'],
            'changed':not np.array_equal(current,chosen),'accepted_source':accepted,
            'initial_defects':initial,'unresolved':defects(chosen,support,config)})
    if contact_pairs(result) or np.any((result>0)&((result!=reference)|~valid)):
        raise AssertionError('Repair violated cross-word separation or source support')
    if np.any((compact>0)&valid&(result!=compact)):
        raise AssertionError('Repair must not erase original compact pixels/words')
    stats={'words':len(rows),'changed_words':sum(r['changed'] for r in rows),
        'missing_before':sum(r['before_components']==0 for r in rows),'missing_after':sum(r['after_components']==0 for r in rows),
        'fragmented_before':sum(r['before_components']>1 for r in rows),'fragmented_after':sum(r['after_components']>1 for r in rows),
        'unresolved_words':sum(bool(r['unresolved']) for r in rows),'contact_pairs_after':0,
        'coverage_failures_after':sum('crop_coverage' in r['unresolved'] for r in rows),
        'thin_protection_failures_after':sum('thin_word_protection' in r['unresolved'] for r in rows),
        'foreground_pixels_before':int(np.count_nonzero((compact>0)&valid)),
        'foreground_pixels_after':int(np.count_nonzero(result))}
    return result,{'stats':stats,'per_word':rows}


class GuardedWordTransform:
    """Legacy-compatible transform adapter with a pinned V1 reference manifest.

    The existing SegmentationDataset automatically calls this separation-aware
    API. set_epoch drives precisely the same paired augmentation as before.
    """
    def __init__(self, base, compact_manifest, reference_manifest, reference_root, config):
        if [s.sample_id for s in compact_manifest.samples]!=[s.sample_id for s in reference_manifest.samples]:
            raise ValueError('V1/V2 sample ordering differs')
        for a,b in zip(compact_manifest.samples,reference_manifest.samples):
            if a.image_sha256!=b.image_sha256:
                raise ValueError('V1/V2 RGB provenance differs')
        self.base=base;self.reference_manifest=reference_manifest;self.reference_root=Path(reference_root)
        self.guard_config=config;self.last_audit=None;self.last_owners=None

    def set_epoch(self, epoch):
        self.base.set_epoch(epoch)

    def _native_owner(self, owner_image, image_size, index):
        dummy=Image.new('RGB',image_size)
        _,owners,_=self.base._augment(dummy,owner_image,None,index)
        owners=TF.resize(owners,list(self.base.config.mask_size),interpolation=InterpolationMode.NEAREST)
        return np.asarray(owners,dtype=np.int32).copy()

    def transform_with_validity_and_separation(self,image,mask,validity,separation,owners,sample_index):
        output=self.base.transform_with_validity_and_separation(image,mask,validity,separation,owners,sample_index)
        rgb,target,valid,sep=output
        ref=self.reference_manifest.samples[sample_index];path=self.reference_root/ref.word_owner_annotation_path
        if file_sha256(path)!=ref.word_owner_annotation_sha256:
            raise ValueError('V1 reference owners changed')
        with Image.open(path) as im:reference_image=im.convert('I')
        if reference_image.size!=image.size:
            raise ValueError('V1 reference and V2 image geometries differ')
        expected=[int(x) for x in np.unique(np.asarray(reference_image)) if x>0]
        raw_reference=self._native_owner(reference_image,image.size,sample_index)
        raw_compact=self._native_owner(owners,image.size,sample_index)
        valid_array=valid[0].numpy()>0
        current=np.where((target[0].numpy()>0)&valid_array,raw_compact,0)
        guarded,audit=guard_native_owners(current,raw_reference,valid_array,expected,self.guard_config)
        foreground=torch.from_numpy(guarded>0).unsqueeze(0)
        new_target=foreground.to(torch.float32)
        # Restored foreground must not retain the old V2 negative separator.
        new_sep=torch.maximum(sep,torch.from_numpy((raw_reference>0)&valid_array&~(guarded>0)).unsqueeze(0).float())
        new_sep*=~foreground
        self.last_owners=guarded;self.last_audit={**audit,'sample_id':ref.sample_id,
            'epoch':self.base.epoch,'augmentation_enabled':bool(self.base.config.rotation_probability or self.base.config.horizontal_flip_probability)}
        return rgb,new_target,valid,new_sep

    def metadata(self):
        return {**self.base.metadata(),'target_variant':'guarded-word-targets-v3','guard':asdict(self.guard_config),
            'claim_label':'approximate reproduction','original_transform_unchanged':True}


def make_guarded_dataset(config_path, *, augmentation_enabled, verify_checksums=True):
    """Explicit opt-in factory; no changes to legacy or STRICT config lookup."""
    from scripts.review_native128_ocr import inference_config
    from segmentation.train import _transform
    path=Path(config_path);c=json.loads(path.read_text())
    if path.resolve().parent.name!='approximate' or c['reproduction_mode']!='approximate' or c['variant']!='guarded-word-targets-v3':
        raise ValueError('Explicit isolated APPROXIMATE V3 config required')
    source=Path(c['source_training_config'])
    if file_sha256(source)!=c['source_training_config_sha256']:
        raise ValueError('Pinned source transform configuration changed')
    manifests={}
    for key in ['reference','compact']:
        record=c[key];mp=Path(record['manifest'])
        if file_sha256(mp)!=record['manifest_sha256']:
            raise ValueError('Pinned source manifest changed')
        manifests[key]=load_segmentation_manifest(mp,require_checksums=True)
        if manifests[key].split_role!='train':
            raise ValueError('V3 recipe is TRAIN-only; TEST annotations are not inputs')
    cfg=inference_config(source)
    base=_transform(cfg.transforms,seed=c['seed'],augmentation_enabled=augmentation_enabled)
    guard=GuardedWordTransform(base,manifests['compact'],manifests['reference'],c['reference']['data_root'],WordGuardConfig(**c['guard']))
    return SegmentationDataset(manifests['compact'],c['compact']['data_root'],transform=guard,verify_checksums=verify_checksums)
