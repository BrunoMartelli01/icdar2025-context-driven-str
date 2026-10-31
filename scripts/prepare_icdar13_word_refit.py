"""Audit actual IC13 glyph masks and create isolated connected full-word targets."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
import cv2

from scripts.icdar_budget_data import read,sha,key
from scripts.run_yolo_f1_quality_study import seal
from segmentation.separated_word_gt import parse_icdar13_character_groups,separated_word_regions,audit_word_owners
from segmentation.scene_text_preparation import clip_instances_to_pixel_extent
from segmentation.data import icdar13_task22_rgb_to_target_and_validity
from segmentation.icdar13_word_refit import WordDataset


def prepare(config,paths):
    root=Path(paths['targets_root']);plan=read(paths['parent_plan']);partition=plan['partitions']['icdar13']
    if root.exists():
        old=read(root/'manifest.json')
        if old['parent_plan_sha256']!=sha(paths['parent_plan']) or old['config']!=config:raise ValueError('Changed word target contract')
        for r in old['records']:
            for name in ['image','mask','validity','separator','owners']:
                if sha(r[name+'_path'])!=r[name+'_sha256']:raise ValueError('Changed prepared word target')
        return old
    root.mkdir(parents=True);records=[];audits={};lookup={r['sample_id']:r for r in plan['records']['icdar13']['train']}
    for sid in sorted(lookup):
        r=lookup[sid]
        if sha(r['image_path'])!=r['image_sha256'] or sha(r['annotation_path'])!=r['annotation_sha256']:raise ValueError('Source changed')
        instances=parse_icdar13_character_groups(Path(r['annotation_path']));instances,clip=clip_instances_to_pixel_extent(instances,tuple(r['image_wh']))
        mask,validity,separator,owners,audit=separated_word_regions(tuple(r['image_wh']),instances,ignore_policy='care_region_precedence')
        raw=Path(paths['icdar13_root'])/'raw/train_segmentation_gt'/(sid.split(':')[-1]+'_GT.bmp')
        with Image.open(raw) as p:char_target,char_valid=icdar13_task22_rgb_to_target_and_validity(p,Path(r['annotation_path']).read_text(encoding='utf-8-sig'))
        item=dict(r,care_words=sum(not v.ignore for v in instances),raw_character_mask_path=str(raw),raw_character_mask_sha256=sha(raw))
        for name,value in [('mask',mask),('validity',validity),('separator',separator),('owners',owners)]:
            dest=root/name/(key(sid)+'.png');dest.parent.mkdir(exist_ok=True);value.save(dest)
            item[name+'_path']=str(dest);item[name+'_sha256']=sha(dest)
        cc=cv2.connectedComponents((np.asarray(char_target)>0).astype(np.uint8),connectivity=8)[0]-1
        audit.update(raw_character_components=cc,raw_character_foreground=int((np.asarray(char_target)>0).sum()),
            full_word_foreground=int((np.asarray(mask)>0).sum()),new_pixels_between_characters=int(((np.asarray(mask)>0)&(np.asarray(char_target)==0)).sum()),clipping=clip)
        records.append(item);audits[sid]=audit
    native=WordDataset(records,config,augment=False)
    for i,r in enumerate(records):
        batch=native[i];topology=audit_word_owners(batch['owners'].numpy(),r['care_words'])
        audits[r['sample_id']]['actual_native_target']=topology
        if topology['fragmented_word_ids'] or topology['interword_contact_pairs']:raise ValueError('Unsafe native word topology')
    old=dict(mode='approximate',claim_label='approximate reproduction',config=config,parent_plan_sha256=sha(paths['parent_plan']),
        records=records,partitions=partition,audits=audits,source='IC13 Task2.2 blank-separated character-group hulls; ASSUMPTION, not original Task1 TRAIN polygons',
        summary=dict(images=len(records),care_words=sum(r['care_words'] for r in records),raw_character_components=sum(v['raw_character_components'] for v in audits.values()),
            native_missing_words=sum(len(v['actual_native_target']['missing_word_ids']) for v in audits.values()),native_fragmented_words=sum(len(v['actual_native_target']['fragmented_word_ids']) for v in audits.values()),
            original_word_components=sum(len(v['components_per_word']) for v in audits.values())))
    seal(root/'manifest.json',old);return old


def plot(manifest,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records=sorted(manifest['records'],key=lambda r:(-manifest['audits'][r['sample_id']]['raw_character_components'],r['sample_id']))[:4]
    fig,axes=plt.subplots(len(records),4,figsize=(15,3*len(records)))
    for i,r in enumerate(records):
        with Image.open(r['image_path']) as p:rgb=np.asarray(p.convert('RGB'))
        with Image.open(r['raw_character_mask_path']) as p:chars,_=icdar13_task22_rgb_to_target_and_validity(p,Path(r['annotation_path']).read_text(encoding='utf-8-sig'))
        with Image.open(r['mask_path']) as p:full=np.asarray(p)
        with Image.open(r['owners_path']) as p:owners=np.asarray(p)
        for ax,data,title in zip(axes[i],[rgb,np.asarray(chars),full,owners],['RGB','Old character foreground','New filled word foreground','Word instance IDs']):
            ax.imshow(data,cmap='nipy_spectral' if title=='Word instance IDs' else 'gray');ax.set_title(title);ax.axis('off')
        axes[i,0].set_ylabel(r['sample_id'])
    fig.suptitle('ICDAR2013 TRAIN — source character masks versus derived full-word targets\nApproximate supervision; original raw masks preserved');fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(output,dpi=140);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--paths',type=Path,required=True);p.add_argument('--plot-only',action='store_true');a=p.parse_args()
    if a.config.resolve().parent.name!='approximate' or a.paths.resolve().parent.name!='approximate':raise ValueError('Explicit approximate namespace required')
    c=read(a.config);paths=read(a.paths)
    if c['mode']!='approximate' or paths['mode']!='approximate':raise ValueError('Mode mismatch')
    if a.plot_only:plot(read(Path(paths['targets_root'])/'manifest.json'),Path(paths['targets_root'])/'mask_comparison.png')
    else:print(json.dumps(prepare(c,paths)['summary']))


if __name__=='__main__':main()
