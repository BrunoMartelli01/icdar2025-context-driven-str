"""Independent full-word topology and official Task1 identity audit for IC13 refits."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import numpy as np
from PIL import Image
from scipy.ndimage import label

from scripts.icdar_budget_data import read,sha
from segmentation.icdar13_word_refit import WordDataset
from segmentation.separated_word_gt import parse_icdar13_character_groups


def normalize(text):return ''.join(c for c in text.casefold() if c.isalnum())


def audit(manifest_path,official_receipt,training_plan):
    manifest=read(manifest_path);official=read(official_receipt);plan=read(training_plan)
    if plan['targets_manifest_sha256']!=sha(manifest_path) or plan['config']!=manifest['config']:raise ValueError('Active training does not use these audited targets')
    if official['archive_md5']!=official['reference_md5']:raise ValueError('Official annotation archive checksum differs')
    by={r['sample_id']:r for r in official['records']};dataset=WordDataset(manifest['records'],manifest['config'],augment=False)
    words=lost=fragmented=fused=0;raw_differences=[];per_image={}
    for i,r in enumerate(manifest['records']):
        sid=r['sample_id'].split(':')[-1];source=by[sid]
        if sha(source['annotation_path'])!=source['sha256']:raise ValueError('Changed official word annotation')
        gt=[]
        for line in Path(source['annotation_path']).read_text(encoding='utf-8-sig').splitlines():
            m=re.fullmatch(r'\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+"(.*)"\s*',line)
            if not m:raise ValueError('Unexpected official Task1 annotation')
            gt.append(m.group(5))
        groups=[w.transcription for w in parse_icdar13_character_groups(Path(r['annotation_path'])) if not w.ignore]
        if Counter(normalize(s) for s in groups)!=Counter(normalize(s) for s in gt):raise ValueError('Normalized official word identities differ: '+sid)
        if Counter(groups)!=Counter(gt):raw_differences.append(sid)
        if len(gt)!=r['care_words']:raise ValueError('Care word denominator differs')
        for name in ['image','mask','validity','separator','owners']:
            if sha(r[name+'_path'])!=r[name+'_sha256']:raise ValueError('Prepared input changed')
        with Image.open(r['mask_path']) as p:mask=np.asarray(p)>0
        with Image.open(r['owners_path']) as p:owners=np.asarray(p)
        if not np.array_equal(mask,owners>0):raise ValueError('Full-word foreground is not the ownership union')
        if set(np.unique(owners))!={0,*range(1,len(gt)+1)}:raise ValueError('Source word IDs differ')
        for owner in range(1,len(gt)+1):
            if label(owners==owner,structure=np.ones((3,3)))[1]!=1:raise ValueError('Original word is disconnected')
        batch=dataset[i];target=batch['mask'][0].numpy()>0;native=batch['owners'].numpy();components,n=label(target,structure=np.ones((3,3)))
        associations={};missing=[];split=[]
        for owner in range(1,len(gt)+1):
            ids=set(int(v) for v in np.unique(components[native==owner]) if v)
            if not ids:missing.append(owner)
            if len(ids)>1:split.append(owner)
            for component in ids:associations.setdefault(component,set()).add(owner)
        merged=sum(len(v)>1 for v in associations.values())
        lost+=len(missing);fragmented+=len(split);fused+=merged;words+=len(gt)
        per_image[r['sample_id']]=dict(words=len(gt),actual_native_components=n,missing_word_ids=missing,fragmented_word_ids=split,fused_components=merged)
    if lost or fragmented or fused:raise ValueError('Actual full-word native targets do not preserve word topology')
    return dict(status='PASS',mode='approximate',claim_label='approximate reproduction',images=len(manifest['records']),official_care_words=words,
        native_lost_words=lost,native_fragmented_words=fragmented,native_fused_components=fused,normalized_word_transcription_mismatches=0,
        raw_punctuation_difference_images=raw_differences,manifest_sha256=sha(manifest_path),official_acquisition_sha256=sha(official_receipt),
        active_training_plan_sha256=sha(training_plan),analysis_code_sha256=sha(__file__),per_image=per_image,
        scope='Independent SciPy connectivity on actual loader targets; official Task1 normalized word identity; hull geometry remains an explicit approximation')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--official-receipt',type=Path,required=True)
    p.add_argument('--training-plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=audit(a.manifest,a.official_receipt,a.training_plan)
    with a.output.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in out.items() if k!='per_image'}))


if __name__=='__main__':main()
