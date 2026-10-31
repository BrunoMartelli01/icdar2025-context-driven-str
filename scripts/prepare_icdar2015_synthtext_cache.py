"""Recover the pinned existing SynthText cohort and cache audited v2 targets."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
from hashlib import sha256
import json
import multiprocessing
import os
from pathlib import Path

from segmentation.data import file_sha256,load_segmentation_manifest
from segmentation.icdar2015_improvement_runtime import REPO,source_hashes
from segmentation.icdar2015_synthtext_cache import prepare_sample
from scripts.train_icdar2015_totaltext_transfer import write_json


def initialize(image_root,output,policy,excluded):
    global WORKER
    import cv2
    import torch
    cv2.setNumThreads(1);torch.set_num_threads(1)
    WORKER=(image_root,output,policy,excluded)


def worker(row):return prepare_sample(row,*WORKER)


def run(a):
    c=json.loads(a.config.read_text())
    if a.config.resolve().parent.name!='approximate' or c['reproduction_mode']!='approximate' or c['claim_label']!='approximate reproduction':
        raise ValueError('Explicit approximate data-preparation profile required')
    with gzip.open(a.source_manifest_gzip,'rb') as f:raw=f.read()
    if sha256(raw).hexdigest()!=c['source_manifest_sha256']:raise ValueError('Original cohort manifest changed')
    source=json.loads(raw);del raw
    if source['images']!=c['images'] or source['coordinate_origin']!=c['coordinate_origin'] or source['gt_mat_sha256']!=c['gt_mat_sha256']:
        raise ValueError('Original cohort/coordinate provenance changed')
    if file_sha256(a.image_root/'gt.mat')!=c['gt_mat_sha256']:raise ValueError('Original MAT annotation bytes changed')
    expected={x['sha256']:x for x in c['excluded_manifests']};exclusions=[];excluded=set()
    if len(a.exclusion_manifest)!=len(expected):raise ValueError('All exclusion populations required')
    for p in a.exclusion_manifest:
        h=file_sha256(p)
        if h not in expected:raise ValueError('Unexpected exposure manifest')
        m=load_segmentation_manifest(p,require_checksums=True)
        if len(m.samples)!=expected[h]['images']:raise ValueError('Incomplete exclusion population')
        excluded.update(s.image_sha256 for s in m.samples);exclusions.append(dict(path=str(p.resolve()),**expected.pop(h)))
    if expected:raise ValueError('Missing exposure population')
    rows=[];ids=set();hashes=set()
    for row in source['samples']:
        if row['sample_id'] in ids or row['image_sha256'] in hashes or row['image_sha256'] in excluded:
            raise ValueError('Duplicate or excluded source image; never silently replace cohort members')
        ids.add(row['sample_id']);hashes.add(row['image_sha256'])
        relative=Path(row['image_path']).relative_to(source['root'])
        rows.append(dict(sample_id=row['sample_id'],image_relative=str(relative),image_sha256=row['image_sha256'],words=row['words']))
    if len(rows)!=c['images']:raise ValueError('Incomplete original cohort')
    contract=dict(config=c,config_sha256=file_sha256(a.config),source_manifest_gzip_sha256=file_sha256(a.source_manifest_gzip),
        image_root=str(a.image_root.resolve()),output_dir=str(a.output_dir.resolve()),exclusions=exclusions,
        source_sha256=source_hashes('segmentation/icdar2015_synthtext_cache.py','segmentation/synthtext_words.py',
            'scripts/prepare_icdar2015_synthtext_cache.py','segmentation/unet_detection_methods.py'))
    p=a.output_dir/'run_contract.json'
    if p.exists():
        if not a.resume or json.loads(p.read_text())!=contract:raise ValueError('Exact-contract resume required')
        if (a.output_dir/'results.json').exists():
            r=json.loads((a.output_dir/'results.json').read_text())
            if r['status']!='SUCCESS' or file_sha256(a.output_dir/'manifest.json')!=r['manifest_sha256']:raise ValueError('Completed cache changed')
            return r
    else:
        if a.resume:raise ValueError('Cannot resume missing preparation')
        write_json(p,contract)
    completed=[]
    write_json(a.output_dir/'live_status.json',dict(pid=os.getpid(),status='PREPARING',completed=0,total=len(rows)))
    with ProcessPoolExecutor(max_workers=c['workers'],mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize,initargs=(str(a.image_root),str(a.output_dir),c['target_policy'],excluded)) as pool:
        for row in pool.map(worker,rows,chunksize=8):
            completed.append(row)
            if len(completed)%500==0:
                write_json(a.output_dir/'live_status.json',dict(pid=os.getpid(),status='PREPARING',completed=len(completed),total=len(rows)))
                print(f'SynthText targets {len(completed)}/{len(rows)}',flush=True)
    totals=dict(annotation_words=sum(r['target_audit']['annotation_words'] for r in completed),
        visible_words=sum(r['target_audit']['visible_words'] for r in completed),
        outside_image_words=sum(len(r['target_audit']['outside_image_word_ids']) for r in completed),
        restored_contact_words=sum(len(r['target_audit']['restored_contact_word_ids']) for r in completed))
    if totals!=c['expected_target_summary']:raise ValueError('Qualified full-cohort target audit changed')
    manifest=dict(schema_version=1,reproduction_mode='approximate',claim_label='approximate reproduction',dataset='SynthText',
        subset='same existing deterministic50000-image external training cohort',images=len(completed),samples=completed,
        source_manifest_sha256=c['source_manifest_sha256'],gt_mat_sha256=c['gt_mat_sha256'],coordinate_origin=c['coordinate_origin'],
        target_policy=c['target_policy'],target_summary=totals,run_contract=contract,
        leakage_limitation='Exact byte overlap excluded; visually similar scenes unresolved. TEST manifest used only for RGB-hash leakage protection.')
    write_json(a.output_dir/'manifest.json',manifest)
    result=dict(status='SUCCESS',reproduction_mode='approximate',claim_label='approximate reproduction',
        result_class='training-data cache preparation; no model training or evaluation',images=len(completed),target_summary=totals,
        manifest_sha256=file_sha256(a.output_dir/'manifest.json'),run_contract=contract)
    write_json(a.output_dir/'results.json',result);write_json(a.output_dir/'live_status.json',dict(pid=os.getpid(),status='SUCCESS',completed=len(completed),total=len(rows)))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['config','source-manifest-gzip','image-root','output-dir']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--exclusion-manifest',type=Path,action='append',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.output_dir.exists() and not a.resume:raise ValueError('Preserve existing output; explicit resume required')
    try:run(a)
    except Exception as e:
        write_json(a.output_dir/'failure.json',dict(status='FAILED',error_type=type(e).__name__,error=str(e)))
        write_json(a.output_dir/'live_status.json',dict(pid=os.getpid(),status='FAILED',error_type=type(e).__name__,error=str(e)))
        raise


if __name__=='__main__':main()
