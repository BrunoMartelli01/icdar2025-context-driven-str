"""Fixed-VAL geometry screen of confidence-based single-output mask splitting."""

import argparse
import json
from pathlib import Path

import numpy as np

from segmentation.data import file_sha256
from segmentation.icdar2015_improvement_runtime import REPO,load_training_population,references,source_hashes
from segmentation.icdar2015_improvements import binary_proposals,original_geometry,valid_canvas
from segmentation.icdar2015_seed_partition import separated_mask
from segmentation.icdar2015_target70 import load_profile
from scripts.train_icdar2015_totaltext_transfer import write_json


def run(args):
    c=json.loads(args.config.read_text())
    if args.config.resolve().parent.name!='approximate' or c['reproduction_mode']!='approximate' or c['claim_label']!='approximate reproduction':
        raise ValueError('Isolated approximate probe required')
    profile=REPO/c['source_training_profile']
    if file_sha256(profile)!=c['source_training_profile_sha256'] or file_sha256(args.geometry)!=c['source_geometry_sha256']:
        raise ValueError('Qualified source profile/maps changed')
    training=load_profile(profile);manifest,_,val=load_training_population(training,args.data_root)
    refs=references(manifest,args.data_root,val);details=json.loads(args.geometry.read_text())
    if {r['sample_id'] for r in details['per_image']}!=set(refs) or len(refs)!=100 or sum(len(r['ground_truth']) for r in refs.values())!=430:
        raise ValueError('Complete fixed VAL100/430care required')
    shape=c['image_hw'];maps={};source_rows={r['sample_id']:r for r in details['per_image']}
    for key,ref in refs.items():
        row=source_rows[key];p=args.mask_dir/row['mask_file']
        if file_sha256(p)!=row['mask_sha256']:raise ValueError('Probability bytes changed')
        with np.load(p,allow_pickle=False) as a:maps[key]=(a['probability'].copy(),a['prediction'].copy())
        probability,prediction=maps[key]
        if probability.shape!=tuple(n//2 for n in shape) or not np.array_equal(prediction,(probability>=training['probability_threshold'])&valid_canvas(ref['image_wh'],shape)):
            raise ValueError('Source probability/canvas mismatch')
    contract=dict(config=c,config_sha256=file_sha256(args.config),data_root=str(args.data_root.resolve()),
        source_geometry=str(args.geometry.resolve()),source_mask_dir=str(args.mask_dir.resolve()),
        source_sha256=source_hashes('segmentation/icdar2015_seed_partition.py','scripts/probe_icdar2015_seed_partition.py',
            'localize/word_multitask.py','localize/crop_recovery.py'))
    write_json(args.output_dir/'run_contract.json',contract);candidates=[]
    for low in c['support_thresholds']:
        for high in c['seed_thresholds']:
            rows=[];tp=props=0;directory=args.output_dir/f'support_{low}_seed_{high}'
            for key,ref in refs.items():
                probability,source_prediction=maps[key]
                prediction,audit=separated_mask(probability,valid_canvas(ref['image_wh'],shape),support_threshold=low,seed_threshold=high)
                if low==high==training['probability_threshold'] and not np.array_equal(prediction,source_prediction):
                    raise RuntimeError('Baseline binary-mask replay differs')
                proposals,_=binary_proposals(prediction,ref['image_wh'],shape)
                counts=original_geometry(proposals,ref['ground_truth'],iou_threshold=c['matching_iou_threshold'])
                p=directory/'masks'/(key.replace(':','_')+'.npz');p.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(p,prediction=prediction)
                rows.append(dict(sample_id=key,**counts,mask_file=p.name,mask_sha256=file_sha256(p),audit=audit))
                tp+=counts['matched_gt'];props+=len(proposals)
            aggregate=dict(matched_gt=tp,gt_words=430,proposals=props,proposal_f1_iou_50=2*tp/(props+430))
            if low==high==training['probability_threshold'] and any(aggregate[k]!=details['aggregate'][k] for k in aggregate):
                raise RuntimeError('Qualified baseline geometry replay differs')
            d=dict(support_threshold=low,seed_threshold=high,aggregate=aggregate,per_image=rows)
            p=directory/'geometry.json';write_json(p,d);candidates.append(dict(support_threshold=low,seed_threshold=high,
                aggregate=aggregate,geometry_file=str(p.relative_to(args.output_dir)),geometry_sha256=file_sha256(p)))
            print(f'support={low} seed={high}: geometryF1={aggregate["proposal_f1_iou_50"]:.6f}, matches={tp}, predictions={props}',flush=True)
    best=max(candidates,key=lambda x:x['aggregate']['proposal_f1_iou_50'])
    write_json(args.output_dir/'results.json',dict(status='SUCCESS',reproduction_mode='approximate',claim_label='approximate reproduction',
        result_class=c['result_class'],run_contract=contract,candidates=candidates,best_geometry=best,
        run_full_ocr=best['aggregate']['proposal_f1_iou_50']>=c['minimum_geometry_f1_for_full_ocr'],failed_samples=0,test_data_used=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['config','data-root','geometry','mask-dir','output-dir']:p.add_argument('--'+n,type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists():raise ValueError('Preserve existing output')
    try:run(args)
    except Exception as e:
        write_json(args.output_dir/'failure.json',dict(status='FAILED',error_type=type(e).__name__,error=str(e)))
        raise


if __name__=='__main__':main()
