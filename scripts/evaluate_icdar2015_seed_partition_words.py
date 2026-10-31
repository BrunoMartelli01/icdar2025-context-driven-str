"""Exploratory fixed-VAL word evaluation of all confidence-partition cases."""

import argparse
import json
from pathlib import Path

from segmentation.data import file_sha256
from segmentation.icdar2015_improvement_runtime import REPO,load_training_population,references,source_hashes
from segmentation.icdar2015_target70 import load_profile,ValidationOCR
from scripts.train_icdar2015_totaltext_transfer import write_json


def run(args):
    c=json.loads(args.config.read_text())
    if args.config.resolve().parent.name!='approximate' or c['reproduction_mode']!='approximate' or c['claim_label']!='approximate reproduction':
        raise ValueError('Explicit approximate word-probe profile required')
    if file_sha256(args.geometry_results)!=c['geometry_results_sha256'] or file_sha256(args.source_words)!=c['source_words_sha256']:
        raise ValueError('Pinned geometry/source-word reports changed')
    geometry=json.loads(args.geometry_results.read_text());original=geometry['run_contract']['config']
    if geometry['status']!='SUCCESS' or len(geometry['candidates'])!=c['expected_candidates']:
        raise ValueError('Complete original geometry grid required')
    p=REPO/original['source_training_profile']
    if file_sha256(p)!=original['source_training_profile_sha256']:raise ValueError('Source profile changed')
    training=load_profile(p);manifest,_,val=load_training_population(training,args.data_root)
    refs=references(manifest,args.data_root,val);shape=original['image_hw'];ocr=ValidationOCR(args,training)
    contract=dict(config=c,config_sha256=file_sha256(args.config),geometry_results=str(args.geometry_results.resolve()),
        data_root=str(args.data_root.resolve()),source_sha256=source_hashes('segmentation/icdar2015_target70.py',
            'segmentation/icdar2015_seed_partition.py','scripts/evaluate_icdar2015_seed_partition_words.py'))
    write_json(args.output_dir/'run_contract.json',contract);candidates=[]
    for index,case in enumerate(geometry['candidates']):
        p=args.geometry_results.parent/case['geometry_file']
        if file_sha256(p)!=case['geometry_sha256']:raise ValueError('Candidate geometry changed')
        detailed=json.loads(p.read_text())
        if len(detailed['per_image'])!=100 or {r['sample_id'] for r in detailed['per_image']}!=set(refs):
            raise ValueError('Every candidate must include fixed VAL100')
        words=ocr.evaluate(detailed,refs,manifest,args.data_root,shape,p.parent/'masks')
        low,high=case['support_threshold'],case['seed_threshold']
        if low==high==training['probability_threshold'] and words['aggregate']!=json.loads(args.source_words.read_text())['aggregate']:
            raise ValueError('Qualified baseline word replay differs')
        directory=args.output_dir/f'support_{low}_seed_{high}';out=directory/'words.json';write_json(out,words)
        row=dict(support_threshold=low,seed_threshold=high,raw=words['aggregate'],emitted=words['emitted_aggregate'],
            emission_policy=words['emission_policy'],words_file=str(out.relative_to(args.output_dir)),words_sha256=file_sha256(out),index=index)
        candidates.append(row);write_json(args.output_dir/'candidates.json',candidates)
        print(f'support={low} seed={high}: rawWordF1={row["raw"]["alphanumeric_casefold"]["f1"]:.6f}, emittedWordF1={row["emitted"]["alphanumeric_casefold"]["f1"]:.6f}',flush=True)
    winner=max(candidates,key=lambda x:(x['emitted']['alphanumeric_casefold']['f1'],x['raw']['alphanumeric_casefold']['f1'],x['raw']['geometry']['f1'],-x['index']))
    write_json(args.output_dir/'results.json',dict(status='SUCCESS',reproduction_mode='approximate',claim_label='approximate reproduction',
        result_class='exploratory complete TRAIN VAL100 correct-word comparison; not TEST',run_contract=contract,
        candidates=candidates,selected=winner,ocr_protocol=ocr.protocol,failed_samples=0,test_data_used=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['config','geometry-results','source-words','data-root','parseq-source','parseq-source-archive','parseq-checkpoint','output-dir']:
        p.add_argument('--'+n,type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists():raise ValueError('Preserve existing output')
    try:run(args)
    except Exception as e:
        write_json(args.output_dir/'failure.json',dict(status='FAILED',error_type=type(e).__name__,error=str(e)))
        raise


if __name__=='__main__':main()
