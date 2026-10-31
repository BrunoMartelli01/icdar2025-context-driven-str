"""Independent counts, fold scopes and threshold replay for corrected-U-Net gates."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image,ImageDraw
import torch


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def square_mask(image,size):
    w,h=image.size;side=max(w,h);out=Image.new('L',(side,side),0)
    out.paste(image,((side-w)//2,(side-h)//2))
    return np.asarray(out.resize((size,size),Image.Resampling.NEAREST))>0


def counts(rows,gt,flags):
    normalize=lambda s:''.join(ch for ch in s.casefold() if ch.isalnum())
    predicted=Counter(normalize(r['raw_text']) for r,flag in zip(rows,flags) if not flag)
    expected=Counter(normalize(g['text']) for g in gt)
    predicted.pop('',None);expected.pop('',None)
    tp=sum(min(n,expected[w]) for w,n in predicted.items())
    n=sum(not flag for flag in flags);g=len(gt)
    return dict(tp=tp,fp=n-tp,fn=g-tp,detected_words=n,gt_words=g,
                all_outputs=len(rows),ignored_outputs=sum(flags))


def aggregate(branches,ids,accepted):
    accepted=set(accepted);result=Counter()
    for sid in ids:result.update(branches[sid]['light' if sid in accepted else 'deepsolo'])
    result=dict(result);result['f1']=2*result['tp']/(result['detected_words']+result['gt_words'])
    return result


def audit(root,snapshot):
    plan=read(root/'resolved_configuration.json');cfg=plan['source_config'];result=read(root/'results.json')
    frozen=read(root/'frozen_choices.json');checks=Counter();source={};scalars={};gt_by_id={}
    assert result['mode']=='approximate' and frozen['test_used'] is False
    assert frozen['plan_sha256']==sha(root/'resolved_configuration.json')
    assert read(root/'completion.json')['results_sha256']==sha(root/'results.json')
    for path,digest in cfg['input_sha256'].items():
        assert sha(path)==digest,('input bytes changed',path)
        checks['pinned_source_files']+=1
    for split in ('train','test'):
        raw=read(root/split/'raw.json');edited=read(root/split/'corrected.json')
        source[split]=read(Path(cfg['paths']['replay_root'])/split/'evaluation_rows.json')['samples']
        expected_profiles=[cfg['training_profile']] if split=='train' else cfg['test_profiles']
        tables={p:read(root/split/('branch_metrics_'+p+'.json')) for p in expected_profiles}
        scalars[split]={variant:{} for variant in ('baseline','corrected')}
        ids=[r['sample_id'] for r in plan['records'][split]]
        assert set(raw)==set(edited)==set(ids)
        for record in plan['records'][split]:
            sid=record['sample_id'];sample=record['token_sample'];gt=source[split][sid]['ground_truth'];gt_by_id[split,sid]=gt
            assert source[split][sid]['rgb_sha256']==sample['rgb_sha256']
            assert sha(sample['validity_path'])==sample['validity_sha256']
            with Image.open(sample['validity_path']) as im:validity=square_mask(im.convert('L'),plan['config']['metric']['native_ignore_size'])
            footprints={}
            for variant,observations in [('baseline',raw[sid]),('corrected',edited[sid])]:
                flags=[]
                for row in observations:
                    box=tuple(row['bbox'])
                    if box not in footprints:
                        a,b,x,y=box;mask=Image.new('L',tuple(record['image_wh']),0)
                        ImageDraw.Draw(mask).rectangle([a,b,x-1,y-1],fill=255)
                        footprint=square_mask(mask,validity.shape[0])
                        footprints[box]=bool(footprint.any() and not np.any(footprint&validity))
                    flags.append(footprints[box])
                actual=counts(observations,gt,flags)
                for profile,table in tables.items():
                    assert actual==table[variant][sid]['light'],('independent count differs',split,sid,variant)
                    assert table[variant][sid]['deepsolo']==source[split][sid]['counts']['casefold_alnum']['profiles'][profile]
                    checks['image_expert_pairs']+=1
                path=root/split/'features'/variant/(sid.replace(':','__')+'.npz')
                meta=read(path.with_suffix('.json'))
                assert sha(path)==meta['arrays_sha256']
                assert meta['ground_truth_used'] is False and meta['transcript_likelihood_used'] is False
                with np.load(path,allow_pickle=False) as z:
                    assert not any('label' in name or 'ground_truth' in name for name in z.files)
                    assert z['word_text'].shape[0]==z['word_image'].shape[0]==len(observations)
                    assert z['word_geometry'].shape==(len(observations),plan['config']['architecture']['word_geometry_dim'])
                    if split=='train':scalars[split][variant][sid]=z['scalars'].copy()
                checks['feature_records']+=1
    train_ids=sorted(r['sample_id'] for r in plan['records']['train']);test_ids=sorted(r['sample_id'] for r in plan['records']['test'])
    train_rgb={r['sample_id']:r['token_sample']['rgb_sha256'] for r in plan['records']['train']}
    test_rgb={r['token_sample']['rgb_sha256'] for r in plan['records']['test']}
    assert not set(train_rgb.values())&test_rgb
    held=Counter()
    for fold in frozen['folds']:
        fit,validation=fold['fit_ids'],fold['held_ids']
        assert set(fit)|set(validation)==set(train_ids)
        assert not set(fit)&set(validation)
        assert not {train_rgb[s] for s in fit}&{train_rgb[s] for s in validation}
        held.update(validation);checks['folds']+=1
    assert held==Counter({s:1 for s in train_ids})
    train_branches=read(root/'train'/('branch_metrics_'+cfg['training_profile']+'.json'))
    cv=read(root/'cv_scores.json')
    source_file=snapshot/'recognition/token_context_router.py'
    spec=importlib.util.spec_from_file_location('qualified_corrected_unet_token_source',source_file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    BenefitRouter=module.BenefitRouter
    for path in (root/'models').glob('*.pt'):
        saved=torch.load(path,map_location='cpu',weights_only=True);prep=saved['preprocessing']
        assert saved['plan_sha256']==sha(root/'resolved_configuration.json')
        match=re.fullmatch(r'(baseline|corrected)__fold(\d+)__epoch(\d+)\.pt',path.name)
        if match:
            variant=match[1];fit=frozen['folds'][int(match[2])]['fit_ids']
        else:
            match=re.fullmatch(r'final__(baseline|corrected)__epoch(\d+)\.pt',path.name)
            assert match,path.name
            variant=match[1];fit=train_ids
        assert prep['training_ids']==fit
        x=np.stack([scalars['train'][variant][s] for s in fit]);mean=x.mean(0);std=x.std(0);std[std==0]=1
        assert np.allclose(mean,prep['mean'],rtol=0,atol=1e-6)
        assert np.allclose(std,prep['scale'],rtol=0,atol=1e-6)
        branches=train_branches[variant]
        y=np.asarray([[branches[s]['light']['tp']-branches[s]['deepsolo']['tp'],
                       branches[s]['light']['detected_words']-branches[s]['deepsolo']['detected_words']] for s in fit],np.float32)
        assert np.allclose(y.mean(0),prep['target_mean'],rtol=0,atol=1e-6)
        target_scale=max(float(np.sqrt(np.mean((y-y.mean(0))**2))),plan['config']['training']['minimum_target_scale'])
        assert abs(target_scale-prep['target_scale'])<1e-9
        reference=aggregate(branches,fit,[])['f1'];assert abs(reference-prep['reference_f1'])<1e-15
        torch.manual_seed(prep['seed'])
        model=BenefitRouter(prep['architecture'],plan['config']['architecture']['scalar_dim'],prep['extra_dim'],prep['arm'])
        initial={k:v.clone() for k,v in model.state_dict().items()}
        model.load_state_dict(saved['state_dict'],strict=True)
        assert any(not torch.equal(initial[k],v) for k,v in model.state_dict().items())
        assert saved['optimizer_steps']==math.ceil(len(fit)/plan['config']['training']['batch_size'])*saved['epoch']
        checks['model_fitting_scopes']+=1
    for variant,choice in frozen['variants'].items():
        points=choice['oof_points']
        for epoch,p in points.items():
            score=cv[variant][epoch]
            accepted=sorted((s for s in train_ids if score[s] is not None),key=lambda s:(-score[s],s))[:round(len(train_ids)*cfg['primary_bypass_fraction'])]
            assert sorted(accepted)==p['accepted_ids']
            actual=aggregate(train_branches[variant],train_ids,accepted)
            assert abs(actual['f1']-p['f1'])<1e-15;checks['oof_epoch_points']+=1
        assert choice['epoch']==int(max(points,key=lambda e:(points[e]['f1'],-int(e))))
        assert sha(root/choice['model_path'])==choice['model_sha256']
    for profile,tables in result['profiles'].items():
        branches=read(root/'test'/('branch_metrics_'+profile+'.json'))
        for variant,entry in tables.items():
            scores=read(root/'test'/('token_scores_'+variant+'.json'))['scores']
            points=entry['batch_quotas']+entry['train_transferred_thresholds']
            for p in points:
                if 'bypass_fraction' in p:
                    selected=sorted((s for s in test_ids if scores[s] is not None),key=lambda s:(-scores[s],s))[:round(len(test_ids)*p['bypass_fraction'])]
                else:selected=[s for s in test_ids if scores[s] is not None and scores[s]>=p['threshold']]
                assert sorted(selected)==p['accepted_ids']
                actual=aggregate(branches[variant],test_ids,selected)
                assert all(actual[k]==p[k] for k in ('tp','fp','fn','detected_words','gt_words','f1'))
                assert p['deepsolo_images']==p['total_images']-len(selected)==len(test_ids)-len(selected)
                assert abs(p['deepsolo_percent']-100*p['deepsolo_images']/len(test_ids))<1e-12
                checks['test_route_points']+=1
            stand=aggregate(branches[variant],test_ids,test_ids)
            assert all(stand[k]==entry['standalone'][k] for k in ('tp','fp','fn','f1'))
            assert entry['standalone']['deepsolo_images']==0
    return dict(status='PASS',checks=dict(checks),results_sha256=sha(root/'results.json'),
                scope='Independent lexical Counters, Pillow ignore masks, route replay, grouped-fold scopes, fit-only scaling/targets and strict CPU checkpoint loading; no independent neural refit.',
                new_gpu_inference=False,analysis_code_sha256=sha(Path(__file__)),
                qualified_model_source_sha256=sha(source_file))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--snapshot',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    result=audit(args.run,args.snapshot)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
