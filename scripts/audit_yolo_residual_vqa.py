"""Independent mask, probability, calibration and routing audit for frozen VQA."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageFilter
from scripts.audit_yolo_context_quality import (
    sha,read,sealed,cdf,empirical_cut,rank_accept,verify_point,counts_for)


def audit(root):
    plan=read(root/'resolved_configuration.json');cfg=plan['config'];ph=sha(root/'resolved_configuration.json')
    result=read(root/'results.json');frozen=sealed(root/'frozen_thresholds.json',ph);reference=sealed(root/'frozen_references.json',ph)
    assert result['status']=='SUCCESS' and result['plan_sha256']==ph
    assert result['optimizer_steps']==0 and result['new_network_training']is False
    assert result['thresholds_sha256']==sha(root/'frozen_thresholds.json')
    assert result['references_sha256']==sha(root/'frozen_references.json')
    selection=plan['selection_ids'];calibration=plan['threshold_calibration_ids'];assert not set(selection)&set(calibration)
    assert reference['reference_ids']==selection and reference['labels_used']is False
    assert frozen['test_features_present_at_freeze']==0 and frozen['test_labels_used']is False
    for name,d in plan['code_sha256'].items():assert sha(root/'code_snapshot'/name)==d
    for name,d in plan['input_plan']['input_sha256'].items():assert sha(name)==d
    source=Path(plan['input_plan']['paths']['coverage_run']);branches=read(source/'results.json')['resolutions']['640']['branch_metrics']
    assert sha(source/'results.json')==plan['input_plan']['config']['coverage_results_sha256']
    features={};images=0;view_count=0;points_verified=0;diagnostics={}
    for split,samples in plan['samples'].items():
        features[split]={};receipt=sealed(root/split/'vqa_receipt.json',ph);model=receipt['model']
        assert model['model_state_sha256_before']==model['model_state_sha256_after']
        assert model['optimizer_steps']==0 and model['models_trained']is False
        assert model['native_forward_check']['status']=='PASS'
        assert model['cuda_peak']['peak_allocated_bytes']>0 and model['cuda_peak']['peak_reserved_bytes']>0
        assert model['precision']=='float32'and model['vision_batch_size']==1 and model['language_batch_size']==2
        ordinary_blank=[];context_blank=[];full_risk=[];context_risk=[]
        for sample in samples:
            sid=sample['sample_id'];key=sid.replace(':','__')+'.json';images+=1
            p=sealed(root/split/'probes'/key,ph);f=sealed(root/split/'features'/key,ph);features[split][sid]=f
            assert p['ground_truth_used']is False and p['models_trained']is False and p['original_word_outputs_changed']is False
            assert p['image_sha256']==sample['image_sha256']and sha(sample['image_path'])==sample['image_sha256']
            cf=read(source/'640'/split/'features'/key);boxes=[r['bbox']for r in cf['rows']]
            assert p['geometry']['boxes']==boxes
            with Image.open(sample['image_path'])as im:rgb=np.array(im.convert('RGB'))
            mask=np.zeros(rgb.shape[:2],bool)
            for x1,y1,x2,y2 in boxes:mask[y1:y2,x1:x2]=True
            assert p['geometry']['masked_pixels']==int(mask.sum())
            assert p['geometry']['mask_fraction']==float(mask.mean())
            hard=rgb.copy();hard[mask]=cfg['mask_value'];blur=rgb.copy()
            radius=max(1.,cfg['blur_fraction']*min(rgb.shape[:2]))
            blurred=np.asarray(Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(radius)))
            blur[mask]=blurred[mask]
            for name,array in [('original',rgb),('hard_mask',hard),('blur_mask',blur)]:
                assert hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()==p['geometry']['views_rgb_sha256'][name]
                view_count+=1
            v=p['vqa']
            for view in ('original','hard_mask','blur_mask','blank'):
                for condition,q in v[view].items():
                    lp=np.array([q['yes_logprob'],q['no_logprob']]);prob=np.exp(lp-lp.max());prob/=prob.sum()
                    assert abs(prob[0]-q['yes_probability'])<2e-7
                    expected=cfg['plain_question']if condition=='plain'else cfg['context_template'].format(caption=p['caption']['text'])
                    assert q['prompt']==expected and q['tokens']<=512
            risk={k:max(v['hard_mask'][k]['yes_probability'],v['blur_mask'][k]['yes_probability'])for k in ('plain','context')}
            drop={k:v['original'][k]['yes_probability']-risk[k]for k in risk}
            signal={'vqa_residual_plain':-risk['plain'],'vqa_residual_context':-risk['context'],
                'vqa_visual_drop_plain':drop['plain'],'vqa_visual_drop_context':drop['context'],
                'vqa_context_specific':drop['context']-drop['plain']}
            assert signal==p['signals']==f['signals']
            scaled={k:cdf(reference['fields'][k],val)for k,val in signal.items()}
            baseline=p['source_scores']['all_checks_softcrop']
            if not p['eligible']:assert all(val is None for val in f['scores'].values())
            else:
                expected={'coverage_baseline':p['source_scores']['coverage_baseline'],'all_checks_softcrop':baseline,
                    **scaled,**{'base_'+k:(baseline*val)**.5 for k,val in scaled.items()}}
                assert all(abs(expected[k]-val)<1e-12 for k,val in f['scores'].items())
            ordinary_blank.append(v['blank']['plain']['yes_probability']);context_blank.append(v['blank']['context']['yes_probability'])
            full_risk.append(risk['plain']);context_risk.append(risk['context'])
        diagnostics[split]={'images':len(samples),'plain_blank_yes_probability_range':[min(ordinary_blank),max(ordinary_blank)],
            'context_blank_yes_probability_median':float(np.median(context_blank)),
            'plain_residual_yes_median':float(np.median(full_risk)),'context_residual_yes_median':float(np.median(context_risk))}
    assert set(selection+calibration)==set(features['train'])
    for k,values in reference['fields'].items():assert values==sorted(features['train'][s]['signals'][k]for s in selection)
    for family in cfg['families']:
        values={s:f['scores'][family]for s,f in features['train'].items()}
        for t in cfg['target_good_image_fractions']:
            expected=empirical_cut(values,branches['train'],calibration,t,cfg['minimum_calibration_accepted'])
            assert frozen['families'][family]['targets'][str(t)]['threshold']==expected[2]
    for target in cfg['target_good_image_fractions']:
        trials=[]
        for i,family in enumerate(cfg['families']):
            values={s:features['train'][s]['scores'][family]for s in selection}
            count,frac,cut=empirical_cut(values,branches['train'],selection,target,cfg['minimum_calibration_accepted'])
            trials.append((-count,-frac,i,family))
        assert frozen['selected_families'][str(target)]==min(trials)[3]
    for b in cfg['bypass_fractions']:
        trials=[]
        for i,family in enumerate(cfg['families']):
            values={s:features['train'][s]['scores'][family]for s in selection}
            accepted=rank_accept(values,selection,round(len(selection)*b))
            good=sum(10*branches['train'][s]['light']['tp']>=4*(2*branches['train'][s]['light']['tp']+branches['train'][s]['light']['fp']+branches['train'][s]['light']['fn'])
                and 2*branches['train'][s]['light']['tp']+branches['train'][s]['light']['fp']+branches['train'][s]['light']['fn']>0 for s in accepted)
            trials.append((-good,i,family))
        assert frozen['selected_budget_families'][str(b)]==min(trials)[2]
    for name,ids,split in [('train',list(features['train']),'train'),('selection',selection,'train'),
                           ('calibration',calibration,'train'),('test',list(features['test']),'test')]:
        for family,operating in result['splits'][name]['families'].items():
            values={s:features[split][s]['scores'][family]for s in ids}
            for point_name,point in operating.items():
                if point_name.startswith('budget_'):
                    n=round(len(ids)*float(point_name.split('_')[1]));accepted=rank_accept(values,ids,n)
                    assert point['unfilled_budget']==n-len(accepted)
                else:
                    cut=point['threshold'];accepted={s for s in ids if cut is not None and values[s]is not None and values[s]>=cut}
                verify_point(point,branches[split],ids,accepted);points_verified+=1
        for branch,key in [('light','light_only'),('deepsolo','deepsolo_only')]:
            for k,val in counts_for(branches[split],ids,set(),branch).items():assert result['splits'][name][key][k]==val
    assert (root/'frozen_thresholds.json').stat().st_mtime_ns<min(p.stat().st_mtime_ns for p in (root/'test/probes').glob('*.json'))
    audit={'status':'PASS','reproduction_mode':'approximate','claim_label':'approximate reproduction','plan_sha256':ph,
        'results_sha256':sha(root/'results.json'),'audit_code_sha256':sha(__file__),'features_verified':images,
        'original_masked_views_verified':view_count,'routing_points_verified':points_verified,
        'frozen_models_unchanged':True,'native_forward_equivalence_pass':True,'optimizer_steps':0,
        'train_only_selection_calibration':True,'test_labels_used_for_selection':False,'diagnostics':diagnostics}
    (root/'independent_audit.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n');print(json.dumps(audit))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run)
