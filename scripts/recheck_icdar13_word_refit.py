"""Independent completed IC13 refit recheck, input equivalence and target bounds."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
import torch

from scripts.icdar_budget_data import read,sha,key,annotation,predictions
from scripts.audit_icdar_budget_router import counts,verify_route,ranked,cut_accept,oracle_certificate
from scripts.finish_icdar13_word_refit import collect,audit_heads
from scripts.replay_router_improved_deepsolo import independent_counts
from segmentation.icdar13_word_refit import WordDataset,strict_fallback_count
from segmentation.icdar2015_improvements import make_transform
from recognition.fallback_feasibility import target_minimum,global_oracle,json_point
from recognition.trainable_image_router import oracle


def recheck(root):
    plan=read(root/'resolved_configuration.json');c=plan['config'];p=plan['partitions'];result=read(root/'results.json');checks={}
    torch.set_num_threads(c['cpu_threads'])
    assert plan['mode']=='approximate' and result['mode']=='approximate'
    assert result['status']=='SUCCESS' and not result['test_used_for_selection']
    assert sha(root/'results.json')==read(root/'completion.json')['results_sha256']==read(root/'independent_audit.json')['results_sha256']
    for relative,digest in plan['code_sha256'].items():assert sha(root/'code_snapshot'/relative)==digest
    for relative,digest in read(root/'finish_code_sha256.json').items():assert sha(root/'finish_code_snapshot'/relative)==digest
    for file,digest in c['pinned_inputs'].items():assert sha(file)==digest
    checks['immutable_training_sources']=len(plan['code_sha256'])
    checks['immutable_finish_sources']=len(read(root/'finish_code_sha256.json'))
    parent=read(plan['paths']['parent_plan']);test=parent['records']['icdar13']['test'];records=plan['records'];lookup={r['sample_id']:r for r in records+test}
    rgb=[{lookup[s]['rgb_sha256'] for s in p[role]} for role in ['fit','selection','calibration']]+[{r['rgb_sha256'] for r in test}]
    assert all(not rgb[i]&rgb[j] for i in range(4) for j in range(i+1,4))
    mask_path=Path(plan['paths']['targets_root'])/'manifest.json'
    assert sha(mask_path)==plan['targets_manifest_sha256']
    manifest=read(mask_path)
    for r in manifest['records']:
        for name in ['image','mask','validity','separator','owners']:assert sha(r[name+'_path'])==r[name+'_sha256']
    # Compare real training-loader RGB against the GT-free inference path,
    # including examples outside model selection and masks of every shape.
    data=WordDataset(records,c,augment=False);transform=make_transform(c,c['image_hw'],augment=False)
    for i,r in enumerate(records):
        image=Image.open(r['image_path']).convert('RGB')
        x,_,_=transform.transform_with_validity(image,Image.new('L',image.size),Image.new('L',image.size,255),0)
        assert torch.equal(x,data[i]['image']),r['sample_id']
    checks['exact_training_inference_RGB_tensors']=len(records)
    for branch in ['frozen','partial']:
        history=read(root/branch/'epochs.json');done=read(root/branch/'training.json')
        assert done['training_complete'] and len(history)==c['training']['epochs']
        assert all(r['samples']==len(p['fit']) for r in history)
        assert done['fit_ids']==p['fit'] and done['selection_ids']==p['selection']
        assert done['checkpoint_sha256']==sha(root/branch/'best.pt')
        assert done['best_validation']==read(root/branch/'selected_validation_replay/metrics.json')['aggregate']
    checks['complete_training_epochs']=2*c['training']['epochs']
    ids=p['fit']+p['selection']+p['calibration'];train=[lookup[s] for s in ids]
    features,branches,keys=collect(root,plan,train)
    assert branches==read(root/'train_branches.json')['branches']
    with np.load(root/'training_features.npz',allow_pickle=False) as z:
        assert z['ids'].tolist()==ids
        for i,s in enumerate(ids):
            assert z['word_scale'][i]==features[s]['scale'] and z['eligible'][i]==features[s]['eligible']
            for group in features[s]['groups']:assert np.array_equal(z[group][i],features[s]['groups'][group])
    checks['rebuilt_training_observations']=len(ids)
    checks['independent_head_fits']=audit_heads(root,plan,features,branches,read(root/'trials.json'))
    test_features,tb,tk=collect(root,plan,test);saved=read(root/'test_scores.json');test_ids=[r['sample_id'] for r in test]
    assert tb==saved['branches'] and tk==saved['tie_keys']
    choice=read(root/'frozen_router.json');b=joblib.load(root/choice['choice']['model_path']);assert sha(root/choice['choice']['model_path'])==choice['choice']['model_sha256']
    x=np.stack([np.concatenate([test_features[s]['groups'][g] for g in b['groups']]) for s in test_ids]);values=b['regressor'].predict(b['scaler'].transform(x))
    scores={s:float(v)*test_features[s]['scale'] if test_features[s]['eligible'] else None for s,v in zip(test_ids,values)}
    assert scores==saved['scores']
    for cap,point in result['strict_budget_quotas'].items():
        keep=len(test_ids)-strict_fallback_count(len(test_ids),float(cap));accepted=ranked(scores,test_ids,keep/len(test_ids),tk)
        verify_route(tb,test_ids,accepted,point)
        upper=result['conditional_gt_oracles'][cap];verify_route(tb,test_ids,upper['accepted_ids'],upper)
        oracle_certificate(tb,test_ids,[s for s in test_ids if scores[s] is not None],upper['accepted_ids'])
    for point in result['thresholds']:verify_route(tb,test_ids,cut_accept(scores,point['policy'],tk),point)
    checks['rebuilt_test_observations']=len(test_ids)
    checks['independent_test_routes']=len(result['strict_budget_quotas'])+len(result['thresholds'])
    # A second multiset implementation checks every light output; crop hashes
    # bind stored recognition to original source pixels without another GPU run.
    from scripts.crop_clip_runtime import array_sha
    crops=0
    for r in records+test:
        value=read(root/'experts/refit'/(key(r['sample_id'])+'.json'));gt,valid=annotation(r)
        independent=independent_counts(predictions(value['rows'],r,valid),gt,'casefold_alnum')
        assert independent==(branches if r['split']=='train' else tb)[r['sample_id']]['light']
        with Image.open(r['image_path']) as im:rgb=np.asarray(im.convert('RGB'))
        for row in value['raw_rows']:
            a,b0,c0,d=row['bbox'];assert array_sha(np.ascontiguousarray(rgb[b0:d,a:c0]))==row['crop_rgb_sha256'];crops+=1
    checks['authenticated_original_RGB_crops']=crops
    diagnostics={}
    for name,eligible in [('unrestricted',test_ids),('router_eligible',[s for s in test_ids if scores[s] is not None])]:
        target=target_minimum(tb,c['target_f1'],eligible)
        if target['witness'] is not None:target['witness']=json_point(target['witness'])
        best=global_oracle(tb,eligible);fixed=oracle(tb,test_ids,eligible,len(result['strict_budget_quotas'][str(c['maximum_deepsolo_fraction'])]['accepted_ids']))
        diagnostics[name]=dict(target_feasibility=target,global_oracle=json_point(best),at_primary_budget=dict(**counts(tb,test_ids,fixed['accepted_ids']),
            deepsolo_images=len(test_ids)-len(fixed['accepted_ids']),total_images=len(test_ids),deepsolo_percentage=100*(len(test_ids)-len(fixed['accepted_ids']))/len(test_ids),certificate=fixed['certificate_max_residual']))
    roles={}
    for role,part in [('fit',p['fit']),('selection',p['selection']),('calibration',p['calibration']),('test',test_ids)]:
        source=tb if role=='test' else branches
        roles[role]=dict(light=counts(source,part,part),deepsolo=counts(source,part,[]),images=len(part))
    return dict(status='PASS',mode='approximate',claim_label='approximate reproduction',result_class='complete-dataset post-hoc diagnostic',results_sha256=sha(root/'results.json'),training_plan_sha256=sha(root/'resolved_configuration.json'),analysis_code_sha256=sha(__file__),
        checks=checks,roles=roles,target_diagnostics=diagnostics,scope='Post-hoc verification only; original weights/policies/results unchanged; no TEST threshold tuning or new model inference')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=recheck(a.run_root)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['roles','target_diagnostics']}))


if __name__=='__main__':main()
