"""Fit one matched30-feature YOLO Ridge on original FIT1028, without TEST fitting."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from recognition.essential_portable_router import predict_portable_essential
from recognition.lightweight_unet_router import ocr_summary,mask_summary
from recognition.unet_corrected_token_interfaces import summarize
from scripts.run_essential_icdar_transfer import score_population,yolo_rows

read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args();c=read(args.config)
    if c['mode']!='approximate':raise ValueError('Explicit approximate model fitting required')
    paths=c['paths'];root=Path(paths['icdar_transfer']);output=Path(c['head_output']);output.mkdir(exist_ok=False)
    source=Path(paths['totaltext_features']);native=Path(paths['totaltext_native']);yolo=Path(paths['totaltext_yolo']);plan=read(root/'resolved_configuration.json')
    original=read(plan['config']['paths']['router_deployment']);map_settings=original['map_features'];pins={str(args.config):sha(args.config)}
    schema=read(source/'fit_feature_schema.json');fit_ids=schema['ids'];train_branches=read(native/'train/branches.json')['branches']['yolo']
    trace=read(source/'train_feature_inputs.json');feature_plan=read(source/'resolved_configuration.json')
    metadata={s['sample_id']:s for population in feature_plan['records'].values() for s in population}
    held=read(Path(paths['totaltext_essential'])/'frozen_choices.json')['calibration_ids']
    if len(fit_ids)!=1028 or set(fit_ids)&set(held):raise ValueError('Original fitting/heldout role mismatch')
    def observe(sid,split):
        p=yolo/split/'light'/(sid.replace(':','__')+'.json');value=read(p);pins[str(p)]=sha(p)
        if value['ground_truth_used'] or value['contract']['image_sha256']!=metadata[sid]['image_sha256']:raise ValueError('Changed native YOLO/RGB binding')
        rows=value['rows'];mp=Path(trace[sid]['maps_path'])
        if sha(mp)!=trace[sid]['maps_sha256']:raise ValueError('Changed model maps')
        pins[str(mp)]=sha(mp)
        with np.load(mp,allow_pickle=False) as z:prob=z['probabilities']
        x=np.r_[ocr_summary(rows,metadata[sid]['image_wh']),mask_summary(prob,metadata[sid]['image_wh'],rows,map_settings)]
        return rows,x
    features=[];weights=[]
    for number,sid in enumerate(fit_ids,1):
        rows,x=observe(sid,'train');features.append(x);weights.append(len(rows)+1)
        if number%250==0:print('YOLO FIT observations',number,len(fit_ids),flush=True)
    x=np.stack(features);weights=np.asarray(weights,float);reference=summarize(train_branches[s]['deepsolo'] for s in fit_ids)['f1']
    target=np.asarray([2*(train_branches[s]['light']['tp']-train_branches[s]['deepsolo']['tp'])-reference*(train_branches[s]['light']['detected_words']-train_branches[s]['deepsolo']['detected_words']) for s in fit_ids])/weights
    scaler=StandardScaler().fit(x);z=scaler.transform(x);head=Ridge(alpha=c['fit']['alpha']).fit(z,target,sample_weight=weights)
    deployment=dict(schema_version=1,reproduction_mode='approximate',claim_label='approximate reproduction',expert='Native YOLO/PARSeq with shared frozen U-Net maps',
        coef=head.coef_.tolist(),intercept=float(head.intercept_),mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),map_features=map_settings,policies=[],
        numeric_inference_bytes=int(head.coef_.nbytes+head.intercept_.nbytes+scaler.mean_.nbytes+scaler.scale_.nbytes),fitting_images=len(fit_ids),fitting_reference_f1=reference)
    # Seal learned weights and all TRAIN identities before observing this new head's TEST scores.
    save(output/'deployment.json',deployment);save(output/'fit_metadata.json',dict(config=c,fit_ids=fit_ids,reference_f1=reference,
        input_sha256=pins.copy(),deployment_sha256=sha(output/'deployment.json'),test_used=False))
    np.savez_compressed(output/'fit_arrays.npz',x=x,y=target,weights=weights)
    old_vb=read(root/'threshold_analysis_v1/validation_branches.json');old_tb_path=root/'threshold_analysis_v1/test_branches.json'
    branches={ds:dict(yolo=old_vb[ds]['yolo']) for ds in c['datasets']};scores={ds:dict(yolo={}) for ds in c['datasets']}
    for sid in held:
        rows,x=observe(sid,'train');score=float(((x-deployment['mean'])/deployment['scale'])@head.coef_+head.intercept_)*(len(rows)+1)
        scores['total_text']['yolo'][sid]=score if rows and all(r['raw_text'].strip() for r in rows) else None
    for ds,samples in plan['population']['calibration'].items():
        for s in samples:
            sid=s['sample_id'];u=read(root/'unet'/(sid.replace(':','__')+'.json'))
            with np.load(root/u['maps_path'],allow_pickle=False) as z:prob=z['probabilities']
            rows,p=yolo_rows(root,plan,'calibration',ds,s);scores[ds]['yolo'][sid]=predict_portable_essential(deployment,sid,rows,s['image_wh'],prob)['score']
    save(output/'validation_branches.json',branches);save(output/'validation_scores.json',scores)
    # Prepared TEST scores are GT-blind predictions. The separate frontier runner
    # still freezes all validation choices before loading these TEST artifacts.
    tb=read(old_tb_path);branches={ds:dict(yolo=tb[ds]['yolo']) for ds in c['datasets']};scores={ds:dict(yolo={}) for ds in c['datasets']}
    test_schema=read(source/'test_feature_schema.json');trace.update(read(source/'test_feature_inputs.json'))
    for sid in test_schema['ids']:
        rows,x=observe(sid,'test');score=float(((x-deployment['mean'])/deployment['scale'])@head.coef_+head.intercept_)*(len(rows)+1)
        scores['total_text']['yolo'][sid]=score if rows and all(r['raw_text'].strip() for r in rows) else None
    for ds,samples in plan['population']['test'].items():
        for s in samples:
            sid=s['sample_id'];u=read(root/'unet'/(sid.replace(':','__')+'.json'))
            with np.load(root/u['maps_path'],allow_pickle=False) as z:prob=z['probabilities']
            rows,p=yolo_rows(root,plan,'test',ds,s);scores[ds]['yolo'][sid]=predict_portable_essential(deployment,sid,rows,s['image_wh'],prob)['score']
    save(output/'test_branches.json',branches);save(output/'test_scores.json',scores)
    save(output/'prepared_receipt.json',dict(status='SUCCESS',deployment_sha256=sha(output/'deployment.json'),
        files_sha256={name:sha(output/name) for name in ['fit_metadata.json','fit_arrays.npz','validation_branches.json','validation_scores.json','test_branches.json','test_scores.json']}))
    print('YOLO-specific essential head fit complete:',deployment['numeric_inference_bytes'],'numerical bytes',flush=True)

if __name__=='__main__':main()
