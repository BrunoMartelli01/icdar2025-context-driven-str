"""Export frozen shared ResNet features and qualify one FIT-only PCA projection."""

import argparse
import copy
import hashlib
from pathlib import Path
import time

import joblib
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch

from scripts.export_uniform_unet import checked_training, model
from scripts.run_icdar13_dense_instances import runtime
from scripts.run_uniform_unet_training import read, write
from segmentation.icdar2015_improvements import make_transform
from segmentation.uniform_unet_fusion import file_hash


def encoder_hash(state):
    h = hashlib.sha256()
    for key in sorted(k for k in state if k.startswith('encoder.')):
        h.update(key.encode()); h.update(state[key].cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def population(root):
    records = {}; plans = {}
    for ds in ['icdar13', 'icdar2015', 'totaltext']:
        plan, _ = checked_training(root/'full'/ds); plans[ds] = plan
        for r in plan['train_records']+plan['test_records']:
            if r['sample_id'] in records: raise ValueError('Repeated context sample ID')
            records[r['sample_id']] = dict(r, dataset=ds)
    protected = {records[s]['rgb_sha256'] for p in plans.values() for role in ['selection','test'] for s in p['partitions'][role]}
    fit = sorted({s for p in plans.values() for s in p['partitions']['fit'] if records[s]['rgb_sha256'] not in protected})
    unique = {}; fit = [unique.setdefault(records[s]['rgb_sha256'], s) for s in fit]
    return records, plans, sorted(set(fit)), protected


def export(args):
    records, plans, fit, _ = population(args.root); out = args.output
    out.mkdir(parents=True, exist_ok=False)
    plan = plans['icdar13']; c = plan['config']; runtime(c)
    net = model(plan, args.root/'full/icdar13'); expected = encoder_hash(net.state_dict())
    transform = make_transform(c, c['image_hw'], augment=False)
    ids = sorted(records); vectors = []; torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
    for sid in ids:
        r = records[sid]
        if file_hash(r['image_path']) != r['image_sha256']: raise ValueError('Changed context RGB')
        with Image.open(r['image_path']) as image:
            rgb = image.convert('RGB'); x, _ = transform(rgb, Image.new('L', rgb.size, 0), 0)
        with torch.inference_mode():
            e4 = net._encode(x[None].to(c['device']))[-1]
            v = torch.cat((e4.mean((2,3)), e4.amax((2,3))), dim=1)[0].cpu().numpy()
        vectors.append(v)
    matrix = np.stack(vectors); assert matrix.shape == (len(ids),4096) and np.isfinite(matrix).all()
    np.save(out/'features.npy', matrix, allow_pickle=False)
    index = {s:dict(row=i,rgb_sha256=records[s]['rgb_sha256'],image_sha256=records[s]['image_sha256']) for i,s in enumerate(ids)}
    write(out/'index.json', index)
    projection = make_pipeline(StandardScaler(), PCA(n_components=args.components, svd_solver='full'))
    projection.fit(matrix[[index[s]['row'] for s in fit]].astype(np.float64)); joblib.dump(projection,out/'projection.joblib')
    write(out/'frozen.json',dict(status='FROZEN',mode='approximate',components=args.components,fit_ids=fit,
        encoder_sha256=expected,features_sha256=file_hash(out/'features.npy'),index_sha256=file_hash(out/'index.json'),
        projection_sha256=file_hash(out/'projection.joblib'),images=len(ids),ground_truth_used_for_features=False,
        pca_fit_excludes_all_domain_validation_and_test_rgb=True,source_sha256=file_hash(__file__),
        device=c['device'],precision='float32',batch_size=1,input_hw=c['image_hw'],seconds=time.monotonic()-started,
        peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
        upstream_encoder_exposure=True,extra_runtime_neural_models=0))
    print('ENCODER CONTEXT FROZEN',len(ids),'images;',len(fit),'unique pooled FIT images',flush=True)


def audit(args):
    records, plans, fit, protected = population(args.root); out=args.output; frozen=read(out/'frozen.json')
    assert frozen['status']=='FROZEN' and frozen['fit_ids']==fit and frozen['components']==args.components
    for name,field in [('features.npy','features_sha256'),('index.json','index_sha256'),('projection.joblib','projection_sha256')]:
        assert file_hash(out/name)==frozen[field]
    index=read(out/'index.json'); matrix=np.load(out/'features.npy',allow_pickle=False)
    assert set(index)==set(records) and matrix.shape==(len(records),4096) and np.isfinite(matrix).all()
    assert not {records[s]['rgb_sha256'] for s in fit}&protected
    for ds in plans:
        state=torch.load(args.root/'full'/ds/'best.pt',map_location='cpu',weights_only=False)
        assert encoder_hash(state['model_state_dict'])==frozen['encoder_sha256']; del state
    rebuilt=make_pipeline(StandardScaler(),PCA(n_components=args.components,svd_solver='full'))
    xx=matrix[[index[s]['row'] for s in fit]].astype(np.float64); rebuilt.fit(xx)
    saved=joblib.load(out/'projection.joblib')
    np.testing.assert_allclose(rebuilt.transform(xx),saved.transform(xx),rtol=0,atol=1e-10)
    write(out/'independent_audit.json',dict(status='PASS',images=len(records),pooled_fit_images=len(fit),
        frozen_sha256=file_hash(out/'frozen.json'),encoder_identical_all3=True,projection_refit=True,
        protected_rgb_excluded=True,source_sha256=file_hash(__file__)))
    print('SHARED ENCODER CONTEXT AUDIT PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--components',type=int,required=True);p.add_argument('--stage',choices=['export','audit'],required=True);a=p.parse_args()
    with threadpool_limits(limits=2):
        (export if a.stage=='export' else audit)(a)
