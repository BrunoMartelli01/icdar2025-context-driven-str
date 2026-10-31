"""Qualified linear projection of already-computed frozen encoder features."""

from pathlib import Path
import joblib
import numpy as np

from scripts.run_uniform_unet_training import read
from segmentation.uniform_unet_fusion import file_hash


def projection(config):
    if 'encoder_context' not in config: return None
    root=Path(config['encoder_context'])
    for name in ['frozen.json','independent_audit.json','projection.joblib']:
        path=root/name
        if config['pinned_inputs'].get(str(path))!=file_hash(path):raise ValueError('Changed qualified encoder projection')
    frozen=read(root/'frozen.json');audit=read(root/'independent_audit.json')
    if audit['status']!='PASS' or audit['frozen_sha256']!=file_hash(root/'frozen.json') or frozen['ground_truth_used_for_features']:
        raise ValueError('Independently qualified annotation-free encoder context required')
    return joblib.load(root/'projection.joblib')


def project(model, vector):
    vector=np.asarray(vector,dtype=np.float64)
    if vector.shape!=(4096,) or not np.isfinite(vector).all():raise ValueError('Finite4096-D actual encoder vector required')
    return model.transform(vector[None])[0]


def raw_context(config, records):
    if 'encoder_context' not in config:return None
    root=Path(config['encoder_context'])
    for name in ['index.json','features.npy']:
        path=root/name
        if config['pinned_inputs'].get(str(path))!=file_hash(path):raise ValueError('Changed actual encoder feature bank')
    index=read(root/'index.json');matrix=np.load(root/'features.npy',allow_pickle=False)
    result={}
    for sid,r in records.items():
        if index[sid]['rgb_sha256']!=r['rgb_sha256']:raise ValueError('Encoder features refer to another RGB image')
        result[sid]=matrix[index[sid]['row']]
    return result


def training_context(config, records):
    model=projection(config)
    if model is None:return None
    return {sid:project(model,value) for sid,value in raw_context(config,records).items()}
