"""Drop optimizer state only from completed non-improving uniform pilot cases."""

import argparse
from pathlib import Path

import numpy as np
import torch

from scripts.run_uniform_unet_training import read,write
from segmentation.uniform_unet_fusion import file_hash


def equal(a,b):
    if type(a)!=type(b):return False
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def compact(root):
    assert read(root/'status.json')['status']=='PILOT_AUDITED'
    launch=read(root/'launch.json');process=Path('/proc')/str(launch['pid'])/'cmdline'
    if process.exists() and b'run_uniform_photometric_pilot' in process.read_bytes():raise ValueError('Pilot controller still live')
    receipt=root/'optimizer_storage_compaction.json'
    if receipt.exists():raise ValueError('Preserve previous compaction receipt')
    jobs=sorted(root.glob('*/smoke/icdar13'))+sorted(root.glob('*/full/icdar13'))
    if len(jobs)!=4:raise ValueError('Four completed matched pilot cases required')
    records=[]
    for folder in jobs:
        complete=read(folder/'complete.json');assert complete['selected_epoch']==0
        assert read(folder/'independent_training_audit.json')['status']=='PASS'
        assert file_hash(folder/'best.pt')==complete['checkpoint_sha256']
    for folder in jobs:
        path=folder/'last.pt';old=torch.load(path,map_location='cpu',weights_only=False)
        if 'optimizer_state_dict' not in old:raise ValueError('Optimizer state already absent')
        before=file_hash(path);size=path.stat().st_size
        new={k:v for k,v in old.items() if k!='optimizer_state_dict'}
        temp=folder/'last.compaction_tmp.pt'
        if temp.exists():raise ValueError('Preserve previous compaction temporary')
        torch.save(new,temp);got=torch.load(temp,map_location='cpu',weights_only=False)
        assert equal(new,got) and equal(old['model_state_dict'],got['model_state_dict'])
        record=dict(path=str(path),old_sha256=before,new_sha256=file_hash(temp),old_bytes=size,new_bytes=temp.stat().st_size,
            preserved_best_sha256=file_hash(folder/'best.pt'),preserved_tensor_count=len(got['model_state_dict']),
            all_nonoptimizer_state_exact=True,optimizer_continuation_available=False)
        write(receipt,dict(status='IN_PROGRESS',completed=records,pending=record))
        temp.replace(path);records.append(record);del old,new,got
    write(receipt,dict(status='COMPLETE',cases=records,reclaimed_bytes=sum(r['old_bytes']-r['new_bytes'] for r in records),
        source_sha256=file_hash(__file__),scope='Final model, RNG, history and metadata preserved exactly. Completed non-improving cases only; optimizer continuation state explicitly removed. Selected checkpoints untouched.'))
    print('Compacted4 completed optimizer states; recovered GiB',sum(r['old_bytes']-r['new_bytes'] for r in records)/2**30,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();compact(a.root)
