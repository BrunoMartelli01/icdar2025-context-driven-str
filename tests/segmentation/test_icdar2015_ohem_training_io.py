"""Omitting unused owners must preserve used tensors/RNG and accumulated updates."""

import copy
import json
import random
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from segmentation.icdar2015_improvement_runtime import REPO
from segmentation.icdar2015_improvements import OwnerDataset,make_transform
from segmentation.icdar2015_target70 import TrainingSupervision
from segmentation.icdar2015_target70_ohem import TrainingPixels,training_config,train_epoch,ohem_loss


@pytest.mark.parametrize("epoch",[106,137])
def test_used_training_tensors_and_rng_are_identical_without_owner_replay(tmp_path,epoch):
    config=json.loads((REPO/"configs/approximate/icdar2015_target70_ohem_v1.json").read_text())
    config["training"].update(rotation_probability=1.,horizontal_flip_probability=1.)
    transform=make_transform(config,[64,96],augment=True);transform.set_epoch(epoch)
    pixels=np.arange(48*96*3,dtype=np.uint16).reshape(48,96,3).astype(np.uint8)
    image=Image.fromarray(pixels);image.save(tmp_path/"rgb.png")
    mask=np.zeros((48,96),np.uint8);mask[10:20,20:40]=255
    valid=np.full_like(mask,255);valid[25:35,60:75]=0
    owners=np.zeros_like(mask,dtype=np.uint16);owners[mask>0]=1
    Image.fromarray(owners).save(tmp_path/"owners.png")
    class Base:
        def __init__(self):
            self.transform=transform;self.data_root=tmp_path
            self.manifest=SimpleNamespace(samples=[SimpleNamespace(image_path='rgb.png',word_owner_annotation_path='owners.png')])
        def __len__(self):return 1
        def __getitem__(self,index):
            rgb,target,validity=transform.transform_with_validity(image,Image.fromarray(mask),Image.fromarray(valid),index)
            return {'sample_id':'fixture','image':rgb,'mask':target,'validity':validity,'separation':torch.zeros_like(target)}
    base=Base();torch_state=torch.get_rng_state().clone();python_state=random.getstate();numpy_state=np.random.get_state()
    old=TrainingSupervision(OwnerDataset(base),negative_ignore=True)[0]
    new=TrainingSupervision(TrainingPixels(base),negative_ignore=True)[0]
    for key in ('image','mask','validity','separation','ignored_negative'):assert torch.equal(old[key],new[key]),key
    assert 'owners' in old and 'owners' not in new
    assert torch.equal(torch.get_rng_state(),torch_state) and random.getstate()==python_state
    current=np.random.get_state();assert current[0]==numpy_state[0] and np.array_equal(current[1],numpy_state[1]) and current[2:]==numpy_state[2:]
    assert training_config(config,config['branches']['ohem_full']).loss=='masked_OHEM_BCEWithLogitsLoss'


def test_ohem_accumulation_updates_the_last_partial_window():
    torch.manual_seed(9);net=torch.nn.Conv2d(1,1,1,bias=False);reference=copy.deepcopy(net)
    batches=[]
    for i in range(3):
        image=torch.full((1,1,1,1),float(i+1))
        batches.append({'image':image,'sample_id':[str(i)],'mask':torch.ones_like(image),'validity':torch.ones_like(image),
            'separation':torch.zeros_like(image),'ignored_negative':torch.zeros_like(image)})
    config={'training':{'hard_negative_ratio':3,'separation_weight':3}}
    branch={'gradient_accumulation':2,'gradient_clip_norm':1.,'ignore_negative_weight':3}
    optimizer=torch.optim.SGD(net.parameters(),lr=.1);gold=torch.optim.SGD(reference.parameters(),lr=.1)
    result=train_epoch(net,batches,optimizer,config,branch,device='cpu')
    for window in (batches[:2],batches[2:]):
        gold.zero_grad();loss=sum(ohem_loss(reference(b['image']),b,negative_ratio=3,separation_weight=3,ignore_negative_weight=3)[0] for b in window)/len(window)
        loss.backward();torch.nn.utils.clip_grad_norm_(reference.parameters(),1.,error_if_nonfinite=True);gold.step()
    torch.testing.assert_close(net.weight,reference.weight,rtol=0,atol=0)
    assert (result['samples'],result['steps'],result['microbatches'])==(3,2,3)
