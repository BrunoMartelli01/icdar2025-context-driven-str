"""Small CPU-trained modules; all segmentation weights and evaluation maps frozen."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def majority_owner(support, owners, minimum_fraction):
    selected=owners[support]
    if not len(selected):return -1
    positive=selected[selected>0]
    if not len(positive):return 0
    ids,counts=np.unique(positive,return_counts=True);best=int(np.argmax(counts))
    return int(ids[best]) if counts[best]/len(selected)>=minimum_fraction else -1


def supervision(candidate, labels, owners, validity, config):
    """GT is consumed only here, after inference features are frozen."""
    owner=owners.copy();owner[~validity]=0
    instance_owner={int(sid):majority_owner((labels==sid) & validity,owner,config['owner_majority']) for sid in candidate['instance_ids']}
    fragment_owner={int(fid):majority_owner((candidate['fragments']==fid) & validity,owner,config['owner_majority'])
                    for fid in np.unique(candidate['fragments']) if fid>0}
    targets=[];keep=[]
    for fid,sid in candidate['pair_ids']:
        a,b=fragment_owner[int(fid)],instance_owner[int(sid)]
        keep.append(a!=-1 and b!=-1)
        targets.append(float(a>0 and a==b))
    reference_boxes=[];box_targets=[];box_keep=[];neighbors=[]
    for sid,box in zip(candidate['instance_ids'],candidate['boxes']):
        oid=instance_owner[int(sid)];yy,xx=np.where(owner==oid) if oid>0 else ([],[])
        eligible=len(xx)>0
        truth=[int(np.min(xx)),int(np.min(yy)),int(np.max(xx))+1,int(np.max(yy))+1] if eligible else box.tolist()
        scale=np.maximum(box[2:]-box[:2],1)
        box_targets.append((np.asarray(truth)-box)/np.concatenate((scale,scale)))
        reference_boxes.append(truth);box_keep.append(eligible)
        others=[]
        for other in np.unique(owner):
            if other<=0 or other==oid:continue
            oy,ox=np.where(owner==other)
            others.append([int(ox.min()),int(oy.min()),int(ox.max())+1,int(oy.max())+1])
        neighbors.append(others)
    return {'pair_target':np.asarray(targets,np.float32),'pair_keep':np.asarray(keep,bool),
            'box_target':np.asarray(box_targets,np.float32).reshape(-1,4),'box_keep':np.asarray(box_keep,bool),
            'reference_boxes':np.asarray(reference_boxes,np.float32).reshape(-1,4),'neighbor_boxes':neighbors}


class EvidenceHead(nn.Module):
    def __init__(self,outputs,hidden):
        super().__init__();self.layers=nn.Sequential(nn.Linear(12,hidden),nn.ReLU(),nn.Linear(hidden,outputs))
        nn.init.zeros_(self.layers[-1].weight);nn.init.zeros_(self.layers[-1].bias)
    def forward(self,x):return self.layers(x)


def crop_loss(prediction, boxes, truth, neighbors, max_delta, config):
    delta=torch.tanh(prediction)*max_delta
    size=(boxes[:,2:]-boxes[:,:2]).clamp_min(1)
    corrected=boxes+delta*torch.cat((size,size),dim=1)
    intersection=(torch.minimum(corrected[:,2:],truth[:,2:])-torch.maximum(corrected[:,:2],truth[:,:2])).clamp_min(0).prod(1)
    true_area=(truth[:,2:]-truth[:,:2]).prod(1).clamp_min(1)
    predicted_area=(corrected[:,2:]-corrected[:,:2]).prod(1).clamp_min(1)
    coverage=(1-intersection/true_area).mean()
    excess=((predicted_area-intersection)/predicted_area).mean()
    normalized=(truth-boxes)/torch.cat((size,size),dim=1)
    regression=F.smooth_l1_loss(delta,normalized)
    if isinstance(neighbors,list):
        indexes=[];flat=[]
        for i,other in enumerate(neighbors):
            indexes.extend([i]*len(other));flat.extend(other)
        neighbors={'indexes':torch.tensor(indexes,dtype=torch.long),
                   'boxes':torch.tensor(flat,dtype=corrected.dtype).reshape(-1,4)}
    idx=neighbors['indexes'];other=neighbors['boxes']
    if len(idx):
        overlap=(torch.minimum(corrected[idx,2:],other[:,2:])-torch.maximum(corrected[idx,:2],other[:,:2])).clamp_min(0).prod(1)
        reference_overlap=(torch.minimum(truth[idx,2:],other[:,2:])-torch.maximum(truth[idx,:2],other[:,:2])).clamp_min(0).prod(1)
        foreign=((overlap-reference_overlap).clamp_min(0)/predicted_area[idx]).sum()/len(corrected)
    else:foreign=corrected.sum()*0
    return (config['regression']*regression+config['coverage']*coverage+config['excess']*excess+
            config['foreign']*foreign)


def fit_head(features,targets,config,*,kind,boxes=None,truth=None,neighbors=None):
    if len(features)==0 or not np.isfinite(features).all():raise ValueError('Missing/nonfinite training features')
    x=torch.as_tensor(features,dtype=torch.float32); y=torch.as_tensor(targets,dtype=torch.float32)
    mean=x.mean(0);scale=x.std(0,unbiased=False).clamp_min(1e-4);x=(x-mean)/scale
    torch.manual_seed(config['seed']);model=EvidenceHead(1 if kind=='affinity' else 4,config['hidden'])
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    if kind=='affinity' and (y.sum()==0 or y.sum()==len(y)):raise ValueError('Association requires positive AND negative training pairs')
    if kind=='crop':
        indexes=[];flat=[]
        for i,other in enumerate(neighbors):
            indexes.extend([i]*len(other));flat.extend(other)
        neighbors={'indexes':torch.tensor(indexes,dtype=torch.long),
                   'boxes':torch.tensor(flat,dtype=torch.float32).reshape(-1,4)}
    best_loss=float('inf');best=None;history=[]
    for epoch in range(config['epochs']):
        optimizer.zero_grad(set_to_none=True);prediction=model(x)
        if kind=='affinity':
            logits=prediction[:,0];positive=y>0
            loss=(F.binary_cross_entropy_with_logits(logits[positive],y[positive])+
                  F.binary_cross_entropy_with_logits(logits[~positive],y[~positive]))/2
        else:
            loss=crop_loss(prediction,torch.as_tensor(boxes),torch.as_tensor(truth),neighbors,
                           config['max_box_delta'],config['crop_loss'])
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite head loss')
        value=float(loss.detach());history.append({'epoch':epoch,'loss':value})
        # Save the state that actually produced this measured loss, before the update.
        if value<best_loss:best_loss=value;best=copy.deepcopy(model.state_dict())
        loss.backward();optimizer.step()
    model.load_state_dict(best,strict=True);model.eval()
    return {'kind':kind,'mean':mean.tolist(),'scale':scale.tolist(),
            'state':{k:v.tolist() for k,v in model.state_dict().items()},'history':history,
            'selected_epoch':min(history,key=lambda r:r['loss'])['epoch'],'selection':'minimum TRAIN loss; no holdout tuning',
            'training_rows':len(features),'hidden':config['hidden'],'max_box_delta':config['max_box_delta']}


def predict_head(receipt,features):
    if not len(features):return np.zeros((0,) if receipt['kind']=='affinity' else (0,4),np.float32)
    model=EvidenceHead(1 if receipt['kind']=='affinity' else 4,receipt['hidden'])
    model.load_state_dict({k:torch.tensor(v) for k,v in receipt['state'].items()},strict=True)
    with torch.inference_mode():
        x=(torch.as_tensor(features)-torch.tensor(receipt['mean']))/torch.tensor(receipt['scale'])
        raw=model(x)
        return (raw[:,0].sigmoid() if receipt['kind']=='affinity' else raw.tanh()*receipt['max_box_delta']).numpy()
