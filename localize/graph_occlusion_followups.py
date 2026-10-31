"""Isolated APPROXIMATE fragmentation augmentation and contour-point GCN."""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def predicted_strip_mask(boxes, shape, generator, *, fraction, maximum_words):
    """GT-blind cuts of predicted word boxes, used only for TRAIN augmentation."""
    if len(shape) != 2 or not 0 < fraction < .5 or maximum_words < 1:
        raise ValueError('Invalid fragmentation contract')
    mask = np.zeros(shape, bool)
    order = torch.randperm(len(boxes), generator=generator).tolist()[:maximum_words]
    for j in order:
        x0,y0,x1,y1 = map(int,boxes[j]); h,w = y1-y0,x1-x0
        if not 0 <= x0 < x1 <= shape[1] or not 0 <= y0 < y1 <= shape[0]:
            raise ValueError('Predicted box outside native raster')
        if max(h,w) < 4: continue
        if w >= h:
            size=max(1,round(w*fraction)); left=x0+(w-size)//2
            mask[y0:y1,left:left+size]=True
        else:
            size=max(1,round(h*fraction)); top=y0+(h-size)//2
            mask[top:top+size,x0:x1]=True
    return mask


def apply_native_occlusion(image, mask):
    """Nearest native->RGB mapping; normalized-zero fill, clean input preserved."""
    if image.ndim != 4 or image.shape[0] != 1 or mask.ndim != 2:
        raise ValueError('Expected one aligned normalized RGB canvas')
    value=torch.as_tensor(mask,device=image.device,dtype=image.dtype)[None,None]
    value=F.interpolate(value,size=image.shape[-2:],mode='nearest').bool()
    return image.masked_fill(value,0.)


def occlude_original_rgb(rgb,mask,fill):
    """Inverse square-letterbox nearest mask; same augmented RGB supplies features."""
    if rgb.ndim!=3 or rgb.shape[2]!=3 or len(fill)!=3 or any(v<0 or v>255 for v in fill):
        raise ValueError('Explicit RGB occluder required')
    h,w=rgb.shape[:2];side=max(h,w);ox=(side-w)//2;oy=(side-h)//2
    full=cv2.resize(mask.astype(np.uint8),(side,side),interpolation=cv2.INTER_NEAREST).astype(bool)
    result=rgb.copy();result[full[oy:oy+h,ox:ox+w]]=np.asarray(fill,np.uint8)
    return result


def positive_edges(graph):
    a=graph['assignments'];e=graph['edges'];v=graph['eligible']
    return int(((a[e[:,0]]==a[e[:,1]]) & (a[e[:,0]]>0) & v[e[:,0]] & v[e[:,1]]).sum())


def use_enriched_graph(clean, augmented, generator, probability):
    """TRAIN-only variant sampling; exactly one update per original image."""
    return positive_edges(augmented)>positive_edges(clean) and float(torch.rand((),generator=generator))<probability


def rectangle_points(boxes, count):
    """Clockwise pixel-center rectangle with all four extrema; exclusive boxes."""
    if count < 8 or count % 4: raise ValueError('Contour size must be divisible by four')
    result=[];t=np.arange(count//4,dtype=np.float32)/(count//4)
    for x0,y0,x1,y1 in boxes:
        if x1<=x0 or y1<=y0: raise ValueError('Empty source box')
        a=np.array([x0+.5,y0+.5],np.float32);b=np.array([x1-.5,y0+.5],np.float32)
        d=np.array([x0+.5,y1-.5],np.float32);c=np.array([x1-.5,y1-.5],np.float32)
        result.append(np.concatenate([u[None]+t[:,None]*(v-u)[None] for u,v in ((a,b),(b,c),(c,d),(d,a))]))
    return np.asarray(result,np.float32).reshape(len(boxes),count,2)


def resample_boundary(mask, count):
    """Largest external TRAIN owner contour; pixel centers, uniform arc length.

    Other components/holes are explicitly counted by the caller. This is the
    repository's filled-word target, not an official polygon or amodal label.
    """
    contours,_=cv2.findContours(np.asarray(mask,np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not contours: return None,0
    contour=max(contours,key=lambda p:(cv2.contourArea(p),len(p)))[:,0].astype(np.float32)+.5
    if len(contour)<3:return None,len(contours)
    area=np.sum(contour[:,0]*np.roll(contour[:,1],-1)-contour[:,1]*np.roll(contour[:,0],-1))
    if area<0:contour=contour[::-1].copy()
    next_point=np.roll(contour,-1,axis=0);length=np.linalg.norm(next_point-contour,axis=1)
    if length.sum()<=0:return None,len(contours)
    cumulative=np.r_[0.,np.cumsum(length)];distance=np.arange(count)*cumulative[-1]/count
    index=np.minimum(np.searchsorted(cumulative,distance,side='right')-1,len(contour)-1)
    ratio=(distance-cumulative[index])/np.maximum(length[index],1e-8)
    return (contour[index]+ratio[:,None]*(next_point[index]-contour[index])).astype(np.float32),len(contours)


def ring_adjacency(count):
    """Four cyclic neighbors plus self, symmetric degree normalization."""
    result=torch.eye(count)
    for shift in (-2,-1,1,2):result+=torch.roll(torch.eye(count),shift,1)
    return result/result.sum(1,keepdim=True)


class ContourPointRefiner(nn.Module):
    """Iterative contour-point GCN and equal-parameter self-only control.

    CPU bilinear feature sampling retains gradients to current control points.
    Frozen CNN maps are not differentiable model parameters. This avoids a
    nondeterministic CUDA grid-sampler backward without detaching coordinates.
    """
    def __init__(self, channels, hidden, layers, *, message_passing, iterations, maximum_step):
        super().__init__()
        if iterations<1 or not 0<maximum_step<.5:raise ValueError('Invalid contour evolution')
        self.message_passing=message_passing;self.iterations=iterations;self.maximum_step=maximum_step
        self.project=nn.Linear(channels+6,hidden)
        self.layers=nn.ModuleList(nn.Linear(2*hidden,hidden) for _ in range(layers))
        self.offset=nn.Linear(hidden,2)
        nn.init.zeros_(self.offset.weight);nn.init.zeros_(self.offset.bias)

    def forward(self, feature_map, initial, boxes, native_size):
        if feature_map.device.type!='cpu' or feature_map.ndim!=4 or feature_map.shape[0]!=1:
            raise ValueError('One CPU feature canvas required')
        if initial.ndim!=3 or initial.shape[-1]!=2 or boxes.shape!=(len(initial),4):
            raise ValueError('Unaligned control points and source boxes')
        n,count,_=initial.shape
        if not n:return initial
        size=(boxes[:,2:]-boxes[:,:2]).clamp_min(1.)[:,None]
        adjacency=ring_adjacency(count).to(initial)
        points=initial
        for _ in range(self.iterations):
            grid=(points.to('cpu')/native_size*2-1).reshape(1,n*count,1,2)
            samples=F.grid_sample(feature_map,grid,mode='bilinear',padding_mode='border',align_corners=False)
            samples=samples[0,:,:,0].T.reshape(n,count,-1).to(initial.device)
            context=torch.cat((samples,points/native_size,(points-boxes[:,:2,None].transpose(1,2))/size,
                               size.expand(-1,count,-1)/native_size),-1)
            h=F.relu(self.project(context))
            for layer in self.layers:
                neighbors=adjacency@h if self.message_passing else h
                h=h+F.relu(layer(torch.cat((h,neighbors),-1)))
            points=points+torch.tanh(self.offset(h))*size*self.maximum_step
            points=points.clamp(.5,native_size-.5)
        return points


def cyclic_contour_loss(prediction, target, valid, boxes, *, beta):
    if prediction.shape!=target.shape or valid.shape!=(len(prediction),):raise ValueError('Contour targets differ')
    if not valid.any():return prediction.sum()*0
    scale=(boxes[:,2:]-boxes[:,:2]).clamp_min(1)[:,None]
    a,b=prediction[valid]/scale[valid],target[valid]/scale[valid]
    candidates=torch.stack([F.smooth_l1_loss(a,b.roll(j,1),reduction='none',beta=beta).mean((1,2)) for j in range(a.shape[1])],1)
    return candidates.min(1).values.mean()


def contour_boxes(points, initial, original, native_size):
    """Exact identity replay; exclusive native boxes from pixel-center contours."""
    if not np.isfinite(points).all():raise ValueError('Nonfinite predicted contour')
    output=[]
    for p,q,box in zip(points,initial,original):
        if np.array_equal(p,q):output.append(list(map(int,box)));continue
        lower=np.floor(p.min(0)-.5+1e-5).astype(int);upper=np.ceil(p.max(0)+.5-1e-5).astype(int)
        lower=np.clip(lower,0,native_size-1);upper=np.minimum(native_size,np.maximum(upper,lower+1))
        output.append([int(lower[0]),int(lower[1]),int(upper[0]),int(upper[1])])
    return output
