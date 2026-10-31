"""APPROXIMATE ownership-conditioned feature transport with full-resolution detail."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from models.adaspot_gap import AdaSpotGapAGUNet


def shifted(value,dy,dx):
    """Neighbor at (y+dy,x+dx), with an explicit no-wrap validity mask."""
    if value.ndim!=4:raise ValueError('Expected BCHW feature map')
    h,w=value.shape[-2:]
    if abs(dy)>=h or abs(dx)>=w:raise ValueError('Transport displacement exceeds feature map')
    padded=F.pad(value,(max(-dx,0),max(dx,0),max(-dy,0),max(dy,0)))
    y=max(dy,0);x=max(dx,0)
    output=padded[...,y:y+h,x:x+w]
    valid=torch.ones((1,1,h,w),device=value.device,dtype=value.dtype)
    if dy>0:valid[...,-dy:,:]=0
    elif dy<0:valid[...,:-dy,:]=0
    if dx>0:valid[...,:,-dx:]=0
    elif dx<0:valid[...,:,:-dx]=0
    return output,valid


class DirectionalTransport(nn.Module):
    """Shared low-rank edge gates at dilated displacements, inside the decoder.

    The matched global control has identical tensors and receives identical
    ownership supervision, but spatially averages its active edge gates.
    """
    def __init__(self,channels,width,radii,*,spatial):
        super().__init__()
        if width<4 or width%4 or not radii or min(radii)<1:raise ValueError('Invalid transport architecture')
        self.spatial=bool(spatial);self.radii=tuple(radii)
        self.project=nn.Sequential(nn.Conv2d(channels,width,1,bias=False),nn.GroupNorm(4,width),nn.GELU())
        self.edge=nn.Sequential(nn.Conv2d(3*width+2,width,1),nn.GELU(),nn.Conv2d(width,1,1))
        self.output=nn.Conv2d(width,channels,1,bias=False)
        self.gain=nn.Parameter(torch.zeros(len(radii)))
        self.directions=((-1,0),(1,0),(0,-1),(0,1))

    def forward(self,feature,*,return_edges=False):
        h=self.project(feature);edges=[];original=h
        for step,radius in enumerate(self.radii):
            total=h;normalizer=torch.ones_like(h[:,:1]);raw=[]
            for dy,dx in self.directions:
                neighbor,valid=shifted(h,dy*radius,dx*radius)
                geometry=h.new_empty((len(h),2,*h.shape[-2:]))
                geometry[:,0].fill_(dy*radius/h.shape[-2]);geometry[:,1].fill_(dx*radius/h.shape[-1])
                logits=self.edge(torch.cat((h,neighbor,(h-neighbor).abs(),geometry),1))
                gate=logits.sigmoid()
                if not self.spatial:gate=(gate*valid).sum((-2,-1),keepdim=True)/valid.sum().clamp_min(1)
                weight=gate*valid
                total=total+weight*neighbor;normalizer=normalizer+weight
                if return_edges:raw.append((logits,valid,dy*radius,dx*radius))
            context=total/normalizer
            h=h+torch.tanh(self.gain[step])*(context-h)
            if return_edges:edges.extend(raw)
        output=feature+self.output(h-original)
        return output,edges


class DirectionalTransportUNet(AdaSpotGapAGUNet):
    """Warm-start every U-Net tensor; new paths initialize to exact identity."""
    def __init__(self,*,skip_floor,initial_gap_bias,widths,radii,detail_width,spatial):
        super().__init__('control',skip_floor=skip_floor,initial_gap_bias=initial_gap_bias)
        self.source_frozen = False
        if len(widths)!=4 or detail_width<4 or detail_width%4:raise ValueError('Invalid new decoder widths')
        self.base_names=tuple(self.state_dict());self.spatial=bool(spatial)
        self.transport=nn.ModuleList(DirectionalTransport(c,d,radii,spatial=spatial)
            for c,d in zip((512,256,128,64),widths))
        self.rgb_detail=nn.Sequential(nn.Conv2d(3,detail_width,3,padding=1,bias=False),
            nn.GroupNorm(4,detail_width),nn.GELU(),nn.Conv2d(detail_width,detail_width,3,padding=1,bias=False),
            nn.GroupNorm(4,detail_width),nn.GELU())
        self.detail_fusion=nn.Sequential(nn.Conv2d(64+detail_width,detail_width,3,padding=1,bias=False),
            nn.GroupNorm(4,detail_width),nn.GELU(),nn.Conv2d(detail_width,2,1))
        nn.init.zeros_(self.detail_fusion[-1].weight);nn.init.zeros_(self.detail_fusion[-1].bias)

    def freeze_source(self):
        """Isolated v2 training scope: preserve source weights AND BN statistics."""
        self.source_frozen = True
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith(('transport.', 'rgb_detail.', 'detail_fusion.')))
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, 'source_frozen', False):
            for module in (self.bottleneck, self.up3, self.up2, self.up1, self.up0, self.head):
                module.eval()
        return self

    @property
    def architecture_identity(self):
        return 'approximate-directional-transport-unet-v1' if self.spatial else 'approximate-global-gate-detail-unet-v1'

    def initialize_source(self,state):
        if set(state)!=set(self.base_names):raise ValueError('Every original U-Net tensor, and no extra source tensor, is required')
        own=self.state_dict()
        for name,value in state.items():
            if own[name].shape!=value.shape or own[name].dtype!=value.dtype:raise ValueError('Source tensor mismatch: '+name)
            own[name]=value
        self.load_state_dict(own,strict=True)
        return dict(copied_tensors=len(state),new_tensors=sorted(set(own)-set(state)),strict_load=True)

    def forward_evidence(self,image,*,return_edges=False):
        if image.ndim!=4 or image.shape[1]!=3 or any(v%32 for v in image.shape[-2:]):raise ValueError('Qualified RGB dimensions divisible by32 required')
        e0,e1,e2,e3,e4=self._encode(image)
        feature=self.bottleneck(e4);edges=[]
        for up,skip,transport in zip((self.up3,self.up2,self.up1,self.up0),(e3,e2,e1,e0),self.transport):
            feature=up(feature,skip);feature,raw=transport(feature,return_edges=return_edges);edges.extend(raw)
        coarse=self.head(feature)
        high=F.interpolate(feature,image.shape[-2:],mode='nearest')
        logits=F.interpolate(coarse,image.shape[-2:],mode='nearest')+self.detail_fusion(torch.cat((high,self.rgb_detail(image)),1))
        return dict(words=logits,coarse=coarse,features=feature,edges=edges)

    def forward(self,image):return self.forward_evidence(image)['words']
