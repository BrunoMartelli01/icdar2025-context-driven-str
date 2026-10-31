"""Frozen PARSeq cross-attention alignment for APPROXIMATE evidence experiments."""
from __future__ import annotations

import numpy as np
from PIL import Image
import torch


def character_centers(attention, grid, box):
    attention=np.asarray(attention,np.float32)
    gh,gw=map(int,grid)
    if attention.ndim!=2 or attention.shape[1]!=gh*gw or not np.isfinite(attention).all():
        raise ValueError('PARSeq patch alignment shape/values differ')
    mass=attention.sum(1,keepdims=True)
    if np.any(mass<=0):raise ValueError('Empty cross-attention')
    weights=attention/mass
    yy,xx=np.mgrid[:gh,:gw];a,b,d,e=box
    centers=np.stack((a+(xx.ravel()+.5)/gw*(d-a), b+(yy.ravel()+.5)/gh*(e-b)),1)
    entropy=-(weights*np.log(weights.clip(1e-12))).sum(1)/np.log(max(2,gh*gw))
    return weights@centers,entropy


def align_batch(recognizer, crops, texts, boxes):
    """Teacher-force the already observed strings; never produce replacement OCR.

    The last decoder layer already returns averaged cross-attention weights in
    the pinned official runtime. The read-only hook changes neither its inputs
    nor outputs. Native cached probabilities remain separate from this causal
    teacher-forcing alignment proxy.
    """
    if not (len(crops)==len(texts)==len(boxes)):raise ValueError('Alignment batch differs')
    system=recognizer.model;model=system.model;tokenizer=system.tokenizer
    if system.training or any(p.requires_grad for p in system.parameters()):
        raise ValueError('Alignment requires a fully frozen eval recognizer')
    if any(not s or len(s)>model.max_label_length for s in texts):
        raise ValueError('Unsupported alignment string length')
    observed=[]
    def capture(module,arguments,output):
        if output[1] is None:raise ValueError('Pinned decoder did not expose cross-attention')
        observed.append(output[1].detach().cpu().numpy())
    handle=model.decoder.layers[-1].cross_attn.register_forward_hook(capture)
    try:
        tensor=torch.stack([recognizer.transform(Image.fromarray(crop)) for crop in crops]).to(recognizer.config.device)
        with torch.inference_mode():
            encoded=model.encode(tensor);tokens=tokenizer.encode(texts,device=tensor.device)
            context=tokens[:,:-1];length=context.shape[1]
            mask=torch.triu(torch.ones(length,length,device=tensor.device,dtype=torch.bool),diagonal=1)
            model.decode(context,encoded,tgt_mask=mask,tgt_query_mask=mask,
                         tgt_padding_mask=context==tokenizer.pad_id)
        if len(observed)!=1:raise ValueError('Unexpected final-layer alignment calls')
        grid=tuple(model.encoder.patch_embed.grid_size)
        rows=[]
        for j,(text,box) in enumerate(zip(texts,boxes)):
            centers,entropy=character_centers(observed[0][j,:len(text)],grid,box)
            rows.append(dict(centers=centers.tolist(),entropy=entropy.tolist(),grid=list(grid),
                alignment='last-layer causal teacher-forcing cross-attention centroid; approximate'))
        return rows
    finally:handle.remove()
