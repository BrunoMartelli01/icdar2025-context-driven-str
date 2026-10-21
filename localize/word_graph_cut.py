"""APPROXIMATE, support-preserving two-word graph cut; fixed background.

No annotations or OCR enter candidate generation. Binary s-t cuts are exact for
the recorded integer-quantized Potts energy. The zero label outside support is
fixed, not a third optimized class. Keep instance labels when forming crops.
"""
from __future__ import annotations

from itertools import combinations
import math

import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow
from scipy.ndimage import distance_transform_edt

from localize.crop_recovery import geodesic_partition


def minimum_cut(unaries, edge_u, edge_v, weights, seeds, *, integer_scale):
    """Return labels1/2 and an auditable exact integer min-cut certificate."""
    d=np.asarray(unaries,dtype=np.float64)
    u=np.asarray(edge_u,dtype=np.int64);v=np.asarray(edge_v,dtype=np.int64)
    w=np.asarray(weights,dtype=np.float64);seeds=np.asarray(seeds)
    if (d.ndim!=2 or d.shape[1]!=2 or seeds.shape!=(len(d),)
        or u.shape!=v.shape or u.shape!=w.shape or not len(d)
        or not np.isfinite(d).all() or not np.isfinite(w).all()
        or (d<0).any() or (w<0).any() or not np.isin(seeds,[0,1,2]).all()
        or (u<0).any() or (v<0).any() or (u>=len(d)).any() or (v>=len(d)).any()
        or integer_scale<1 or not (seeds==1).any() or not (seeds==2).any()):
        raise ValueError('Invalid finite nonnegative binary-cut energy or seeds')
    unary=np.rint(d*integer_scale).astype(np.int64)
    pair=np.rint(w*integer_scale).astype(np.int64)
    # Greater than the sum of all finite capacities, with explicit int32 guard.
    hard=int(unary.sum()+2*pair.sum()+1)
    if hard>=np.iinfo(np.int32).max:
        raise OverflowError('Integer graph capacities exceed the qualified solver bound')
    unary[seeds==1,1]=hard;unary[seeds==2,0]=hard
    n=len(d);source=n;sink=n+1;indices=np.arange(n)
    rows=np.concatenate([u,v,np.full(n,source),indices])
    cols=np.concatenate([v,u,indices,np.full(n,sink)])
    caps=np.concatenate([pair,pair,unary[:,1],unary[:,0]])
    graph=csr_matrix((caps,(rows,cols)),shape=(n+2,n+2),dtype=np.int64)
    graph.eliminate_zeros();graph.sort_indices()
    flow=maximum_flow(graph,source,sink,method='dinic')
    residual=graph-flow.flow;residual.data=(residual.data>0).astype(np.int64)
    residual.eliminate_zeros()
    reachable=breadth_first_order(residual,source,directed=True,return_predecessors=False)
    label=np.full(n,2,np.uint8);label[reachable[reachable<n]]=1
    if not np.array_equal(label[seeds>0],seeds[seeds>0]):
        raise AssertionError('Hard seeds violated')
    energy=int(unary[np.arange(n),label-1].sum()+pair[label[u]!=label[v]].sum())
    if energy!=int(flow.flow_value):raise AssertionError('Max-flow/min-cut certificate differs')
    return label,{'integer_energy':energy,'max_flow':int(flow.flow_value),
        'integer_scale':integer_scale,'nodes':n,'undirected_edges':len(pair)}


def grid_edges(support, probability, rgb, config):
    support=np.asarray(support,bool);p=np.asarray(probability,np.float64)
    rgb=np.asarray(rgb,np.float64)/255.
    if p.shape!=support.shape or rgb.shape!=(*support.shape,3):
        raise ValueError('Graph geometry differs')
    if not np.isfinite(p).all() or not np.isfinite(rgb).all() or (p<0).any() or (p>1).any():
        raise ValueError('Invalid probability or RGB')
    h,w=support.shape;index=np.full((h,w),-1,np.int64)
    index[support]=np.arange(int(support.sum()))
    us=[];vs=[];ws=[];ps=[]
    for dy,dx in [(0,1),(1,-1),(1,0),(1,1)]:
        ay=slice(0,h-dy);by=slice(dy,h)
        ax=slice(max(0,-dx),w-max(0,dx));bx=slice(max(0,dx),w-max(0,-dx))
        good=support[ay,ax]&support[by,bx]
        if not good.any():continue
        q=np.minimum(p[ay,ax],p[by,bx])[good]
        contrast=np.sum((rgb[ay,ax]-rgb[by,bx])**2,axis=-1)[good]
        affinity=(config['edge_floor']+q**config['probability_power'])*np.exp(
            -contrast/(2*config['color_sigma']**2))/math.hypot(dx,dy)
        us.append(index[ay,ax][good]);vs.append(index[by,bx][good]);ws.append(affinity);ps.append(q)
    if not us:
        return index,np.empty(0,np.int64),np.empty(0,np.int64),np.empty(0),np.empty(0)
    return index,np.concatenate(us),np.concatenate(vs),np.concatenate(ws),np.concatenate(ps)


def partition(support, probability, rgb, seeds, energy, *, method):
    support=np.asarray(support,bool);seeds=np.asarray(seeds,np.int32)
    if support.shape!=seeds.shape or np.any((seeds>0)&~support) or not np.isin(seeds,[0,1,2]).all():
        raise ValueError('Invalid support or seed labels')
    if not (seeds==1).any() or not (seeds==2).any():raise ValueError('Two seed regions required')
    index,u,v,w,q=grid_edges(support,probability,rgb,energy)
    audit={}
    if method=='graphcut':
        # Distances are continuous costs; no GT or transcription is involved.
        distances=[distance_transform_edt(seeds!=k)[support] for k in [1,2]]
        den=np.maximum(distances[0]+distances[1],1.)
        unary=np.stack([distances[0]/den,distances[1]/den],axis=1)*energy['unary_weight']
        values,audit=minimum_cut(unary,u,v,w*energy['pairwise_weight'],seeds[support],
            integer_scale=energy['integer_scale'])
        labels=np.zeros(support.shape,np.int32);labels[support]=values
    elif method=='geodesic':
        labels=geodesic_partition(np.asarray(probability),support,seeds)
    else:raise ValueError('Unknown partition method')
    if not np.array_equal(labels>0,support) or not np.array_equal(labels[seeds>0],seeds[seeds>0]):
        raise AssertionError('Partition changed support or seeds')
    values=labels[support];cut=values[u]!=values[v]
    audit.update(cut_edges=int(cut.sum()),cut_probability=float(q[cut].mean()) if cut.any() else 1.,
        normalized_cut_affinity=float(w[cut].mean()/max(w.mean(),1e-12)) if cut.any() else 1.,
        label_hash=__import__('hashlib').sha256(labels.astype('<i4').tobytes()).hexdigest())
    return labels,audit


def seed_pairs(support, probability, config):
    """Shared bounded threshold/erosion seeds for graph cut and its control."""
    pairs=[];seen=set();area=int(support.sum())
    for kind,value in [('threshold',v) for v in config['seed_thresholds']]+[('erosion',v) for v in config['erosion_radii']]:
        if kind=='threshold':core=(probability>=value)&support
        else:
            kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*value+1,2*value+1))
            core=cv2.erode(support.astype(np.uint8),kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
        count,labels,stats,_=cv2.connectedComponentsWithStats(core.astype(np.uint8),connectivity=8)
        ids=[k for k in range(1,count) if stats[k,cv2.CC_STAT_AREA]>=config['minimum_seed_area']
            and stats[k,cv2.CC_STAT_AREA]>=config['minimum_seed_fraction']*area]
        ids.sort(key=lambda k:(-stats[k,cv2.CC_STAT_AREA],k))
        if not 2<=len(ids)<=config['maximum_seed_regions']:continue
        a,b=sorted(ids[:2]);seeds=np.zeros(support.shape,np.int32)
        seeds[labels==a]=1;seeds[labels==b]=2
        key=seeds.tobytes()
        if key in seen:continue
        seen.add(key);pairs.append((seeds,{'kind':kind,'value':value}))
    return pairs


def original_box(native, image_hw):
    h,w=image_hw;x0,y0,x1,y1=native
    return [max(0,math.floor(x0*w/128)),max(0,math.floor(y0*h/128)),
        min(w,math.ceil(x1*w/128)),min(h,math.ceil(y1*h/128))]


def make_proposal(box,image_hw,padding,**metadata):
    padded=[max(0,box[0]-padding),max(0,box[1]-padding),min(128,box[2]+padding),min(128,box[3]+padding)]
    return {'bbox_native':list(box),'bbox':original_box(box,image_hw),
        'padded_bbox_native':padded,'padded_bbox_original':original_box(padded,image_hw),**metadata}


def proposal_pool(probability, rgb, baseline, config):
    """Shared seeds on 2x native ROI using actual original-RGB pixels."""
    p=np.asarray(probability)
    if p.shape!=(128,128) or rgb.ndim!=3 or rgb.shape[2]!=3:raise ValueError('Invalid scene inputs')
    binary=(p>=config['probability_threshold']).astype(np.uint8)
    _,components=cv2.connectedComponents(binary,connectivity=8)
    groups=[];labels_for_audit={};scale=config['graph_grid_scale'];stats={'parents':len(baseline),'seed_pairs':0,'solves':0,'rejected_geometry':0}
    for parent_index,parent in enumerate(baseline):
        x0,y0,x1,y1=map(int,parent['bbox_native'])
        if min(x1-x0,y1-y0)<config['minimum_parent_short_side']:continue
        boxlabels=components[y0:y1,x0:x1]
        # Same external contour may surround holes; identify its foreground by
        # matching the actual global component bounding box, never by filling it.
        ids=[int(k) for k in np.unique(boxlabels) if k>0]
        matches=[]
        for k in ids:
            ys,xs=np.where(components==k)
            if [int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]==[x0,y0,x1,y1]:matches.append(k)
        if len(matches)!=1:raise ValueError('Parent bbox does not identify one native component')
        support=boxlabels==matches[0]
        pairs=seed_pairs(support,p[y0:y1,x0:x1],config);stats['seed_pairs']+=len(pairs)
        if not pairs:continue
        shape=((x1-x0)*scale,(y1-y0)*scale)
        fg=cv2.resize(support.astype(np.uint8),shape,interpolation=cv2.INTER_NEAREST)>0
        prob=cv2.resize(p[y0:y1,x0:x1],shape,interpolation=cv2.INTER_LINEAR)
        original=original_box([x0,y0,x1,y1],rgb.shape[:2]);a,b,c,d=original
        colors=cv2.resize(rgb[b:d,a:c],shape,interpolation=cv2.INTER_AREA)
        for seed_index,(small,origin) in enumerate(pairs):
            seeds=cv2.resize(small,shape,interpolation=cv2.INTER_NEAREST)
            for method in ['geodesic','graphcut']:
                energies=[config['energies'][0]] if method=='geodesic' else config['energies']
                for energy_index,energy in enumerate(energies):
                    labels,audit=partition(fg,prob,colors,seeds,energy,method=method);stats['solves']+=1
                    children=[];areas=[]
                    for k in [1,2]:
                        region=(labels==k);n=cv2.connectedComponents(region.astype(np.uint8),connectivity=8)[0]-1
                        ys,xs=np.where(region);area=int(region.sum());areas.append(area)
                        if n!=1 or area<config['minimum_child_area']*scale**2:break
                        box=[x0+float(xs.min())/scale,y0+float(ys.min())/scale,
                            x0+float(xs.max()+1)/scale,y0+float(ys.max()+1)/scale]
                        if min(box[2]-box[0],box[3]-box[1])<config['minimum_child_short_side']:break
                        children.append(make_proposal(box,rgb.shape[:2],config['padding_native'],
                            parent_index=parent_index,method=method))
                    if len(children)!=2:stats['rejected_geometry']+=1;continue
                    left,right=[c['bbox_native'] for c in children]
                    inter=max(0,min(left[2],right[2])-max(left[0],right[0]))*max(0,min(left[3],right[3])-max(left[1],right[1]))
                    aa=(left[2]-left[0])*(left[3]-left[1]);bb=(right[2]-right[0])*(right[3]-right[1])
                    overlap=inter/(aa+bb-inter)
                    if min(areas)/sum(areas)<config['minimum_child_fraction'] or overlap>config['maximum_child_bbox_iou']:
                        stats['rejected_geometry']+=1;continue
                    gid=f'p{parent_index}-s{seed_index}-{method}-e{energy_index}'
                    groups.append({'id':gid,'parent_index':parent_index,'method':method,
                        'energy_index':energy_index,'seed':origin,'children':children,'audit':audit,
                        'child_area_fraction':min(areas)/sum(areas),'bbox_iou':overlap})
                    labels_for_audit[gid]=labels
    return groups,labels_for_audit,stats


def select(baseline, groups, policy, cap):
    if policy['method']=='baseline':return list(baseline),[]
    options={}
    for g in groups:
        if g['method']!=policy['method']:continue
        if g['method']=='graphcut' and g['energy_index']!=policy['energy_index']:continue
        a=g['audit']
        if a['cut_probability']>policy['max_cut_probability'] or a['normalized_cut_affinity']>policy['max_cut_affinity']:continue
        key=(a['normalized_cut_affinity'],a['cut_probability'],-g['child_area_fraction'],g['id'])
        parent=g['parent_index']
        if parent not in options or key<options[parent][0]:options[parent]=(key,g)
    result=[];chosen=[]
    for i,parent in enumerate(baseline):
        if i in options:
            g=options[i][1];result.extend(g['children']);chosen.append(g['id'])
        else:result.append(parent)
    result.sort(key=lambda r:-(r['padded_bbox_native'][2]-r['padded_bbox_native'][0])*(r['padded_bbox_native'][3]-r['padded_bbox_native'][1]))
    return result[:cap],chosen
