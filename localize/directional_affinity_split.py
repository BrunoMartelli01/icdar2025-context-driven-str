"""Conservative graph splitting from learned ownership, preserving every pixel."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from localize.crop_recovery import geodesic_partition
from localize.directional_closedness import neighbor


def affinity_split(labels, edges, canvas, *, threshold, minimum_cells,
                   minimum_fraction, minimum_coverage):
    if threshold is None:
        return labels.copy(),dict(split_instances=0,added_instances=0)
    if not 0 < threshold <= 1 or minimum_cells < 1 or not 0 < minimum_fraction < .5 or not 0 < minimum_coverage <= 1:
        raise ValueError('Invalid affinity split controls')
    shape=edges[0][0].shape
    if (canvas.shape!=shape or labels.ndim!=2 or labels.shape[0]%shape[0]
            or labels.shape[1]%shape[1] or not np.issubdtype(labels.dtype,np.integer)
            or np.any(labels<0)):
        raise ValueError('Aligned integer instance labels and affinities required')
    sy,sx=labels.shape[0]//shape[0],labels.shape[1]//shape[1]
    coarse=labels[::sy,::sx];inside=(coarse>0)&canvas
    if not inside.any():return labels.copy(),dict(split_instances=0,added_instances=0)
    probabilities={};validity={}
    for probability,border,dy,dx in edges:
        if probability.shape!=shape or not np.isfinite(probability).all() or np.any((probability<0)|(probability>1)):
            raise ValueError('Invalid ownership affinity')
        probabilities[(dy,dx)]=probability
        validity[(dy,dx)]=border.astype(bool)&canvas&neighbor(canvas,dy,dx)
    mutual={}
    for (dy,dx),p in probabilities.items():
        if (-dy,-dx) not in probabilities:raise ValueError('Paired opposite ownership directions required')
        mutual[(dy,dx)]=np.where(validity[(dy,dx)],np.minimum(p,neighbor(probabilities[(-dy,-dx)],dy,dx)),0.)
    # Diagonal reliability is the strongest of two two-step axial paths.
    diagonal=[]
    for dy,dx in ((1,1),(1,-1)):
        horizontal,vertical=mutual[(0,dx)],mutual[(dy,0)]
        value=np.maximum(np.minimum(horizontal,neighbor(vertical,0,dx)),
                         np.minimum(vertical,neighbor(horizontal,dy,0)))
        diagonal.append((dy,dx,value))
    candidates=[(dy,dx,p) for (dy,dx),p in mutual.items() if dy>0 or (dy==0 and dx>0)] + diagonal
    index=np.full(shape,-1,np.int32);index[inside]=np.arange(int(inside.sum()))
    rows=[];cols=[];weights=[]
    for dy,dx,p in candidates:
        other=neighbor(coarse,dy,dx)
        eligible=inside&neighbor(inside,dy,dx)&(coarse==other)
        rows.append(index[eligible]);cols.append(neighbor(index,dy,dx)[eligible]);weights.append(p[eligible])
    row,col,weight=map(np.concatenate,(rows,cols,weights));n=int(inside.sum())
    def clusters(keep):
        graph=csr_matrix((np.ones(int(keep.sum()),np.uint8),(row[keep],col[keep])),shape=(n,n))
        _,value=connected_components(graph,directed=False)
        field=np.full(shape,-1,np.int32);field[inside]=value
        return field
    geometric=clusters(np.ones(len(row),bool))
    learned=clusters(weight>=threshold)
    result=labels.copy();next_id=int(labels.max())+1;split=added=0
    for identity in np.unique(coarse[inside]):
        support=(coarse==identity)&inside
        # Never call a subsampling-only disconnection a learned split.
        if len(np.unique(geometric[support]))!=1:continue
        ids,counts=np.unique(learned[support],return_counts=True)
        cutoff=max(minimum_cells,minimum_fraction*int(support.sum()))
        selected=[int(i) for i,c in zip(ids,counts) if c>=cutoff]
        if len(selected)<2 or sum(int(c) for i,c in zip(ids,counts) if int(i) in selected)<minimum_coverage*int(support.sum()):continue
        markers=np.zeros(shape,np.int32)
        for cluster in selected:
            markers[support&(learned==cluster)]=next_id;next_id+=1
        fine=np.repeat(np.repeat(markers,sy,axis=0),sx,axis=1)
        original=labels==identity;fine[~original]=0
        partition=geodesic_partition(np.ones(labels.shape,np.float32),original,fine)
        if np.any(original&(partition==0)):raise RuntimeError('Affinity split lost foreground support')
        result[original]=partition[original];split+=1;added+=len(selected)-1
    if not np.array_equal(result>0,labels>0):raise RuntimeError('Affinity splitting changed foreground pixels')
    return result,dict(split_instances=split,added_instances=added)
