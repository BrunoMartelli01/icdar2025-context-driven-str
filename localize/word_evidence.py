"""GT-free fragment association and crop proposals on frozen Text+Kernel maps."""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from segmentation.totaltext_word_kernel import instance_proposals


def candidates(probabilities, glyph_probability, labels, canvas, config):
    """Inference inputs only: no GT, owner IDs, validity or transcriptions."""
    p, g, labels = np.asarray(probabilities), np.asarray(glyph_probability), np.asarray(labels)
    if p.shape != (2, *labels.shape) or g.shape != labels.shape or canvas.shape != labels.shape:
        raise ValueError('Evidence map geometry mismatch')
    if (not np.isfinite(p).all() or not np.isfinite(g).all() or np.any(p < 0)
            or np.any(p > 1) or np.any(g < 0) or np.any(g > 1) or np.any(labels < 0)):
        raise ValueError('Invalid evidence probabilities or labels')
    props = instance_proposals(labels)
    n, fragments = cv2.connectedComponents(((g >= config['glyph_threshold']) & canvas).astype('uint8'), connectivity=8)
    size = labels.shape[0]
    fragment_rows = []
    for fid in range(1, n):
        yy, xx = np.where(fragments == fid)
        fragment_rows.append({'id': fid, 'bbox': [int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1],
                              'center': [float(xx.mean()),float(yy.mean())], 'area': len(xx)})
    pair_features, pair_ids, distances = [], [], {}
    boxes, box_features = [], []
    for prop in props:
        sid = prop['instance_id']; support = labels == sid
        x1,y1,x2,y2 = prop['bbox_native']; w,h = x2-x1,y2-y1
        seed = support & (p[1] >= config['kernel_threshold'])
        # Unseeded fallback instances remain explicit; association cannot invent a seed.
        distances[sid] = distance_transform_edt(~seed) if seed.any() else None
        box_features.append([x1/size,y1/size,x2/size,y2/size,w/size,h/size,
            float(support.sum())/(w*h),float(p[0][support].mean()),float(p[0][support].std()),
            float(p[1][support].mean()),float(g[support].mean()),float(seed.sum())/max(1,int(support.sum()))])
        boxes.append([x1,y1,x2,y2])
    for fragment in fragment_rows:
        fid=fragment['id']; fx1,fy1,fx2,fy2=fragment['bbox']; fw,fh=fx2-fx1,fy2-fy1
        cx,cy=fragment['center']; support=fragments==fid
        nearest=sorted(props,key=lambda r: (((r['bbox_native'][0]+r['bbox_native'][2])/2-cx)**2+
                       ((r['bbox_native'][1]+r['bbox_native'][3])/2-cy)**2,r['instance_id']))[:config['nearest_instances']]
        for prop in nearest:
            sid=prop['instance_id']; d=distances[sid]
            if d is None: continue
            x1,y1,x2,y2=prop['bbox_native']; w,h=x2-x1,y2-y1
            gapx=max(x1-fx2,fx1-x2,0); gapy=max(y1-fy2,fy1-y2,0)
            overlap=int(np.count_nonzero(support & (labels==sid)))/fragment['area']
            pair_features.append([gapx/max(w,fw),gapy/max(h,fh),(cx-(x1+x2)/2)/w,(cy-(y1+y2)/2)/h,
                fw/w,fh/h,overlap,float(d[support].min())/max(w,h),
                float(p[0][support].mean()),float(p[1][support].mean()),float(g[support].mean()),
                fragment['area']/(w*h)])
            pair_ids.append([fid,sid])
    return {'pair_features':np.asarray(pair_features,np.float32).reshape(-1,12),
            'pair_ids':np.asarray(pair_ids,np.int32).reshape(-1,2),'fragments':fragments,
            'box_features':np.asarray(box_features,np.float32).reshape(-1,12),
            'boxes':np.asarray(boxes,np.float32).reshape(-1,4),
            'instance_ids':np.asarray([r['instance_id'] for r in props],np.int32)}


def associate(labels, fragments, pair_ids, scores, threshold):
    """Attach each fragment once; preserve all existing IDs and occupied pixels."""
    if len(pair_ids)!=len(scores) or not np.isfinite(scores).all():
        raise ValueError('Invalid association scores')
    result=labels.copy(); attachments=conflicts=0
    for fid in sorted(set(int(v) for v in pair_ids[:,0])):
        indexes=np.where(pair_ids[:,0]==fid)[0]
        best=sorted(indexes,key=lambda i:(-float(scores[i]),int(pair_ids[i,1])))[0]
        if scores[best] < threshold: continue
        sid=int(pair_ids[best,1]); support=fragments==fid
        if not np.any(labels==sid): raise ValueError('Association selected a missing instance')
        if np.any(support & (labels>0) & (labels!=sid)):
            conflicts+=1;continue
        new=support & (labels==0)
        if new.any(): result[new]=sid;attachments+=1
    if not np.array_equal(result[labels>0],labels[labels>0]):
        raise RuntimeError('Association changed preexisting instance ownership')
    return result,{'attached_fragments':attachments,'conflicting_fragments_preserved':conflicts}


def adjust_boxes(boxes, deltas, shape):
    boxes=np.asarray(boxes,np.float32);deltas=np.asarray(deltas,np.float32)
    if boxes.shape!=deltas.shape or not np.isfinite(deltas).all(): raise ValueError('Invalid box corrections')
    scale=np.maximum(boxes[:,2:]-boxes[:,:2],1)
    result=boxes+deltas*np.concatenate((scale,scale),axis=1)
    result[:,[0,2]]=np.clip(result[:,[0,2]],0,shape[1])
    result[:,[1,3]]=np.clip(result[:,[1,3]],0,shape[0])
    result=np.concatenate((np.floor(result[:,:2]),np.ceil(result[:,2:])),axis=1).astype(int)
    if np.any(result[:,2:]<=result[:,:2]):raise ValueError('Crop correction produced an empty box')
    return result
