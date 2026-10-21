"""Match word readings across detector resolutions without labels."""
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from localize.independent_word_slots import overlap_iou
from recognition.group_word_selection import normalize
from recognition.yolo_word_refinement import joint_score,nms


def scale_pairs(base,high,min_iou):
    if not 0<min_iou<=1:raise ValueError('invalid pair IoU')
    if not base or not high:return []
    weights=np.full((len(base),len(high)+len(base)),-np.inf)
    for i,a in enumerate(base):
        weights[i,len(high)+i]=0
        for j,b in enumerate(high):
            overlap=overlap_iou(a['bbox'],b['bbox'])
            if normalize(a['raw_text']) and normalize(a['raw_text'])==normalize(b['raw_text']) and overlap>=min_iou:
                weights[i,j]=math.sqrt(joint_score(a)*joint_score(b))
    x,y=linear_sum_assignment(weights,maximize=True)
    return [{**base[i],'agreement_score':float(weights[i,j]),
             'paired_candidate_id':high[j]['candidate_id'],'pair_iou':overlap_iou(base[i]['bbox'],high[j]['bbox'])}
            for i,j in zip(x,y) if j<len(high)]


def agreement_controls(base,high,cfg):
    pairs=scale_pairs(base,high,cfg['pair_iou'])
    return {f'agreement_{t}':nms([r for r in pairs if r['agreement_score']>=t],
                               cfg['text_nms_iou'],same_text=True,joint=True)
            for t in cfg['joint_controls']}
