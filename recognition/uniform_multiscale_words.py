"""Conservative observable word additions from an unchanged second-scale expert."""

import math


def iou(a,b):
    area=lambda z:max(0.,z[2]-z[0])*max(0.,z[3]-z[1])
    intersection=area([max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])])
    union=area(a)+area(b)-intersection
    return intersection/union if union else 0.


def augment(native,larger,*,minimum_sequence_probability,maximum_overlap,maximum_candidates):
    if (not math.isfinite(minimum_sequence_probability) or not 0<=minimum_sequence_probability<=1 or
        not math.isfinite(maximum_overlap) or not 0<maximum_overlap<=1 or
        type(maximum_candidates)is not int or maximum_candidates<len(native)):
        raise ValueError('Explicit valid augmentation policy preserving all native rows required')
    rows=[dict(r) for r in native]
    eligible=[]
    for i,row in enumerate(larger):
        p=row['sequence_confidence_native']
        if not math.isfinite(p) or not 0<=p<=1:raise ValueError('Finite sequence probability required')
        if p>=minimum_sequence_probability and any(c.isalnum() for c in row['raw_text']):
            eligible.append((i,row))
    eligible.sort(key=lambda pair:(-pair[1]['sequence_confidence_native'],tuple(pair[1]['bbox']),pair[1]['raw_text'],pair[0]))
    for i,row in eligible:
        if len(rows)==maximum_candidates:break
        if all(iou(row['bbox'],r['bbox'])<maximum_overlap for r in rows):
            rows.append(dict(row,multiscale_added=True,scale_source_row=i,source_row=len(rows)))
    return rows
