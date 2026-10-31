"""GT-blind components and OCR crops from approximate U-Net probability maps."""

import math

import cv2
import numpy as np

from segmentation.icdar2015_improvements import padding_geometry


def component_crop(rgb,labels,proposal,image_hw,method):
    if method not in {"axis0","axis2","oriented0","oriented2"}:
        raise ValueError("Unknown explicit OCR crop method")
    pad=int(method[-1]);h,w=rgb.shape[:2]
    if method.startswith("axis"):
        x1,y1,x2,y2=proposal["bbox"]
        return np.ascontiguousarray(rgb[max(0,y1-pad):min(h,y2+pad),max(0,x1-pad):min(w,x2+pad)])
    x1,y1,x2,y2=proposal["bbox_native"]
    yy,xx=np.nonzero(labels[y1:y2,x1:x2]==proposal["component_index"])
    if not len(xx):raise ValueError("Missing labeled component")
    cw,ch,left,top=padding_geometry((w,h),image_hw)
    sx=cw/(image_hw[1]//2);sy=ch/(image_hw[0]//2)
    points=np.column_stack(((xx+x1+.5)*sx-left,(yy+y1+.5)*sy-top)).astype(np.float32)
    (cx,cy),(rw,rh),angle=cv2.minAreaRect(points)
    # Include the finite footprint of native foreground pixels, rather than
    # treating pixel centers as geometric boundaries (including singleton CCs).
    rw+=sx;rh+=sy
    if rh>rw:rw,rh,angle=rh,rw,angle+90
    angle=(angle+90)%180-90
    u=np.array([math.cos(math.radians(angle)),math.sin(math.radians(angle))])
    v=np.array([-u[1],u[0]]);center=np.array([cx,cy])
    rw+=2*pad;rh+=2*pad
    quad=np.asarray([center-u*rw/2-v*rh/2,center+u*rw/2-v*rh/2,
        center+u*rw/2+v*rh/2,center-u*rw/2+v*rh/2],dtype=np.float32)
    width,height=max(2,int(math.ceil(rw))),max(2,int(math.ceil(rh)))
    target=np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],dtype=np.float32)
    matrix=cv2.getPerspectiveTransform(quad,target)
    return cv2.warpPerspective(rgb,matrix,(width,height),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)


def operating_selection_key(candidate):
    emitted=candidate["emitted_aggregate"]["alphanumeric_casefold"]
    return (emitted["f1"],candidate["raw_aggregate"]["alphanumeric_casefold"]["f1"],
        emitted["recall"],candidate["crop_method"].startswith("axis"),
        -abs(candidate["probability_threshold"]-.5),-int(candidate["crop_method"][-1]))
