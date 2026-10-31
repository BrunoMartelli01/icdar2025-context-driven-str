"""GT-blind APPROXIMATE proposals and policies for the four occlusion tests.

SS-MSER-inspired is an adaptation, not an exact reproduction of Das et al.
All geometry uses exclusive original-RGB coordinates. No label is an input.
"""
from __future__ import annotations

from collections import Counter
import math
import cv2
import numpy as np

from localize.crop_recovery import geodesic_partition
from localize.mask_crop_hypotheses import select_groups
from localize.word_multitask import parseq_word_proposals
from segmentation.totaltext_image_frequency_augment import remap_letterbox_native_box


def decode_instance_labels(words, gap, canvas, postprocessing, threshold, decoder):
    """Normalize the Torch RGB-canvas boundary before NumPy instance decoding."""
    from localize.adaspot_gap_preserve import decode_gap
    return decode_gap(words, gap, np.asarray(canvas, dtype=bool), postprocessing, threshold, decoder)


def bind_baseline(item):
    """Rebind serialized baseline copies to their recognized pool identities."""
    lookup = {r["id"]: r for r in item["pool"]}
    if len(lookup) != len(item["pool"]):raise ValueError("Duplicate candidate IDs")
    item["baseline"] = [lookup[r["id"]] for r in item["baseline"]]
    if any("crop_array_sha256" not in r for r in item["baseline"]):
        raise ValueError("Baseline crop was not recognized")


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    area = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/area if area else 0.


def checked_box(box, hw):
    h, w = hw
    if len(box) != 4 or not all(math.isfinite(v) for v in box):
        raise ValueError("Nonfinite/malformed box")
    x0, y0, x1, y1 = map(int, box)
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError("Box outside RGB raster")
    return [x0, y0, x1, y1]


def row(box, hw, identity, method, parents=()):
    box = checked_box(box, hw)
    return {"id": identity, "bbox": box, "padded_bbox_original": box.copy(),
            "method": method, "parents": list(parents)}


def instance_pool(probability, labels, rgb, c):
    """Preserve all decoded instances; add bounded, conflicting split/merge choices."""
    size = labels.shape[0]
    if probability.shape != labels.shape or labels.shape != (size, size):
        raise ValueError("Aligned square maps required")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Invalid probability")
    hw = rgb.shape[:2]
    base = []
    for p in parseq_word_proposals(labels, hw[::-1], padding=0, min_area=0, max_instances=None):
        r = row(p["bbox"], hw, f"base:{p['instance_id']}", "baseline")
        r.update(bbox_native=p["bbox_native"], instance_id=p["instance_id"])
        base.append(r)
    pool = list(base); groups = []; seen = set()

    def add(boxes, parents, method):
        signature = (tuple(parents), tuple(map(tuple, boxes)))
        if signature in seen or len(pool)+len(boxes) > len(base)+c["maximum_extra"]:
            return
        seen.add(signature); gid = f"group:{len(groups)}"; children = []
        for j, box in enumerate(boxes):
            mapped = remap_letterbox_native_box(box, hw[::-1], native_size=size)
            if mapped[0] >= mapped[2] or mapped[1] >= mapped[3]:
                return
        for j, box in enumerate(boxes):
            identity = f"{gid}:{j}"
            r = row(remap_letterbox_native_box(box, hw[::-1], native_size=size), hw, identity, method, parents)
            r["bbox_native"] = box; pool.append(r); children.append(identity)
        groups.append({"id": gid, "parents": list(parents), "children": children, "method": method})

    for parent in base:
        support = labels == parent["instance_id"]
        x0, y0, x1, y1 = parent["bbox_native"]
        patch = probability[y0:y1, x0:x1]; support = support[y0:y1, x0:x1]
        for threshold in c["seed_thresholds"]:
            n, markers = cv2.connectedComponents(((patch >= threshold) & support).astype(np.uint8), connectivity=8)
            counts = np.bincount(markers.ravel(), minlength=n)
            ids = [i for i in range(1, n) if counts[i] >= c["minimum_seed_area"]]
            if not 2 <= len(ids) <= c["maximum_pieces"]:
                continue
            seeds = np.zeros_like(markers)
            for j, identity in enumerate(ids, 1): seeds[markers == identity] = j
            partition = geodesic_partition(patch, support, seeds)
            boxes = []
            for j in range(1, len(ids)+1):
                yy, xx = np.nonzero(partition == j)
                if len(xx) < c["minimum_piece_area"]: break
                boxes.append([int(xx.min())+x0, int(yy.min())+y0,
                              int(xx.max())+x0+1, int(yy.max())+y0+1])
            if len(boxes) == len(ids): add(boxes, [parent["id"]], "split")
    for index, a in enumerate(base):
        neighbors = []
        x0,y0,x1,y1 = a["bbox_native"]; ah = y1-y0
        for b in base[index+1:]:
            u0,v0,u1,v1 = b["bbox_native"]; bh = v1-v0
            overlap = max(0, min(y1,v1)-max(y0,v0))/min(ah,bh)
            gap = max(0, max(x0,u0)-min(x1,u1))
            if overlap >= c["merge_vertical_overlap"] and gap <= c["merge_gap_height"]*min(ah,bh):
                neighbors.append((gap, b["id"], b))
        for _, _, b in sorted(neighbors)[:c["maximum_merge_neighbors"]]:
            u0,v0,u1,v1 = b["bbox_native"]
            add([[min(x0,u0), min(y0,v0), max(x1,u1), max(y1,v1)]], [a["id"],b["id"]], "merge")
    return {"baseline": base, "pool": pool, "groups": groups}


def mser_pools(rgb, c):
    """Shared grouping on standard/filtered MSER. Filtering is a stated assumption.

    Stability compares rectangles across delta values, without exact-centroid
    matching. This is intentionally not the paper's underspecified soft-set sum.
    """
    h, w = rgb.shape[:2]; scale = min(1., c["maximum_side"]/max(h,w))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if scale < 1: gray = cv2.resize(gray, (max(1, round(w*scale)), max(1, round(h*scale))))
    area = gray.size; by_delta = []
    for delta in c["deltas"]:
        detector = cv2.MSER_create(delta, c["minimum_area"], max(c["minimum_area"]+1, int(area*c["maximum_area_fraction"])))
        regions, boxes = detector.detectRegions(gray); values = []
        for points, (x,y,bw,bh) in zip(regions, boxes):
            if bh < c["minimum_height"] or not c["aspect_range"][0] <= bw/bh <= c["aspect_range"][1]: continue
            fill = len(points)/(bw*bh)
            if not c["fill_range"][0] <= fill <= c["fill_range"][1]: continue
            values.append({"box": [int(x),int(y),int(x+bw),int(y+bh)], "area": len(points),
                           "fill": fill, "tone": float(gray[points[:,1],points[:,0]].mean())/255})
        # Deterministic capacity; no OCR/labels select components.
        values.sort(key=lambda r: (-r["area"], r["box"]))
        by_delta.append(values[:c["maximum_components"]])
    reference = by_delta[c["reference_delta_index"]]
    for r in reference:
        matches = [max((iou(r["box"], q["box"]) for q in other), default=0.)
                   for index, other in enumerate(by_delta) if index != c["reference_delta_index"]]
        r["stability"] = float(np.mean(matches))

    def grouped(components, name):
        boxes = {}
        def save(members):
            a = [min(r["box"][0] for r in members), min(r["box"][1] for r in members),
                 max(r["box"][2] for r in members), max(r["box"][3] for r in members)]
            box = [max(0, math.floor(a[0]/scale)), max(0, math.floor(a[1]/scale)),
                   min(w, math.ceil(a[2]/scale)), min(h, math.ceil(a[3]/scale))]
            priority = len(members)*float(np.mean([r["stability"] for r in members]))
            key = tuple(box)
            if key not in boxes or priority > boxes[key][0]: boxes[key] = (priority, len(members))
        for r in components: save([r])
        # Oriented-neighbor links permit sloped lines, but not arbitrary curve fitting.
        adjacency = [set() for _ in components]
        for i,a in enumerate(components):
            x0,y0,x1,y1 = a["box"]; ah=y1-y0
            for j in range(i+1,len(components)):
                b=components[j];u0,v0,u1,v1=b["box"];bh=v1-v0
                dx=abs((x0+x1-u0-u1)/2);dy=abs((y0+y1-v0-v1)/2)
                gap=max(0,max(x0,u0)-min(x1,u1))
                if (max(ah,bh)/min(ah,bh) <= c["height_ratio"] and dx > c["minimum_center_dx"]*min(ah,bh)
                        and dy <= c["maximum_center_dy"]*max(ah,bh) and gap <= c["maximum_gap_height"]*min(ah,bh)
                        and abs(a["tone"]-b["tone"]) <= c["maximum_tone_difference"]):
                    adjacency[i].add(j);adjacency[j].add(i)
        remaining=set(range(len(components)))
        while remaining:
            initial=min(remaining);remaining.remove(initial);queue=[initial];members=[]
            while queue:
                current=queue.pop();members.append(current)
                for neighbor in sorted(adjacency[current] & remaining):
                    remaining.remove(neighbor);queue.append(neighbor)
            ordered=sorted(members,key=lambda i: (components[i]["box"][0],components[i]["box"][1]))
            # Contiguous bounded groups retain alternatives to overmerged whole lines.
            for start in range(len(ordered)):
                for count in c["group_lengths"]:
                    if start+count <= len(ordered): save([components[i] for i in ordered[start:start+count]])
            if len(ordered)>1:save([components[i] for i in ordered])
        values=sorted(boxes.items(),key=lambda item:(-item[1][0],item[0]))[:c["maximum_word_candidates"]]
        return [dict(row(list(box), (h,w), f"{name}:{i}", name), component_count=n, stability=priority/max(n,1))
                for i,(box,(priority,n)) in enumerate(values)]
    return {"mser":grouped(reference,"mser"),
            "ss_mser_inspired":grouped([r for r in reference if r["stability"] >= c["minimum_stability"]],"ss_mser_inspired")}


def features(r, rgb, probability=None, *, text_threshold=None):
    """No OCR is used in the first part: suitable for genuine pre-OCR ranking."""
    h,w=rgb.shape[:2]; x0,y0,x1,y1=checked_box(r["padded_bbox_original"],(h,w));bw=x1-x0;bh=y1-y0
    patch=rgb[y0:y1,x0:x1];small=cv2.resize(patch,(24,12),interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(small,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    mean=small.mean((0,1))/255;std=small.std((0,1))/255
    values=[x0/w,y0/h,bw/w,bh/h,np.log1p(bw/bh),np.log1p(bw*bh)/20,
            *mean,*std,float(np.abs(np.diff(gray,axis=1)).mean()),float(np.abs(np.diff(gray,axis=0)).mean()),
            r.get("component_count",1)/16,r.get("stability",0),len(r["parents"])/4,
            *(float(r["method"]==m) for m in ("baseline","split","merge","mser","ss_mser_inspired"))]
    if probability is not None:
        if text_threshold is None or not 0 < text_threshold < 1:raise ValueError("Explicit feature support threshold required")
        side=max(h,w);size=probability.shape[0];ox=(side-w)//2;oy=(side-h)//2
        u0=max(0,math.floor((x0+ox)*size/side));v0=max(0,math.floor((y0+oy)*size/side))
        u1=min(size,max(u0+1,math.ceil((x1+ox)*size/side)));v1=min(size,max(v0+1,math.ceil((y1+oy)*size/side)))
        p=probability[v0:v1,u0:u1];values.extend([float(p.mean()),float(p.std()),float((p>=text_threshold).mean())])
    else: values.extend([-1.,-1.,-1.])
    return list(map(float,values))


def ocr_features(reading):
    logs=reading.get("token_log_probabilities",[])
    return [len(reading["raw_text"])/25,float(reading["sequence_confidence_native"]),
            float(np.mean(logs)) if logs else -20.,min(reading.get("token_probabilities",[]),default=0.)]


def expand_box(box, hw, fraction):
    if not math.isfinite(fraction) or fraction < 0:raise ValueError("Invalid expansion")
    x0,y0,x1,y1=checked_box(box,hw);h,w=hw
    padding=math.ceil((y1-y0)*fraction)
    return [max(0,x0-padding),max(0,y0-padding),min(w,x1+padding),min(h,y1+padding)]


def same_reading_nms(rows, readings, scores, threshold):
    kept=[]
    for r in sorted(rows,key=lambda r:(-scores[r["id"]],r["id"])):
        text=readings[r["crop_array_sha256"]]["raw_text"].casefold()
        if any(text==readings[q["crop_array_sha256"]]["raw_text"].casefold() and iou(r["bbox"],q["bbox"])>=threshold for q in kept):continue
        kept.append(r)
    return kept


def correction_choice(trace, readings, minimum_votes, confidence_gain, context_weight):
    """Exact ties and insufficient evidence KEEP the original. No GT-dependent gate."""
    original=trace["views"][0];anchor=readings[original["crop_array_sha256"]]["raw_text"]
    votes=Counter(readings[v["crop_array_sha256"]]["raw_text"] for v in trace["views"])
    confidence={}
    for v in trace["views"]:
        o=readings[v["crop_array_sha256"]];t=o["raw_text"]
        logs=o.get("token_log_probabilities",[])
        confidence[t]=max(confidence.get(t,0.),math.exp(float(np.mean(logs))) if logs else 0.)
    context=trace.get("context",{});anchor_context=context.get(anchor)
    best=anchor;gain_best=confidence_gain
    for text in sorted(votes):
        if text==anchor or not text or votes[text]<minimum_votes:continue
        delta=confidence[text]-confidence[anchor]
        if context_weight:
            if anchor_context is None or context.get(text) is None:continue
            delta+=context_weight*(context[text]-anchor_context)
        if delta>gain_best:best=text;gain_best=delta
    return best
