"""Post-hoc TRAIN crop clipping, using original GT glyph colors; no inference."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image

from diagnostics.directional_neighbors_train import annotation_word_colors
from diagnostics.gt_mask_localization_oracle import save_json, sha, verify_file
from segmentation.data import load_segmentation_manifest


def color_code(color):
    r,g,b = map(int,color)
    return (r << 16) | (g << 8) | b


def packed_rgb(rgb):
    v = np.asarray(rgb,dtype=np.uint32)
    return (v[:,:,0] << 16) | (v[:,:,1] << 8) | v[:,:,2]


def color_counts(packed):
    codes,counts=np.unique(packed,return_counts=True)
    return {int(c):int(n) for c,n in zip(codes,counts)}


def crop_glyph_retention(packed, colors, box, totals):
    if box is None: selected={}
    else:
        x1,y1,x2,y2=box
        selected=color_counts(packed[y1:y2,x1:x2])
    rows=[]
    for color in colors:
        code=color_code(color);total=totals.get(code,0)
        if not total: raise ValueError('annotated glyph color absent from target')
        kept=selected.get(code,0)
        rows.append({'color':list(color),'pixels_total':total,'pixels_in_crop':kept,'retained_fraction':kept/total})
    return rows,selected


def summarize(rows, limit):
    return {'words':len(rows),
        'any_glyph_pixels_outside_crop':sum(r['minimum_glyph_fraction']<1 for r in rows),
        'at_least_one_glyph_completely_outside_crop':sum(r['minimum_glyph_fraction']==0 for r in rows),
        'at_least_one_glyph_retained_fraction_le_limit':sum(r['minimum_glyph_fraction']<=limit for r in rows),
        'all_glyph_pixels_inside_crop':sum(r['minimum_glyph_fraction']==1 for r in rows),
        'crop_includes_dont_care_pixels':sum(r['dont_care_pixels_in_crop']>0 for r in rows)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--expected-sha256',required=True)
    p.add_argument('--glyph-retention-limit',type=float,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if not 0<a.glyph_retention_limit<1:raise ValueError('descriptive limit must be in (0,1)')
    verify_file(a.audit,a.expected_sha256);audit=json.loads(a.audit.read_text())
    if audit['protocol']['scope']!={'split':'TRAIN','images':229,'gt_words':849}:raise ValueError('TRAIN only')
    oracle_path=Path(audit['protocol']['oracle_path']);verify_file(oracle_path,audit['protocol']['oracle_sha256'])
    oracle=json.loads(oracle_path.read_text());paths=oracle['protocol']['source_paths']
    verify_file(paths['source_report'],audit['protocol']['source_report_sha256']);source=json.loads(Path(paths['source_report']).read_text())
    verify_file(paths['train_manifest'],oracle['protocol']['train_manifest_sha256'])
    manifest=load_segmentation_manifest(paths['train_manifest'],require_checksums=False)
    if manifest.split_role!='train' or len(manifest.samples)!=229:raise ValueError('TRAIN only')
    root=Path(paths['data_root']);result=[]
    for index,sample in enumerate(manifest.samples,1):
        if index==1 or index%50==0 or index==229:print(f'[crop glyph audit CPU] {index}/229',flush=True)
        for rel in (sample.mask_path,sample.validity_annotation_path):verify_file(root/rel,source['protocol']['dataset_files'][str(rel)]['sha256'])
        with Image.open(root/sample.mask_path) as opened:packed=packed_rgb(opened.convert('RGB'))
        totals=color_counts(packed);annotation=(root/sample.validity_annotation_path).read_text(encoding='utf-8-sig')
        colors=annotation_word_colors(annotation)
        dc={color_code(list(map(int,line.lstrip()[1:].split()[:3]))) for line in annotation.splitlines() if line.lstrip().startswith('#')}
        words=[r for r in audit['per_word'] if r['sample_id']==sample.sample_id]
        if len(words)!=len(colors):raise RuntimeError('word/color grouping mismatch')
        for row,word_colors in zip(words,colors):
            glyphs,selected=crop_glyph_retention(packed,word_colors,row['crop_bbox'] if row['available'] else None,totals)
            result.append({'sample_id':sample.sample_id,'gt_index':row['gt_index'],'gt_text':row['gt_text'],
                'parseq_text':row['parseq_text'],'exact':row['exact'],'crop_failure':row['crop_failure'],
                'best_iou':row['best_iou'],'gt_coverage_bbox':row['gt_coverage'],
                'glyphs':glyphs,'minimum_glyph_fraction':min(g['retained_fraction'] for g in glyphs),
                'care_foreground_fraction_in_crop':sum(g['pixels_in_crop'] for g in glyphs)/sum(g['pixels_total'] for g in glyphs),
                'dont_care_pixels_in_crop':sum(selected.get(code,0) for code in dc)})
    if len(result)!=849:raise RuntimeError('word population changed')
    report={'classification':'APPROXIMATE_DIAGNOSTIC','scope':audit['protocol']['scope'],
        'audit_sha256':sha(a.audit),'code_sha256':sha(__file__),'glyph_retention_limit':a.glyph_retention_limit,
        'interpretation':'Original GT glyph pixel inclusion only; no OCR GT_BOX control. Clipping is evidence of crop incompleteness, not proof of sole error cause. Limit is post-hoc descriptive, not inference tuning.',
        'summary':{c:summarize([r for r in result if r['crop_failure']==c],a.glyph_retention_limit) for c in sorted({r['crop_failure'] for r in result})},
        'per_word':result}
    save_json(a.output,report);print(json.dumps(report['summary'],indent=2));print('JSON',a.output,'SHA256',sha(a.output))


if __name__=='__main__':main()
