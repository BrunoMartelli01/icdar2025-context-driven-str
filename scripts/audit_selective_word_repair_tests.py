"""Independent SciPy replay of selective repair counts, pixels and invariants."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from recognition.scene_text_datasets import parse_totaltext_annotation


def read(path):return json.loads(path.read_text())


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def overlap(a,b):
    x=max(0,min(a[2],b[2])-max(a[0],b[0]));y=max(0,min(a[3],b[3])-max(a[1],b[1]))
    intersection=x*y
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union else 0.


def independently_measure(predictions,words):
    matrix=np.zeros((len(predictions),len(words)),dtype=np.int8)
    for i,p in enumerate(predictions):
        for j,w in enumerate(words):
            matrix[i,j]=p['text'].casefold()==w['text'].casefold() and overlap(p['bbox'],w['bbox'])>=.5
    spatial=int((maximum_bipartite_matching(csr_matrix(matrix),perm_type='column')>=0).sum()) if matrix.size else 0
    expected=Counter(w['text'].casefold() for w in words)
    emitted=Counter(t for p in predictions for t in (p['text'].casefold().split() or ['']))
    lexical=sum(min(expected[k],v) for k,v in emitted.items())
    return dict(spatial=dict(correct=spatial,predictions=len(predictions),gt_words=len(words)),
                lexical=dict(correct=lexical,predictions=sum(emitted.values()),gt_words=len(words)))


def run(root):
    config=read(root/'resolved_configuration.json');plan=read(root/'protocol.json');result=read(root/'results.json')
    checked_inputs=0
    for name,digest in {**config['input_sha256'],**plan['input_exports_sha256']}.items():
        if sha(Path(name))!=digest:raise AssertionError('Input changed: '+name)
        checked_inputs+=1
    data=read(root/'samples.json');words={};pixel_checks=0;route_checks=0;boundary_checks=0
    for phase,samples in data.items():
        words[phase]={}
        variants=read(root/'reread'/(phase+'_variants.json'))
        routes=read(root/'selectors'/(phase+'_reread_routes.json'))
        for sample in samples:
            sid=sample['sample_id'];path=Path(sample['annotation_path'])
            assert sha(path)==sample['annotation_sha256']
            h,w=sample['image_hw']
            words[phase][sid]=[dict(text=x.transcription,bbox=[max(0,x.bbox_xyxy[0]),max(0,x.bbox_xyxy[1]),
                min(w,x.bbox_xyxy[2]+1),min(h,x.bbox_xyxy[3]+1)]) for x in parse_totaltext_annotation(path) if not x.ignore]
            assert len({p['crop_array_sha256'] for p in sample['pool']})<=config['budget']
            assert len(routes[sid])<=config['triage']['maximum_reread_parents'];route_checks+=1
            image=Path(sample['image_path']);assert sha(image)==sample['image_sha256']
            with Image.open(image) as im:rgb=np.asarray(im.convert('RGB'))
            for variant in variants[sid]:
                for child in variant['children']:
                    b=child['bbox'];crop=np.ascontiguousarray(rgb[b[1]:b[3],b[0]:b[2]])
                    hsh=hashlib.sha256(np.asarray(crop.shape,dtype='<i8').tobytes());hsh.update(crop.tobytes())
                    assert hsh.hexdigest()==child['crop_array_sha256'];pixel_checks+=1
    checks=0;aggregate_checks=0
    for path in sorted(root.rglob('*_results.json')):
        prefix=path.name.split('_',1)[0]
        if prefix not in words:continue
        predictions_path=path.with_name(path.name.replace('_results.json','_predictions.json'))
        if not predictions_path.exists():raise AssertionError('Missing predictions: '+str(path))
        record=read(path);predictions=read(predictions_path)
        by={r['sample_id']:r for r in record['per_image']}
        assert set(by)==set(predictions)==set(words[prefix])
        sums={key:Counter() for key in record['aggregate']}
        for sid,pp in predictions.items():
            measured=independently_measure(pp,words[prefix][sid])
            for metric in record['aggregate']:
                for key,value in measured[metric].items():
                    assert by[sid][metric][key]==value,(str(path),sid,metric,key)
                    sums[metric][key]+=value
            checks+=1
        for metric,counts in sums.items():
            for key,value in counts.items():assert record['aggregate'][metric][key]==value
            assert abs(record['aggregate'][metric]['f1']-2*counts['correct']/(counts['predictions']+counts['gt_words']))<1e-12
            aggregate_checks+=1
        if path.parent.name=='boundaries' and ('gap_head_' in path.name):
            samples={s['sample_id']:s for s in data[prefix]}
            for sid,pp in predictions.items():
                original={b['id']:b['text'] for b in samples[sid]['text_blocks']}
                assert len(original)==len(pp)
                for p in pp:
                    assert ''.join(p['text'].split())==''.join(original[p['id']].split())
                    boundary_checks+=1
        if path.name.endswith('_additions_results.json') and path.parent.name=='selectors':
            samples={s['sample_id']:s for s in data[prefix]}
            for sid,pp in predictions.items():
                baseline=samples[sid]['baseline_filter']
                assert pp[:len(baseline)]==baseline
    if not result['smoke']:
        assert len(data['fit'])==1030 and len(data['calibration'])==100
        assert len(data['evaluation'])==125 and len(data['test'])==300
        assert sum(len(v) for v in words['test'].values())==2204
    for memory in [*result['memory']['boundaries'],result['memory']['reread']]:
        assert memory['oom'] is False and memory['precision']=='float32'
        assert memory['peak_allocated_bytes']>0 and memory['peak_reserved_bytes']>=memory['peak_allocated_bytes']
    audit=dict(status='PASS',image_arm_checks=checks,aggregate_checks=aggregate_checks,
        final_rgb_crop_checks=pixel_checks,routing_budget_checks=route_checks,
        exact_unicode_copy_checks=boundary_checks,input_hash_checks=checked_inputs,
        result_sha256=sha(root/'results.json'),audit_code_sha256=sha(Path(__file__)),
        scope='Independent SciPy maximum matching and stdlib lexical replay; original RGB pixel hashes; locked-prefix and Unicode invariants.')
    (root/'independent_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',required=True,type=Path)
    run(parser.parse_args().output_root.resolve())
