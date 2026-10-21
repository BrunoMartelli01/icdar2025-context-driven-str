"""Independent SciPy matching and frozen-OCR audit of the cap-only diagnostic."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from scripts.audit_totaltext_compact_v2_selected_ocr import independent_metrics
from segmentation.data import file_sha256


def audit(directory):
    p=json.loads((directory/'protocol.json').read_text());r=json.loads((directory/'results.json').read_text());c=p['plan']
    assert r['status']=='SUCCESS' and r['protocol_sha256']==file_sha256(directory/'protocol.json')
    for name,digest in p['code_sha256'].items():assert file_sha256(Path(name))==digest,name
    source=Path(c['source_directory'])
    for name,digest in c['source_sha256'].items():assert file_sha256(source/name)==digest
    gt=json.loads((directory/'ground_truth.json').read_text())['ground_truth']
    assert set(gt)==set(p['sample_ids']) and sum(map(len,gt.values()))==r['care_words']
    ocr={json.loads(line)['crop_array_sha256']:json.loads(line) for line in (directory/'ocr.jsonl').read_text().splitlines()}
    reused=0
    for line in (source/'ocr.jsonl').read_text().splitlines():
        old=json.loads(line);digest=old['crop_array_sha256']
        if digest not in ocr or 'reuse_source_line_sha256' not in ocr[digest]:continue
        new=ocr[digest];assert new['reuse_source_line_sha256']==sha256((line+'\n').encode()).hexdigest()
        assert new['raw_text']==old['raw_text'] and new['sequence_confidence_native']==old['sequence_confidence_native']
        reused+=1
    assert reused==r['reused_ocr_crops'] and len(ocr)-reused==r['new_ocr_crops']
    per_image=[];used=set()
    for model,areas in r['results'].items():
        shards=[json.loads(path.read_text()) for path in sorted((directory/model/'proposals').glob('*.json'))]
        assert len(shards)==r['images'] and {s['sample_id'] for s in shards}==set(gt)
        for area,caps in areas.items():
            for cap,paddings in caps.items():
                limit=None if cap=='all' else int(cap)
                for pad,metrics in paddings.items():
                    totals=Counter()
                    for s in shards:
                        assert s['protocol_sha256']==r['protocol_sha256']
                        all_proposals=s['variants'][pad]['proposals']
                        used.update(x['crop_array_sha256'] for x in all_proposals)
                        proposals=[x for x in all_proposals if x['bbox_area_native']>=int(area)][:limit]
                        assert limit is None or len(proposals)<=limit
                        values=independent_metrics(proposals,gt[s['sample_id']],'bbox',ocr)
                        totals.update(values);per_image.append({'model':model,'area':int(area),'cap':cap,'padding':int(pad),'sample_id':s['sample_id'],**values})
                    for field,value in totals.items():
                        target=metrics['geometry'] if field in metrics['geometry'] else metrics['recognition']
                        assert target[field]==value,(model,area,cap,pad,field)
                    exact=metrics['exact_end_to_end'];tp=totals['exact_casefold_spatial_one_to_one'];n=totals['retained_proposals'];g=totals['gt_instances']
                    assert exact['tp']==tp and exact['predictions']==n and exact['gt']==g
                    assert exact['precision']==(tp/n if n else 0) and exact['recall']==(tp/g if g else 0)
                    assert exact['f1']==(2*tp/(n+g) if n+g else 0)
            for pad in map(str,c['native_paddings']):
                counts=[caps['all' if cap is None else str(cap)][pad]['exact_end_to_end']['tp'] for cap in c['limits']]
                assert counts==sorted(counts)
    assert used==set(ocr) and len(ocr)==r['unique_rgb_crops']
    assert not any(path.suffix in ['.png','.jpg','.pt','.npy','.npz'] for path in directory.rglob('*') if path.is_file())
    report={'status':'PASS','claim_label':r['claim_label'],'images':r['images'],'care_words':r['care_words'],
        'independent_per_image_evaluations':len(per_image),'qualified_reused_ocr_verified':reused,
        'monotonic_correct_word_count_with_nested_caps':'PASS','method':'NumPy IoU, SciPy maximum matching, Counter lexical',
        'per_image':per_image}
    (directory/'audit.json').write_text(json.dumps(report,sort_keys=True,separators=(',',':'))+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='per_image'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--result-directory',type=Path,required=True)
    audit(p.parse_args().result_directory)
