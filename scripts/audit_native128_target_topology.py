"""Check actual native128 survival and fragmentation, not full-resolution GT only."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from segmentation.data import load_segmentation_manifest, file_sha256
from segmentation.tighter_word_gt import native_training_owners


def main(config_path, statistics_path, output):
    c=json.loads(config_path.read_text())
    stats=json.loads(statistics_path.read_text())
    reports={}
    for family in ['separated','compact']:
        source=next(v for v in c['gt_versions'] if v['id']==family)
        manifest=load_segmentation_manifest(Path(source['manifest']),require_checksums=True)
        frozen={r['sample_id']:r for r in stats['versions'][family]['per_image']}
        root=Path(source['data_root']);rows=[];missing=set();fragmented=set();extras=0;total_words=0
        for index,s in enumerate(manifest.samples,1):
            arrays=[]
            for path,digest,mode in [(s.word_owner_annotation_path,s.word_owner_annotation_sha256,'I'),
                                    (s.validity_annotation_path,s.validity_annotation_sha256,'L'),
                                    (s.separation_annotation_path,s.separation_annotation_sha256,'L')]:
                if file_sha256(root/path)!=digest:raise ValueError('GT source changed')
                with Image.open(root/path) as im:arrays.append(np.asarray(im.convert(mode)).copy())
            expected=[int(v) for v in np.unique(arrays[0]) if v>0];total_words+=len(expected)
            native=native_training_owners(*arrays,(128,128))
            binary=(native>0).astype(np.uint8)
            if sha256(binary.tobytes()).hexdigest()!=frozen[s.sample_id]['native_target_array_sha256']:
                raise ValueError('Fast owner replay differs from actual TRAIN target')
            lost=[];split=[];excess=0;count=0
            for owner in expected:
                ys,xs=np.where(native==owner)
                if not len(ys):lost.append(owner);missing.add(f'{s.sample_id}:{owner}');continue
                roi=(native[ys.min():ys.max()+1,xs.min():xs.max()+1]==owner).astype(np.uint8)
                n=cv2.connectedComponents(roi,connectivity=8)[0]-1
                count+=n;excess+=n-1
                if n>1:split.append({'owner':owner,'components':n});fragmented.add(f'{s.sample_id}:{owner}')
            if count!=frozen[s.sample_id]['native_components']:
                raise ValueError('Cross-owner foreground connectivity or component mismatch')
            extras+=excess
            rows.append({'sample_id':s.sample_id,'words':len(expected),'missing_owner_ids':lost,
                         'fragmented_words':split,'excess_components':excess,'components':count})
            if index%250==0 or index==len(manifest.samples):print(f'[topology] {family} {index}/{len(manifest.samples)}',flush=True)
        reports[family]={'images':len(rows),'expected_words':total_words,'missing_words':len(missing),
            'fragmented_words':len(fragmented),'excess_components':extras,'missing_ids':sorted(missing),
            'fragmented_ids':sorted(fragmented),'per_image':rows}
    a,b=reports['separated'],reports['compact']
    report={'status':'PASS','claim_label':'approximate reproduction','scope':'all1255 TRAIN actual native targets; no augmentation',
        'native_array_hashes_match_actual_training_transform':True,'versions':reports,
        'newly_missing_words':sorted(set(b['missing_ids'])-set(a['missing_ids'])),
        'newly_fragmented_words':sorted(set(b['fragmented_ids'])-set(a['fragmented_ids'])),
        'fragmentation_repaired_words':sorted(set(a['fragmented_ids'])-set(b['fragmented_ids'])),
        'code_sha256':file_sha256(Path(__file__)),'config_sha256':file_sha256(config_path),
        'statistics_sha256':file_sha256(statistics_path)}
    output.write_text(json.dumps(report,sort_keys=True,separators=(',',':'))+'\n')
    print(json.dumps({'status':report['status'],'versions':{k:{n:r[n] for n in ['expected_words','missing_words','fragmented_words','excess_components']} for k,r in reports.items()},'newly_fragmented_words':len(report['newly_fragmented_words']),'newly_missing_words':len(report['newly_missing_words'])}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--statistics',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.config,a.statistics,a.output)
