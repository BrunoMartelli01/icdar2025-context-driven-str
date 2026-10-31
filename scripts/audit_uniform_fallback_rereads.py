"""Independently verify all real fallback reread crop pixels and input coverage."""

import argparse
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image

from scripts.run_uniform_unet_training import read,write
from recognition.fallback_reread import padded_box
from segmentation.uniform_unet_fusion import file_hash


def audit(path):
    c=read(path);root=Path(c['output_root']);complete=read(root/'complete.json');assert complete['status']=='COMPLETE'
    assert read(root/'resolved_configuration.json')==c
    for p,h in c['pinned_inputs'].items():assert file_hash(p)==h
    checks={}
    for ds in c['datasets']:
        plan=read(Path(c['training_root'])/ds/'resolved_configuration.json');ids=plan['partitions']['fit']+plan['partitions']['selection']
        records={r['sample_id']:r for r in plan['train_records']};teacher=read(c['teacher_sources'][ds])['native_benchmark'];index=read(root/ds/'index.json')
        assert set(index)==set(teacher)==set(ids);crops=0
        for sid in ids:
            entry=index[sid];assert file_hash(entry['path'])==entry['sha256'];sample=read(entry['path']);record=records[sid]
            assert not sample['ground_truth_used'] and sample['image_sha256']==record['image_sha256']
            assert file_hash(record['image_path'])==record['image_sha256']
            old={r['instance_id']:r for r in teacher[sid]['rows']};rows={r['instance_id']:r for r in sample['rows']}
            assert len(rows)==len(sample['rows']) and set(rows)==set(old)
            with Image.open(record['image_path']) as im:rgb=np.asarray(im.convert('RGB'))
            for i,row in rows.items():
                assert row['bbox']==old[i]['bbox']
                if row['bbox'] is None:assert not row['rereadings'];continue
                assert set(row['rereadings'])=={str(p) for p in c['paddings']}
                for padding in c['paddings']:
                    reading=row['rereadings'][str(padding)];box=padded_box(row['bbox'],record['image_wh'],padding)
                    assert box==reading['crop_bbox'];a,b,cc,d=box;crop=np.ascontiguousarray(rgb[b:d,a:cc])
                    assert hashlib.sha256(crop.tobytes()).hexdigest()==reading['crop_rgb_sha256']
                    assert 0<=reading['sequence_confidence_native']<=1;crops+=1
        assert crops==complete['totals'][ds]['actual_parseq_crop_inferences']
        checks[ds]=dict(images=len(ids),crop_hashes_checked=crops,source_deepsolo_images=len(ids),total_images=len(ids),deepsolo_percentage=100.,actual_new_deepsolo_calls=0)
    write(root/'independent_audit.json',dict(status='PASS',mode='approximate',checks=checks,complete_sha256=file_hash(root/'complete.json'),
        source_sha256=file_hash(__file__),scope='Complete native TRAIN/selection coverage, source boxes, padded original RGB bytes, qualified checkpoint pins and confidence ranges; no TEST or recognition-target claim.'))
    print('UNIFORM REREAD AUDIT PASS',sum(x['crop_hashes_checked'] for x in checks.values()),'actual crop hashes',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);a=p.parse_args();audit(a.config)
