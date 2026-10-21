"""Independently verify compact v2 outputs against their unchanged v1 sources."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import find_objects

from segmentation.data import file_sha256
from segmentation.tighter_word_gt import native_training_owners


def verify(root, manifest_path):
    data=json.loads(manifest_path.read_text())
    config_path=manifest_path.parent/'resolved_config.json'
    config=json.loads(config_path.read_text())
    assert data['claim_label']==config['claim_label']=='approximate reproduction'
    assert config['target_source']==data['target_source']=='compact_word_cores_v2'
    assert file_sha256(config_path)==data['resolved_config_sha256']
    assert len(data['samples'])==data['images']
    assert len({r['id'] for r in data['samples']})==data['images']
    params=config['parameters']; words=fragmented=before_area=after_area=0
    for offset,row in enumerate(data['samples'],1):
        source_path=Path(row['source_owners'])
        assert file_sha256(source_path)==row['source_owners_sha256']
        assert file_sha256(Path(row['source_record']))==row['source_record_sha256']
        source_row=json.loads(Path(row['source_record']).read_text())
        assert row['instances']==source_row['instances'] and row['image']==source_row['image']
        assert file_sha256(Path(row['image']))==row['image_sha256']
        with Image.open(row['image']) as im: image_size=im.size
        with Image.open(source_path) as im: before=np.asarray(im).copy()
        arrays={}
        for name in ['owners','masks','validity','separators']:
            path=root/row[name]
            assert file_sha256(path)==row[name+'_sha256']
            with Image.open(path) as im: arrays[name]=np.asarray(im).copy()
            assert arrays[name].shape==before.shape
        own=arrays['owners']; mask=arrays['masks']; valid=arrays['validity']; sep=arrays['separators']
        assert (own.shape[1],own.shape[0])==image_size==tuple(row['size_wh'])
        old_valid_path=source_path.parent.parent/'validity'/source_path.name
        assert file_sha256(old_valid_path)==source_row['validity_sha256']
        with Image.open(old_valid_path) as im: old_valid=np.asarray(im)
        assert np.array_equal(valid,old_valid)
        assert all(set(np.unique(arrays[n])).issubset({0,255}) for n in ['masks','validity','separators'])
        assert np.array_equal(own>0,mask>0)
        assert not np.any((own>0)&(own!=before))
        assert not np.any((own>0)&((valid==0)|(sep>0)))
        expected=len(row['instances'])
        assert set(np.unique(own[own>0]))==set(range(1,expected+1))
        count,labels=cv2.connectedComponents((mask>0).astype(np.uint8),connectivity=8)
        shared={}; counts={}
        windows=find_objects(before,max_label=expected)
        for owner_id,window in enumerate(windows,1):
            original=before[window]==owner_id; current=own[window]==owner_id
            current_y,current_x=np.where(current); old_y,old_x=np.where(original)
            sy,sx=params['reference_wh'][1]/own.shape[0],params['reference_wh'][0]/own.shape[1]
            inset=[(current_x.min()-old_x.min())*sx,(current_y.min()-old_y.min())*sy,
                   (old_x.max()-current_x.max())*sx,(old_y.max()-current_y.max())*sy]
            assert max(inset)<=params['crop_padding_reference']+1e-10
            assert current.sum()>=params['min_area_fraction']*original.sum()
            assert max(np.ptp(current_x)+1,np.ptp(current_y)+1)>=params['min_span_fraction']*max(np.ptp(old_x)+1,np.ptp(old_y)+1)
            ids=np.unique(labels[window][current]); counts[str(owner_id)]=len(ids)
            for component in ids: shared.setdefault(int(component),set()).add(owner_id)
        assert all(len(v)==1 for v in shared.values()) and len(shared)==count-1
        assert counts==row['compact_audit']['components_per_word']
        old_sep_path=source_path.parent.parent/'separators'/source_path.name
        assert file_sha256(old_sep_path)==source_row['separators_sha256']
        with Image.open(old_sep_path) as im: old_sep=np.asarray(im).copy()
        assert not np.any((old_sep>0)&(sep==0))
        a=native_training_owners(before,valid,old_sep,tuple(params['reference_wh']))
        b=native_training_owners(own,valid,sep,tuple(params['reference_wh']))
        assert set(np.unique(a[a>0]))<=set(np.unique(b[b>0]))
        words+=expected;fragmented+=sum(n>1 for n in counts.values())
        before_area+=int(np.count_nonzero(before));after_area+=int(np.count_nonzero(own))
        if offset%100==0 or offset==data['images']:
            print(f"[independent compact audit] {data['dataset']}/{data['split']}: {offset}/{data['images']}",flush=True)
    assert words==data['words'] and fragmented==data['fragmented_words']
    assert before_area==data['source_foreground_pixels'] and after_area==data['foreground_pixels']
    return {'dataset':data['dataset'],'split':data['split'],'images':data['images'],'words':words,
            'missing_words':0,'merged_word_components':0,'fragmented_words':fragmented,
            'extra_words_lost_at_native128':0,'source_foreground_pixels':before_area,'foreground_pixels':after_area,
            'foreground_reduction_fraction':1-after_area/before_area,
            'manifest_sha256':file_sha256(manifest_path)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    reports=[verify(args.root,p) for p in sorted(args.root.glob('*/*/manifest.json'))]
    if not reports: raise ValueError('No completed compact GT split found')
    args.output.write_text(json.dumps({'claim_label':'approximate reproduction','reproduction_mode':'approximate',
        'verifier':'OpenCV binary8-components; source subset/padding/native-survival checks',
        'verifier_sha256':file_sha256(Path(__file__)),'datasets':reports},indent=2)+'\n')


if __name__=='__main__': main()
