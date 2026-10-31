import numpy as np
import pytest

from recognition.unet_corrected_token_interfaces import (native_observation, verified_source_crop, lexical_counts,
    ranked_ids, route_summary, weak_word_labels)


def test_native_confidence_after_character_deletion_keeps_original_provenance():
    parent=dict(id='a',text='B&B',bbox=[1,2,30,40],crop_array_sha256='rgb')
    edited=dict(parent,id='edit:a',text='BB',source_proposal_id='a')
    reading=dict(raw_text='B&B',token_probabilities=[.9,.8,.9])
    result=native_observation(edited,parent,reading,.85,.95)
    assert result['raw_text']=='BB' and result['original_raw_text']=='B&B'
    assert result['bbox']==parent['bbox'] and result['confidence']==.85
    assert result['token_probabilities']==reading['token_probabilities']
    assert result['inherited_box_and_confidence']
    with pytest.raises(ValueError):
        native_observation(edited,parent,dict(reading,raw_text='BB'),.85,.95)
    with pytest.raises(ValueError):
        native_observation(dict(edited,bbox=[0,0,3,3]),parent,reading,.85,.95)


def test_symbol_only_and_repeated_words_remain_in_denominators():
    result = lexical_counts([dict(raw_text='A!'), dict(raw_text='?'), dict(raw_text='A')],
                            [dict(text='a'), dict(text='a'), dict(text='!')], [False, False, True])
    assert result == dict(tp=1, fp=1, fn=2, detected_words=2, gt_words=3,
                          all_outputs=3, ignored_outputs=1)


def test_router_counts_use_actual_denominator_and_exclude_ineligible():
    scores = {'b': .5, 'a': .5, 'c': None}
    assert ranked_ids(scores, list(scores), 1) == ['a', 'b']
    metric = dict(tp=1, fp=0, fn=0, detected_words=1, gt_words=1, all_outputs=1, ignored_outputs=0)
    summary = route_summary({s: dict(light=metric, deepsolo=metric) for s in scores},
                            list(scores), ranked_ids(scores, list(scores), 1))
    assert (summary['deepsolo_images'], summary['total_images']) == (1, 3)
    assert summary['deepsolo_percent'] == pytest.approx(100/3)
    with pytest.raises(ValueError):
        ranked_ids({'a': np.nan}, ['a'], .5)
    with pytest.raises(ValueError):
        route_summary({}, [], [])


def test_weak_spatial_labels_use_same_normalization_but_reject_empty():
    rows = [dict(raw_text='CO-OP', bbox=[0, 0, 10, 10]), dict(raw_text='!', bbox=[0, 0, 10, 10])]
    assert weak_word_labels(rows, [dict(text='coop', bbox=[0, 0, 10, 10])], .5).tolist() == [1., -1.]


def test_source_crop_hash_includes_shape_and_rejects_changed_pixels():
    import hashlib
    rgb=np.arange(4*5*3,dtype=np.uint8).reshape(4,5,3)
    crop=rgb[1:3,1:4]
    digest=hashlib.sha256(np.asarray(crop.shape,dtype='<i8').tobytes()+crop.tobytes()).hexdigest()
    row=dict(bbox=[0,0,5,4],crop_bbox=[1,1,4,3],crop_rgb_sha256=digest)
    assert np.array_equal(verified_source_crop(rgb,row),crop)
    bad=rgb.copy();bad[1,1,0]+=1
    with pytest.raises(ValueError):verified_source_crop(bad,row)
    with pytest.raises(ValueError):verified_source_crop(rgb,dict(row,crop_rgb_sha256=hashlib.sha256(crop.tobytes()).hexdigest()))


def test_oom_is_recorded_and_propagated(tmp_path,monkeypatch):
    import json
    import sys
    import torch
    from scripts import run_unet_corrected_token_router as runner
    from scripts import run_word_slot_methods as source
    config=tmp_path/'configs/approximate/test.json';config.parent.mkdir(parents=True)
    upstream=tmp_path/'upstream';upstream.mkdir()
    (upstream/'protocol.json').write_text(json.dumps(dict(config=dict(runtime={}))))
    config.write_text(json.dumps(dict(reproduction_mode='approximate',minimum_free_gpu_bytes=1,
                                     paths=dict(slot_root=str(upstream)))))
    monkeypatch.setattr(runner,'REPO',tmp_path)
    def fail(_):raise torch.cuda.OutOfMemoryError('injected qualification OOM')
    monkeypatch.setattr(source,'setup',fail)
    output=tmp_path/'run'
    monkeypatch.setattr(sys,'argv',['runner','--mode','approximate','--config',str(config),'--output',str(output)])
    with pytest.raises(torch.cuda.OutOfMemoryError):runner.main()
    failure=json.loads((output/'failure.json').read_text())
    assert failure['oom'] and failure['error_type']=='OutOfMemoryError'


def test_feature_stage_reuse_accepts_only_a_resource_margin_change(tmp_path):
    import hashlib
    import json
    from scripts.run_unet_corrected_token_router import reuse_train_features
    previous=tmp_path/'previous';previous.mkdir()
    records=dict(train=[dict(sample_id='a')])
    before=dict(config={},records=records,smoke=True,student_selection='frozen',
                source_config=dict(minimum_free_gpu_bytes=8,scientific='fixed'))
    (previous/'resolved_configuration.json').write_text(json.dumps(before))
    (previous/'memory').mkdir();(previous/'memory/clip_train.json').write_text(json.dumps(dict(oom=False)))
    for variant in ('baseline','corrected'):
        folder=previous/'train/features'/variant;folder.mkdir(parents=True)
        path=folder/'a.npz';np.savez_compressed(path,scalars=np.zeros(18,np.float32))
        path.with_suffix('.json').write_text(json.dumps(dict(arrays_sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
    for name in ('raw','corrected'):
        (previous/'train'/(name+'.json')).write_text(json.dumps({'a':[dict(raw_text='A')]}))
    (previous/'train/correction_trace.json').write_text('{}')
    (previous/'train/branch_metrics_original.json').write_text('{}')
    (previous/'train_expert_qualification.json').write_text('{}')
    output=tmp_path/'new';output.mkdir()
    after=dict(before,source_config=dict(minimum_free_gpu_bytes=2,scientific='fixed'))
    features=reuse_train_features(previous,output,after)
    assert features['corrected']['a']['eligible']
    assert json.loads((output/'train_stage_reuse.json').read_text())['scientific_fields_identical']
    bad=dict(after,source_config=dict(minimum_free_gpu_bytes=2,scientific='changed'))
    with pytest.raises(ValueError):reuse_train_features(previous,tmp_path/'bad',bad)
