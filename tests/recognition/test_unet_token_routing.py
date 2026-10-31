"""Scientific boundaries for frozen U-Net token routing and transferred cuts."""
import numpy as np
import pytest

from recognition.unet_token_routing import unet_proposals, selected_rows, weak_word_labels, fit_thresholds, apply_threshold, fallback_counts


SETTINGS = dict(native_size=8, text_threshold=.65, kernel_threshold=.5, maximum_ocr_candidates=20,
                minimum_transcript_probability=.7)


def test_two_predicted_kernels_keep_neighbouring_words_separate():
    p = np.zeros((2,8,8),np.float32); p[0,2:5,1:7]=1; p[1,3,2]=p[1,3,5]=1
    boxes,labels,audit = unet_proposals(p,[80,80],SETTINGS)
    assert len(boxes)==2 and audit['seed_components']==2
    assert set(np.unique(labels))=={0,1,2}
    assert all(0<=r['bbox'][0]<r['bbox'][2]<=80 for r in boxes)
    assert all(r['detector_score']==1. for r in boxes)


def test_unet_uses_original_pixels_and_declared_candidate_cap():
    p=np.zeros((2,8,8),np.float32);p[:,2:4,2:4]=1;p[:,5:7,5:7]=1
    values,_,_=unet_proposals(p,[80,40],dict(SETTINGS,maximum_ocr_candidates=1))
    assert len(values)==1 and all(isinstance(x,int) for x in values[0]['bbox'])
    assert 0<=values[0]['bbox'][1]<values[0]['bbox'][3]<=40


def test_transcript_product_filter_matches_declared_threshold_including_boundary():
    raw=[dict(raw_text='a',token_probabilities=[.9],token_log_probabilities=[np.log(.9)],sequence_confidence_native=.7),
         dict(raw_text='bb',token_probabilities=[.99,.99],token_log_probabilities=[np.log(.99)]*2,sequence_confidence_native=.69)]
    values=selected_rows(raw,SETTINGS)
    assert [v['raw_text'] for v in values]==['a']
    assert values[0]['confidence']==pytest.approx(.9)


def test_weak_labels_use_alphanumeric_text_and_spatial_overlap():
    rows=[dict(raw_text='CAFÉ!',bbox=[0,0,10,10]),dict(raw_text='café',bbox=[20,20,30,30])]
    labels=weak_word_labels(rows,[dict(text='café',bbox=[0,0,10,10])],dict(auxiliary_iou=.5))
    assert labels.tolist()==[1.,0.]


def test_scalar_threshold_is_fit_on_train_and_transferred_without_test_quantiles():
    train={'a':3.,'b':2.,'c':1.,'d':None};test={'x':2.5,'y':1.5,'z':None}
    policy=fit_thresholds(train,[.5])[0]
    assert policy['threshold']==2.
    assert apply_threshold(test,policy)==['x']


def test_scalar_cuts_preserve_ties_and_reject_empty_readings():
    values={'a':1.,'b':1.,'c':None};policy=fit_thresholds(values,[.3])[0]
    assert apply_threshold(values,policy)==['a','b']
    assert apply_threshold(values,fit_thresholds(values,[0.])[0])==[]


def test_cost_counts_report_actual_rounding_and_total_images():
    value=fallback_counts(dict(images=1253,accepted_images=626,fallback_images=627))
    assert value['deepsolo_images']==627 and value['total_evaluated_images']==1253
    assert value['deepsolo_percentage']==pytest.approx(50.03990422984836)
    with pytest.raises(ValueError):fallback_counts(dict(images=300,accepted_images=150,fallback_images=151))


def test_invalid_maps_and_nonfinite_cut_scores_fail():
    with pytest.raises(ValueError):unet_proposals(np.zeros((1,8,8)),[80,80],SETTINGS)
    with pytest.raises(ValueError):fit_thresholds({'a':float('nan')},[.5])


def test_paired_features_unpack_qualified_archive_and_keep_backbone_specific_words(tmp_path, monkeypatch):
    from scripts import unet_token_router_stages as stages
    scene = np.ones((5, 4), np.float32)
    original = dict(patches=scene, word_text=np.ones((2, 4), np.float32))
    monkeypatch.setattr(stages, 'arrays', lambda *args: (dict(original), dict(eligible=False)))
    monkeypatch.setattr(stages, 'light_rows', lambda *args: [dict(raw_text='word')])
    stages.write_arrays(tmp_path/'train/unet_features/image.npz', dict(word_text=np.zeros((1, 4), np.float32)))
    plan = dict(config=dict(backbones=['unet', 'yolo']), paths=dict(token_root=str(tmp_path)),
                samples=dict(train=[dict(sample_id='image')]))
    values = stages.load_features(plan, tmp_path, 'train')
    assert values['yolo']['image']['eligible'] is False
    assert values['unet']['image']['eligible'] is True
    assert values['yolo']['image']['word_text'].shape == (2, 4)
    assert values['unet']['image']['word_text'].shape == (1, 4)
    assert values['unet']['image']['patches'] is scene


def test_fresh_test_features_prevent_train_selection(tmp_path):
    from scripts.unet_token_router_stages import train
    directory = tmp_path/'test/unet_features'
    directory.mkdir(parents=True)
    (directory/'sample.npz').write_bytes(b'fresh TEST features')
    with pytest.raises(ValueError, match='Cannot fit or select'):
        train({}, tmp_path)
