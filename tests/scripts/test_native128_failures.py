import numpy as np
import pytest

from scripts.analyze_native128_failures import (
    augment, box_matrices, classify_image, owner_stats, summarize_owner_rows,
)


def proposal(index, box, digest):
    return {'contour_index':index,'bbox':box,'padded_bbox_original':box,
            'crop_array_sha256':digest}


def test_matching_preserves_seed_gt_when_augmenting_path_changes_pairs():
    initial={0:0}
    edges=np.asarray([[1,1],[1,0]],bool)
    result=augment(edges,initial)
    assert len(result)==2 and set(initial.values())<=set(result.values())
    assert result=={1:0,0:1}


def test_matching_handles_empty_rectangular_populations():
    assert augment(np.zeros((0,3),bool))=={}
    assert augment(np.zeros((4,0),bool))=={}


def test_initial_matching_must_be_feasible_and_one_to_one():
    with pytest.raises(ValueError,match='Invalid initial'):
        augment(np.eye(2,dtype=bool),{0:1})
    with pytest.raises(ValueError,match='Invalid initial'):
        augment(np.ones((2,2),bool),{0:0,1:0})


def test_failure_categories_are_additive_and_ocr_is_separate_from_geometry():
    words=[{'bbox':[i*20,0,i*20+10,10],'text':str(i)} for i in range(4)]
    rows=[proposal(i,w['bbox'],str(i)) for i,w in enumerate(words[:3])]
    ocr={str(i):{'raw_text':str(i) if i!=2 else 'WRONG'} for i in range(3)}
    shard={'sample_id':'synthetic','image_hw':[128,128],'variants':{'1':{'proposals':rows}}}
    d=classify_image(shard,words,ocr,cap=1)
    assert d['categories']=={'correct_top10':1,'correct_only_beyond_cap':1,
        'localized_but_ocr_or_crop_error':1,'not_localized_iou50':1}
    assert d['correct_unlimited']==2 and d['geometry_unlimited']==3


def test_padding_oracle_cannot_use_two_reads_of_one_component_as_two_words():
    words=[{'bbox':[0,0,10,10],'text':'ONE'},{'bbox':[0,0,10,10],'text':'TWO'}]
    rows=[proposal(1,[0,0,10,10],'one')]
    other=[proposal(1,[0,0,10,10],'two')]
    shard={'sample_id':'synthetic','image_hw':[128,128],
        'variants':{'1':{'proposals':rows},'2':{'proposals':other}}}
    ocr={'one':{'raw_text':'ONE'},'two':{'raw_text':'TWO'}}
    d=classify_image(shard,words,ocr)
    assert d['correct_per_contour_padding_oracle']==1


def test_empty_predictions_leave_all_words_unlocalized():
    shard={'sample_id':'synthetic','image_hw':[128,128],'variants':{'1':{'proposals':[]}}}
    words=[{'bbox':[0,0,10,10],'text':'ONE'}]
    d=classify_image(shard,words,{})
    assert d['categories']=={'not_localized_iou50':1}


def test_padded_box_sensitivity_does_not_change_primary_detection_box_metric():
    p=proposal(0,[3,3,7,7],'one');p['padded_bbox_original']=[0,0,10,10]
    shard={'sample_id':'synthetic','image_hw':[128,128],'variants':{'1':{'proposals':[p]}}}
    d=classify_image(shard,[{'bbox':[0,0,10,10],'text':'ONE'}],{'one':{'raw_text':'ONE'}})
    assert d['correct_unlimited']==0 and d['correct_padded_box_metric']==1


def test_bbox_proxies_do_not_require_mask_topology_or_claim_pixel_merges():
    ps=[proposal(0,[0,0,20,10],'x')]
    words=[{'bbox':[0,0,10,10],'text':'ONE'},{'bbox':[10,0,20,10],'text':'TWO'}]
    iou,coverage_gt,coverage_pred=box_matrices(ps,words)
    np.testing.assert_allclose(iou,[[.5,.5]])
    np.testing.assert_allclose(coverage_gt,[[1,1]])
    np.testing.assert_allclose(coverage_pred,[[.5,.5]])


def test_actual_owner_topology_counts_missing_and_fragmented_separately():
    native=np.zeros((8,8),np.int32)
    native[1,1]=1;native[5,5]=1;native[2:4,5:7]=2
    rows=owner_stats(native,[1,2,3])
    assert rows[1]['components']==2 and rows[2]['components']==1 and rows[3]['components']==0
    summary=summarize_owner_rows(list(rows.values()))
    assert summary['fragmented_words']==1 and summary['missing_words']==1
    assert summary['excess_components']==1 and summary['foreground_pixels']==6
