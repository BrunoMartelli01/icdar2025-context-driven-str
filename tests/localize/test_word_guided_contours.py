import cv2
import numpy as np
import pytest

from localize.repository_contours import RepositoryContourParams, audit_repository_contours
from localize.word_guided_contours import findcontour_labels, refine_instances, word_contour_crops


POST = {"text_threshold": .65, "kernel_threshold": .5, "center_threshold": .5,
        "center_min_distance": 3, "watershed_weights": [.5, .25, .25],
        "method": "watershed", "padding": 0, "min_area": 0, "max_instances": None}
REFINE = {"bridge_text_threshold": .5, "maximum_foreground_distance": 3,
          "kernel_merge": True, "center_merge": True}


def fragments(channels=2):
    p = np.zeros((channels, 32, 32), np.float32)
    p[0, 10:16, 4:11] = .9
    p[0, 10:16, 13:20] = .9
    p[0, 11:15, 11:13] = .55
    p[1, 11:15, 6:18] = .9
    return p, np.ones((32, 32), bool)


def peaks(shape, locations, sigma=1):
    yy, xx = np.indices(shape)
    return np.maximum.reduce([np.exp(-((yy-y)**2+(xx-x)**2)/(2*sigma**2)) for y,x in locations]).astype(np.float32)


def test_baseline_matches_frozen_findcontours_boxes_and_keeps_holes():
    p = np.zeros((1, 32, 32), np.float32)
    p[0, 2:30, 2:30] = 1
    p[0, 4:28, 4:28] = 0
    p[0, 12:17, 12:17] = 1  # Nested island is omitted by frozen RETR_EXTERNAL.
    foreground = p[0].astype(bool)
    original = audit_repository_contours(foreground.astype(np.uint8), RepositoryContourParams(0, 0, 1024))
    rows, labels, audit = word_contour_crops(p, np.ones_like(foreground), (32, 32), POST, REFINE, method="findcontours")
    assert [row['bbox_native'] for row in rows] == [row['bbox'] for row in original['retained']]
    assert audit['nested_foreground_pixels_omitted'] == 25
    assert labels[7, 7] == labels[14, 14] == 0
    assert np.count_nonzero(labels) == np.count_nonzero(foreground) - 25


def test_shared_kernel_joins_fragments_without_adding_pixels_and_keeps_one_crop():
    p, canvas = fragments()
    base, _, _ = word_contour_crops(p, canvas, (64, 32), POST, REFINE, method="findcontours")
    rows, labels, audit = word_contour_crops(p, canvas, (64, 32), POST, REFINE, method="refined")
    assert len(base) == 2 and len(rows) == 1
    assert rows[0]['bbox_native'] == [4, 10, 20, 16]
    assert rows[0]['mask_fragment_contours'] == 2
    assert audit['unions'][0]['evidence'] == 'shared_kernel_bridge'
    assert np.array_equal(labels > 0, p[0] >= .65)
    assert labels[12, 11] == labels[12, 12] == 0


def test_distinct_centers_veto_shared_kernel_merge():
    p, canvas = fragments(3)
    p[2] = peaks(canvas.shape, [(12, 8), (12, 16)])
    labels, audit = refine_instances(p, canvas, POST, REFINE)
    assert len(np.unique(labels[labels > 0])) == 2
    assert any(row['reason'] == 'multiple_center_peaks' for row in audit['rejected_unions'])


def test_foreground_distance_gate_uses_pixels_instead_of_bbox_overlap():
    p, canvas = fragments()
    labels, audit = refine_instances(p, canvas, POST, {**REFINE, 'maximum_foreground_distance': 2})
    assert len(np.unique(labels[labels > 0])) == 2
    assert any(row['reason'] == 'foreground_distance' for row in audit['rejected_unions'])


def test_nearby_masks_without_common_auxiliary_evidence_are_not_joined():
    p, canvas = fragments()
    p[1, :, 11:13] = 0
    labels, audit = refine_instances(p, canvas, POST, REFINE)
    assert len(np.unique(labels[labels > 0])) == 2 and not audit['unions']


def test_existing_seeded_split_survives_inside_one_contour():
    p = np.zeros((2, 32, 32), np.float32)
    p[0, 10:16, 4:24] = .9
    p[1, 11:15, 5:10] = .9
    p[1, 11:15, 18:23] = .9
    canvas = np.ones((32, 32), bool)
    base, _, _ = word_contour_crops(p, canvas, (32,32), POST, REFINE, method='findcontours')
    refined, labels, _ = word_contour_crops(p, canvas, (32,32), POST, REFINE, method='refined')
    assert len(base) == 1 and len(refined) == 2
    assert np.array_equal(labels > 0, p[0] >= .65)


def test_single_center_region_can_recombine_false_seeded_partitions():
    p = np.zeros((3,32,32),np.float32)
    p[0,10:16,4:24] = .9
    p[1,11:15,5:10] = .9
    p[1,11:15,18:23] = .9
    p[2] = peaks((32,32),[(13,14)],sigma=9)
    canvas = np.ones((32,32),bool)
    labels,audit = refine_instances(p,canvas,POST,REFINE)
    assert len(np.unique(labels[labels>0])) == 1
    assert audit['unions'][0]['evidence'] == 'single_center_region'


def test_distinct_centers_also_veto_transitive_kernel_unions():
    p = np.zeros((3,32,32),np.float32)
    for start in [2,10,18]: p[0,8:16,start:start+6] = .9
    p[0,9:11,8:10] = p[0,13:15,16:18] = .55
    p[1,9:11,4:14] = p[1,13:15,12:22] = .9
    p[2] = peaks((32,32),[(9,5),(14,21)],sigma=.6)
    _,labels = cv2.connectedComponents((p[0] >= .65).astype(np.uint8),connectivity=8)
    refined,audit = refine_instances(p,np.ones((32,32),bool),POST,REFINE,cached_labels=labels)
    assert len(np.unique(refined[refined>0])) == 2
    assert any(row['reason']=='distinct_centers_in_union' for row in audit['rejected_unions'])


def test_component_area_filter_is_not_bbox_area():
    p=np.zeros((1,32,32),np.float32)
    for value in range(4): p[0,5+value,5+value]=1
    rows,labels,_=word_contour_crops(p,np.ones((32,32),bool),(32,32),{**POST,'min_area':5},REFINE,method='findcontours')
    assert rows==[] and not labels.any()


def test_empty_foreground_and_canvas_are_preserved():
    p=np.zeros((3,32,32),np.float32)
    for method in ['findcontours','seeded','refined']:
        rows,labels,_=word_contour_crops(p,np.ones((32,32),bool),(32,32),POST,REFINE,method=method)
        assert rows==[] and not labels.any()


def test_one_pixel_words_are_kept_and_output_is_deterministic():
    p=np.zeros((2,32,32),np.float32);p[:,7,9]=1
    first=word_contour_crops(p,np.ones((32,32),bool),(32,32),POST,REFINE,method='refined')
    second=word_contour_crops(p,np.ones((32,32),bool),(32,32),POST,REFINE,method='refined')
    assert first[0]==second[0] and first[2]==second[2]
    assert np.array_equal(first[1],second[1])
    assert first[0][0]['foreground_area']==1


@pytest.mark.parametrize('change',[{'bridge_text_threshold':.7},{'maximum_foreground_distance':0},{'kernel_merge':1}])
def test_invalid_refinement_settings_fail(change):
    p,canvas=fragments()
    with pytest.raises(ValueError): refine_instances(p,canvas,POST,{**REFINE,**change})


def test_cached_maps_require_exact_foreground_and_auxiliary_head():
    p,canvas=fragments()
    with pytest.raises(ValueError,match='Cached seeded'): refine_instances(p,canvas,POST,REFINE,cached_labels=np.zeros_like(canvas,dtype=np.int32))
    with pytest.raises(ValueError,match='kernel head'): refine_instances(p[:1],canvas,POST,REFINE)
