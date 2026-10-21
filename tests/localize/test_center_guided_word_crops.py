import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from localize.center_guided_word_crops import center_guided_masks, center_guided_crops, _nearest_centers
from segmentation.word_center_offsets import make_word_center_offsets


@pytest.fixture
def config():
    return {"reproduction_mode":"approximate", "text_threshold":.65, "growth_text_threshold":.35,
            "smoothing_sigma":1., "min_peak_density":.1, "min_center_distance":4,
            "min_center_support":2, "assignment_radius":3., "growth_distance":3}


def predict(owners, p, config, canvas=None):
    canvas = np.ones_like(owners, bool) if canvas is None else canvas
    offsets, _ = make_word_center_offsets(owners, canvas)
    return center_guided_masks(p, offsets, canvas, config)


def test_two_centers_partition_fused_mask_without_erasing_text(config):
    owners = np.zeros((32,32), np.int32)
    owners[5:27,2:16] = 1
    owners[5:27,16:30] = 2
    p = np.where(owners,.9,0).astype(np.float32)
    result = predict(owners,p,config)
    assert result["audit"]["base_components"] == 1
    assert result["audit"]["retained_centers"] == result["audit"]["preserved_instances"] == 2
    assert result["preserved_labels"][10,5] != result["preserved_labels"][10,25]
    assert np.array_equal(result["preserved_labels"] > 0, p >= .65)
    assert np.array_equal(result["expanded_labels"],result["preserved_labels"])


def test_disconnected_same_word_fragments_share_one_crop(config):
    owners = np.zeros((32,32),np.int32)
    owners[8:24,3:11] = owners[8:24,19:27] = 1
    result = predict(owners,(owners>0).astype(float),config)
    assert result["audit"]["base_components"] == 2
    assert result["audit"]["preserved_instances"] == 1
    center = result["centers"][0]
    x,y = center["vote_peak_xy_native"]
    mx,my = center["marker_xy_native"]
    assert owners[y,x] == 0 and owners[my,mx] == 1
    crops = center_guided_crops(result["preserved_labels"],(64,32),
                               {"padding":0,"min_area":0,"max_instances":None})
    assert len(crops) == 1 and crops[0]["foreground_area"] == 2*16*8
    # Same ID spans both fragments; foreground area is not the bounding rectangle area.
    assert crops[0]["bbox_native"] == [3,8,27,24]


def test_growth_requires_probability_endpoint_identity_distance_and_connection(config):
    owners = np.zeros((32,32),np.int32)
    owners[7:25,2:16] = 1
    owners[7:25,16:30] = 2
    p = np.where(owners,.4,0).astype(float)
    p[10:22,5:13] = p[10:22,19:27] = .9
    p[10,3] = .2  # Below growth threshold, never admitted.
    result = predict(owners,p,config)
    core, grown = result["core"], result["expanded_labels"]
    assert result["audit"]["added_pixels"] > 0
    assert grown[10,3] == 0
    assert np.all(distance_transform_edt(~core)[grown>0] <= 3)
    assert np.all(grown[core] == result["preserved_labels"][core])
    left_ids = np.unique(grown[(owners == 1)&(grown>0)])
    right_ids = np.unique(grown[(owners == 2)&(grown>0)])
    assert len(left_ids) == len(right_ids) == 1 and left_ids[0] != right_ids[0]
    # A separate uncertain island cannot start a crop of its own.
    p2 = np.zeros((32,32));p2[10:20,3:8] = .9;p2[10:20,11:13] = .4
    one_owner = (p2>0).astype(np.int32)
    isolated = predict(one_owner,p2,config)
    assert not isolated["expanded_labels"][:,11:13].any()


def test_missing_or_invalid_cues_fall_back_to_preserved_components(config):
    p = np.zeros((32,32));p[3:12,3:12] = .9;p[20:27,20:27] = .9
    canvas = np.ones_like(p,bool)
    result = center_guided_masks(p,np.ones((2,32,32)),canvas,config)
    assert result["audit"]["retained_centers"] == 0
    assert result["audit"]["unseeded_fallback_components"] == 2
    assert np.array_equal(result["expanded_labels"]>0,p>.65)
    result = center_guided_masks(p,np.zeros((2,32,32)),canvas,{**config,"min_center_support":10000})
    assert result["audit"]["retained_centers"] == 0
    assert result["audit"]["preserved_instances"] == 2


def test_empty_canvas_and_deterministic_repeat(config):
    p = np.full((16,16),.9);offsets = np.zeros((2,16,16))
    result = center_guided_masks(p,offsets,np.zeros((16,16),bool),config)
    assert not result["expanded_labels"].any() and result["audit"]["candidate_peaks"] == 0
    owners = np.zeros((32,32),np.int32);owners[4:28,8:24] = 1
    a = predict(owners,(owners>0).astype(float),config)
    b = predict(owners,(owners>0).astype(float),config)
    assert a["centers"] == b["centers"] and a["audit"] == b["audit"]
    assert np.array_equal(a["expanded_labels"],b["expanded_labels"])


def test_exact_endpoint_distance_ties_keep_first_center():
    ids, distance = _nearest_centers(np.array([[5.]]),np.array([[5.]]),
                                    [(1.,5,3),(1.,5,7)],np.array([[True]]))
    assert ids.item() == 1 and distance.item() == 4


@pytest.mark.parametrize("problem",["strict","nan","shape","canvas","probability","offset","growth","support","distance"])
def test_invalid_inputs_and_strict_mode_fail(config,problem):
    p = np.zeros((16,16));offsets = np.zeros((2,16,16));canvas = np.ones((16,16),bool)
    if problem == "strict": config["reproduction_mode"] = "strict"
    if problem == "nan": offsets[0,0,0] = np.nan
    if problem == "shape": offsets = offsets[:1]
    if problem == "canvas": canvas = canvas.astype(np.uint8)
    if problem == "probability": p[0,0] = 1.1
    if problem == "offset": offsets[0,0,0] = 1.1
    if problem == "growth": config["growth_text_threshold"] = .9
    if problem == "support": config["min_center_support"] = True
    if problem == "distance": config["growth_distance"] = -1
    with pytest.raises(ValueError): center_guided_masks(p,offsets,canvas,config)
