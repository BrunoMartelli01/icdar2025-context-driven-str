import inspect
import json

import numpy as np
import pytest
from PIL import Image
import torch

from scripts.evaluate_center_guided_word_crops import (
    CONFIG, METHODS, read_config, prepare_recipes, source_report, verify_exports,
)
from scripts.prune_segmentation_checkpoints import sha256_file
from segmentation.word_center_offsets import make_word_center_offsets
from scripts.predict_center_guided_word_crops import image_tensor
from scripts.train_totaltext_v4_image_frequency_ablation import _config, _transform


def test_config_is_approximate_and_does_not_fall_back(tmp_path):
    cfg = read_config(CONFIG,"approximate")
    assert cfg["methods"] == METHODS
    with pytest.raises(ValueError,match="APPROXIMATE"):
        read_config(CONFIG,"strict")
    copied = tmp_path/"copied.json"
    copied.write_text(CONFIG.read_text())
    with pytest.raises(ValueError,match="namespace"):
        read_config(copied,"approximate")


def test_inference_boundary_joins_fragments_without_annotations():
    assert list(inspect.signature(prepare_recipes).parameters) == ["p","offsets","canvas","image_wh","cfg","local_voting"]
    cfg = read_config(CONFIG,"approximate")
    cfg["refinement"].update(smoothing_sigma=1.,min_peak_density=.1,min_center_distance=4,
                             min_center_support=2,assignment_radius=3.)
    owners = np.zeros((32,32),np.int32)
    owners[6:26,3:11] = owners[6:26,19:27] = 1
    canvas = np.ones_like(owners,bool)
    offsets,_ = make_word_center_offsets(owners,canvas)
    voting = {"text_threshold":.65,"smoothing_sigma":1.,"min_peak_density":.1,
              "min_center_distance":4,"assignment_radius":3.}
    recipes,joint = prepare_recipes((owners>0).astype(float),offsets,canvas,(64,32),cfg,voting)
    assert len(recipes["baseline"]["proposals"]) == len(recipes["local_centers"]["proposals"]) == 2
    assert len(recipes["global_centers"]["proposals"]) == 1
    assert recipes["global_centers"]["proposals"][0]["foreground_area"] == 320
    assert joint["audit"]["removed_core_pixels"] == 0


def test_source_checkpoint_mutation_and_incomplete_training_are_rejected(tmp_path):
    checkpoint = tmp_path/"best.pt";checkpoint.write_bytes(b"qualified-cache-fixture")
    contract = {"fixture":True}
    report = {"status":"SUCCESS","result_class":"approximate reproduction","run_contract":contract,
              "checkpoint":{"path":"best.pt","sha256":sha256_file(checkpoint)}}
    (tmp_path/"run_manifest.json").write_text(json.dumps(report))
    (tmp_path/"validation.json").write_text(json.dumps({"run_contract":contract}))
    assert source_report(tmp_path)[0] == report
    checkpoint.write_bytes(b"different")
    with pytest.raises(ValueError,match="changed"):
        source_report(tmp_path)
    checkpoint.write_bytes(b"qualified-cache-fixture")
    report["status"] = "IN_PROGRESS"
    (tmp_path/"run_manifest.json").write_text(json.dumps(report))
    with pytest.raises(ValueError,match="Incomplete"):
        source_report(tmp_path)


def test_completed_export_checksums_are_mandatory(tmp_path):
    crop = tmp_path/"crop.png";crop.write_bytes(b"fixture")
    result = {"branches":{"two_unets":{"baseline":{"per_image":[{"exports":{"crop.png":sha256_file(crop)}}]}}}}
    verify_exports(result,tmp_path)
    crop.write_bytes(b"modified")
    with pytest.raises(ValueError,match="export changed"):
        verify_exports(result,tmp_path)


@pytest.mark.parametrize("size",[(31,17),(17,31),(19,19)])
def test_fresh_rgb_preprocessing_exactly_matches_qualified_validation(size):
    rng = np.random.default_rng(12)
    image = Image.fromarray(rng.integers(0,256,(size[1],size[0],3),dtype=np.uint8))
    cfg = _config()
    qualified,_ = _transform(cfg,"letterbox",train=False)(image,Image.new("L",size),0)
    actual = image_tensor(image,cfg)
    assert torch.equal(actual,qualified)
    assert actual.shape == (3,512,512)


def test_fresh_inference_rejects_unknown_spatial_protocol():
    cfg = _config();cfg["input_size"] = [256,256]
    with pytest.raises(ValueError,match="input512"):
        image_tensor(Image.new("RGB",(17,31)),cfg)
