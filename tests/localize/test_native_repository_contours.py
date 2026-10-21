import json
from types import SimpleNamespace

import numpy as np
import pytest

from localize.native_repository_contours import localize_native_repository_contours, remap_native_box
from localize.repository_contours import RepositoryContourParams, mask_to_blocks
from diagnostics.repository_contours_native_train import geometry, write_once


def test_native_area_padding_and_mapping():
    mask = np.zeros((128, 128), np.uint8)
    mask[10:20, 10:25] = 1  # 150 native pixels
    params = RepositoryContourParams(150, 6, 10)
    result = localize_native_repository_contours(mask, (512, 640), params)
    box = result["retained"][0]
    assert box["native_bbox_area"] == 150
    assert box["native_padded_bbox"] == [4, 4, 31, 26]
    assert box["padded_bbox_original"] == [20, 16, 155, 104]
    assert result["mask_resize"] == "none"
    assert result["native_localization"]["coordinate_convention"] == "native_128_xyxy_exclusive"
    mask[10:20, 24] = 0  # 140: must not pass just because original bbox is large
    assert localize_native_repository_contours(mask, (512, 640), params)["retained"] == []


def test_area_remains_repository_rectangle_area():
    mask = np.zeros((128, 128), np.uint8)
    mask[10:20, 10] = 1
    mask[19, 10:25] = 1
    result = localize_native_repository_contours(mask, (128, 128), RepositoryContourParams(150, 6, 10))
    assert mask.sum() == 24
    assert len(result["retained"]) == 1


def test_clipping_and_fractional_mapping():
    assert remap_native_box([0, 0, 128, 128], (333, 777)) == [0, 0, 777, 333]
    assert remap_native_box([1, 1, 2, 2], (333, 777)) == [6, 2, 13, 6]
    mask = np.zeros((128, 128), np.uint8)
    mask[0:10, 0:15] = 1
    row = localize_native_repository_contours(mask, (512, 640), RepositoryContourParams(150, 6, 10))["retained"][0]
    assert row["native_padded_bbox"] == [0, 0, 21, 16]
    assert row["padded_bbox_original"] == [0, 0, 105, 64]


def test_top10_sort_and_no_mask_mutation():
    mask = np.zeros((128, 128), np.uint8)
    mask[::8, ::8] = 1
    before = mask.copy()
    params = RepositoryContourParams(1, 6, 10)
    result = localize_native_repository_contours(mask, (301, 777), params)
    assert len(result["retained"]) == 10
    assert [b["native_padded_bbox"] for b in result["retained"]] == [list(b) for b in mask_to_blocks(mask, params)]
    np.testing.assert_array_equal(mask, before)


def test_invalid_shapes():
    with pytest.raises(ValueError, match="resizing"):
        localize_native_repository_contours(np.zeros((256, 256)), (512, 640), RepositoryContourParams(150, 6, 10))
    with pytest.raises(ValueError):
        remap_native_box([0, 0, 129, 128], (512, 640))


def test_no_overwrite(tmp_path):
    path = tmp_path / "results.json"
    write_once(path, {"geometry": 1})
    write_once(path, {"geometry": 1})
    with pytest.raises(RuntimeError):
        write_once(path, {"geometry": 2})
    assert json.loads(path.read_text()) == {"geometry": 1}


def test_source_hash_fail_closed(tmp_path):
    source = tmp_path / "old.json"
    source.write_text("{}")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        geometry(SimpleNamespace(source_report=source, source_sha256="wrong"))


def test_ocr_resume_and_reuse_without_real_inference(tmp_path, monkeypatch):
    from PIL import Image
    from diagnostics import repository_contours_native_train as runner
    import recognition.parseq as parseq

    image_path = tmp_path / "image.png"
    Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(image_path)
    oldbox, newbox = [0, 0, 2, 2], [1, 1, 4, 4]
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"per_crop": [{"sample_id": "100", "bbox": oldbox, "text": "OLD"}]}))
    protocol = {"scope": {"split": "TRAIN", "images": 229, "gt_words": 849},
                "source_report_sha256": runner._sha(source)}
    fixture = {"protocol": protocol, "protocol_sha256": runner.digest(protocol),
        "samples": [{"sample_id": "100", "image_path": "image.png", "image_sha256": runner._sha(image_path),
            "localization": {"retained": [{"rank": 1, "padded_bbox_original": oldbox}, {"rank": 2, "padded_bbox_original": newbox}]}}],
        "per_word": [{"sample_id": "100", "matched_crop_rank": 1 if i % 2 == 0 else 2,
                      "text": "OLD" if i % 2 == 0 else "NEW"} for i in range(849)]}
    write_once(tmp_path / "geometry.json", fixture)
    checkpoint, archive = tmp_path / "weights.pt", tmp_path / "runtime.tar.gz"
    checkpoint.write_bytes(b"synthetic weights")
    archive.write_bytes(b"synthetic runtime")
    monkeypatch.setattr(parseq, "PARSEQ_CHECKPOINT_SHA256", runner._sha(checkpoint))
    monkeypatch.setattr(parseq, "PARSEQ_RUNTIME_ARCHIVE_SHA256", runner._sha(archive))
    monkeypatch.setattr(parseq, "verify_parseq_runtime_source", lambda *args, **kwargs: {})
    calls = []

    class FakeRecognizer:
        def __init__(self, config):
            assert config.device == "cpu"
        def recognize(self, crop, *, input_role):
            assert crop.shape == (3, 3, 3)
            assert input_role == "T3_localized_crop"
            calls.append("recognize")
            return SimpleNamespace(raw_text="NEW")
        def release(self):
            calls.append("release")

    monkeypatch.setattr(parseq, "ParseqApproxRecognizer", FakeRecognizer)
    args = SimpleNamespace(source_report=source, source_sha256=runner._sha(source),
        output_dir=tmp_path, data_root=tmp_path, parseq_source=tmp_path,
        parseq_checkpoint=checkpoint, parseq_archive=archive, device="cpu")
    runner.ocr(args)
    assert calls == ["recognize", "release"]
    first_hash = runner._sha(tmp_path / "ocr-results.json")
    result = json.loads((tmp_path / "ocr-results.json").read_text())
    assert result["summary"]["exact"] == 849
    assert result["new_ocr"] == result["reused_ocr"] == 1
    runner.ocr(args)
    assert calls == ["recognize", "release"]
    assert runner._sha(tmp_path / "ocr-results.json") == first_hash
