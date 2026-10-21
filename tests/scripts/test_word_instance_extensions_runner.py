import copy
import json
from types import SimpleNamespace

import pytest
import torch
import numpy as np

from scripts import run_word_instance_extensions as runner


CONFIG = runner.REPO/"configs/approximate/word_instance_extensions_v1.json"


def test_explicit_mode_namespace_and_full_budget():
    c = runner.read_config(CONFIG, "approximate")
    assert c["epochs"] == 20 and c["calibration_epochs"] == [0, 5, 10, 20]
    assert c["sampling_device"] == "cpu"
    with pytest.raises(ValueError, match="APPROXIMATE"): runner.read_config(CONFIG, "strict")


def test_calibration_and_fit_are_disjoint_reproducible_and_never_use_evaluation():
    source = SimpleNamespace(ids=lambda split: ["total_text:"+str(i) for i in (range(20) if split == "training" else range(20, 25))])
    c = {"seed": 10, "calibration_images": 4}
    a = runner.split_ids(source, c); b = runner.split_ids(source, c)
    assert a == b and len(a["fit"]) == 16 and len(a["calibration"]) == 4
    assert not set(a["fit"]) & set(a["calibration"])
    assert not (set(a["fit"]) | set(a["calibration"])) & set(a["evaluation"])


def test_no_prompt_control_changes_only_prompt_channel():
    original = torch.randn(5, 4, 4)
    packet = {"inputs": original.clone()}
    a = runner.source_inputs(packet, "prompt_control", "cpu")[0]
    b = runner.source_inputs(packet, "prompt_instance", "cpu")[0]
    torch.testing.assert_close(a[:4], b[:4], rtol=0, atol=0)
    assert not a[4].any()
    torch.testing.assert_close(packet["inputs"], original, rtol=0, atol=0)


def test_selection_uses_exact_ocr_before_geometry_and_keeps_off_on_ties():
    def report(ocr, hmean, geometry):
        return {"aggregate": {"ocr_correct": ocr, "ocr_hmean": hmean, "geometry_hmean": geometry}}
    assert runner.selection_key(report(10, .2, .3)) > runner.selection_key(report(9, .9, .9))
    assert not runner.selection_key(report(10, .2, .3)) > runner.selection_key(report(10, .2, .3))


def test_original_gt_boxes_are_json_safe_without_changing_coordinates(monkeypatch):
    from segmentation import word_instance_data as data
    monkeypatch.setattr(data, "_train_gt", lambda *a: ([{"bbox": [np.int64(v) for v in (1, 2, 3, 4)], "text": "word"}], "sha"))
    source = data.EvidenceSource.__new__(data.EvidenceSource)
    source.samples = {"x": object()}; source.word_root = CONFIG.parent
    truth = source.gt("x", {"image_wh": np.array([10, 20])})
    assert json.loads(json.dumps(truth)) == [{"bbox": [1, 2, 3, 4], "text": "word"}]


def test_oom_propagates_and_writes_actionable_failure_without_fallback(tmp_path, monkeypatch):
    def oom(*a, **kw): raise torch.OutOfMemoryError("injected OOM, no substitute")
    monkeypatch.setattr(runner, "EvidenceSource", oom)
    args = SimpleNamespace(device="cpu", evidence_root=tmp_path, word_root=tmp_path)
    c = runner.read_config(CONFIG, "approximate")
    with pytest.raises(torch.OutOfMemoryError): runner.run_locked(args, c, tmp_path)
    failed = json.loads((tmp_path/"failure.json").read_text())
    assert failed["oom"] is True and failed["stage"] == "preflight"
