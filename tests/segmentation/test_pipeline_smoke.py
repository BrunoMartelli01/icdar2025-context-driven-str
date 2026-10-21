from __future__ import annotations

from segmentation.smoke import run_smoke


def test_tiny_end_to_end_approximate_segmentation_smoke() -> None:
    report = run_smoke(device_name="cpu", batch_size=1, train_step=False)
    assert report["claim_label"] == "approximate reproduction"
    assert report["result_class"] == "smoke-test result"
    assert report["input_shape"] == [1, 3, 256, 256]
    assert report["target_shape"] == [1, 1, 128, 128]
    assert report["logit_shape"] == [1, 1, 128, 128]
    assert report["parameter_count"] == 48_454_733
    assert set(report["metrics"]) == {"per_image_macro", "global_micro"}
    assert report["localization_audit"]["parameters"] == {
        "min_area": 4,
        "pad": 2,
        "connectivity": 8,
        "tie_break": "top_to_bottom_then_left_to_right",
        "nmax": 10,
    }
    assert report["strict_status"] == "BLOCKED"
    assert report["cuda_peak_allocated_bytes"] is None
