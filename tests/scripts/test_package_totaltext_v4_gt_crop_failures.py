from scripts.package_totaltext_v4_gt_crop_failures import _select


def test_visual_selection_is_deterministic_and_uses_distinct_images():
    records = [
        {"sample_id": "image:1", "rank": 1, "iou": 0.99},
        {"sample_id": "image:1", "rank": 2, "iou": 0.98},
        {"sample_id": "image:2", "rank": 1, "iou": 0.95},
        {"sample_id": "image:3", "rank": 1, "iou": 0.55},
        {"sample_id": "image:4", "rank": 1, "iou": 0.51},
    ]
    selected = _select(records, per_side=2)
    assert [(row["sample_id"], row["rank"]) for row in selected] == [
        ("image:1", 1), ("image:2", 1), ("image:4", 1), ("image:3", 1)
    ]
