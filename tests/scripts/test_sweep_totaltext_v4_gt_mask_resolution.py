import numpy as np
from PIL import Image

from scripts.sweep_totaltext_v4_gt_mask_resolution import (
    _proposals, remap_box, resize_v4_target, scaled_area,
)
from segmentation.transforms import PairedSegmentationTransform, SegmentationTransformConfig


def test_resolution128_matches_qualified_target_transform():
    owner = np.zeros((32, 48), dtype=np.int32)
    owner[5:20, 3:20] = 1
    owner[8:25, 25:42] = 2
    target = Image.fromarray((owner > 0).astype(np.uint8) * 255, mode="L")
    validity = Image.fromarray(np.full(owner.shape, 255, dtype=np.uint8), mode="L")
    separation = Image.fromarray(np.zeros(owner.shape, dtype=np.uint8), mode="L")
    owners = Image.fromarray(owner, mode="I")
    config = SegmentationTransformConfig(
        horizontal_flip_probability=0.0, rotation_probability=0.0,
        rotation_degrees=(0.0, 0.0), image_interpolation="bilinear",
        mask_interpolation="bilinear", image_fill=(0, 0, 0), mask_fill=0,
        normalization="imagenet", resize_geometry="stretch",
        operation_order="flip_rotate_resize", target_binarization_threshold=0.5,
    )
    transform = PairedSegmentationTransform(config, base_seed=123)
    image = Image.new("RGB", target.size, "white")
    _, qualified, _, _ = transform.transform_with_validity_and_separation(
        image, target, validity, separation, owners, 0
    )
    observed = resize_v4_target(target, validity, separation, owners, 128)
    assert np.array_equal(observed, qualified[0].numpy().astype(np.uint8))


def test_area_scales_quadratically_and_remap_uses_exclusive_edges():
    assert [scaled_area(r) for r in (128, 256, 512, 1024)] == [150, 600, 2400, 9600]
    assert remap_box([1, 2, 128, 256], 256, (100, 75)) == [0, 0, 38, 100]
    assert remap_box([0, 0, 1024, 1024], 1024, (100, 75)) == [0, 0, 75, 100]


def test_no_filter_keeps_small_components_and_more_than_ten():
    mask = np.zeros((128, 128), dtype=np.uint8)
    for index in range(12):
        mask[5, 5 + 5 * index] = 1
    image = np.full((128, 128, 3), 255, dtype=np.uint8)
    audit, proposals = _proposals(mask, 128, image, filter_mode="none")
    assert audit["raw_external_contours"] == 12
    assert audit["candidates_before_cap"] == 12
    assert len(proposals) == 12
    assert all(row["bbox_area_mask"] == 1 for row in proposals)
    _, filtered = _proposals(mask, 128, image, filter_mode="frozen")
    assert filtered == []
