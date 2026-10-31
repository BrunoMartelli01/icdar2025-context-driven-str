"""GT-blind size admission for cached ICDAR2015 component proposals."""

from math import isfinite

from segmentation.icdar2015_improvements import padding_geometry


def projected_component_area(prediction, image_wh, image_hw):
    """Native foreground-cell area in original-canvas square-pixel units.

    This scales the *component's actual pixel count*, not its bounding box.
    Border cells are not fractionally clipped, so the value is an approximation.
    """
    native_h, native_w = image_hw[0] // 2, image_hw[1] // 2
    if min(native_h, native_w) <= 0 or len(prediction['bbox_native']) != 4:
        raise ValueError('Invalid native geometry')
    area = prediction['component_area']
    if not isinstance(area, int) or area <= 0:
        raise ValueError('Positive integer connected-component pixel area required')
    cw, ch, _, _ = padding_geometry(image_wh, image_hw)
    result = area * (cw / native_w) * (ch / native_h)
    if not isfinite(result):
        raise ValueError('Invalid projected connected-component area')
    return result


def size_admitted(prediction, image_wh, image_hw, *, minimum_area, minimum_height):
    if minimum_area < 0 or minimum_height < 0 or not isfinite(minimum_area) or not isfinite(minimum_height):
        raise ValueError('Invalid explicit component size thresholds')
    x1, y1, x2, y2 = prediction['bbox']
    if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 > image_wh[0] or y2 > image_wh[1]:
        raise ValueError('Invalid unpadded original-coordinate box')
    return (projected_component_area(prediction, image_wh, image_hw) >= minimum_area
            and y2-y1 >= minimum_height)
