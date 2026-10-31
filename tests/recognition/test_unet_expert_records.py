import math

import numpy as np
import pytest

from recognition.unet_expert_records import observation


def test_unet_candidate_preserves_real_geometry_and_produces_available_maps():
    row = dict(bbox=[1, 1, 3, 3], candidate_id='sample:1', detector_score=0.9,
        raw_text='ab', token_log_probabilities=[math.log(0.9), math.log(0.8)],
        token_probabilities=[0.9, 0.8], sequence_confidence_native=0.8)
    policy = dict(sequence_threshold=0.0, joint_threshold=0.0, lexical='none',
                  require_alphanumeric=True, same_text_nms_iou=0.5)
    raw, selected, maps = observation([row], np.ones((2, 4, 4), dtype=np.float32),
        [4, 4], policy, dict(probability_threshold=0.5, minimum_component_area=1))
    assert raw[0]['bbox'] == row['bbox'] and selected[0]['source_row'] == 0
    assert 0 < raw[0]['confidence'] <= 1
    assert len(maps) == 14 and maps[-2:] == [1.0, 1.0]
    with pytest.raises(ValueError, match='duplicate'):
        observation([row, row], np.ones((2, 4, 4)), [4, 4], policy,
                    dict(probability_threshold=0.5, minimum_component_area=1))
