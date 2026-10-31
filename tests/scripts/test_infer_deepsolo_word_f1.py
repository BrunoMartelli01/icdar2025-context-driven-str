from pathlib import Path

import ast
import pytest

from scripts.infer_deepsolo_word_f1 import raw_predictions, validate_selected_config


def config():
    return dict(schema_version=1, reproduction_mode='approximate', claim_label='approximate reproduction',
                table_5_claim=False, precision='float32', batch_size=1, device='cuda:0',
                inference_floor=.05, resolutions={'native': [1024, 1892]},
                selected_policy=dict(decoder='greedy', same_text_nms_iou=None,
                    resolution='native', score_threshold=.35, recognition_threshold=.925))


@pytest.mark.parametrize('change', [{'reproduction_mode': 'strict'}, {'precision': 'float16'},
                                   {'selected_policy': dict(config()['selected_policy'], resolution='unknown')}])
def test_deployment_does_not_fall_back_to_different_scientific_mode(change):
    with pytest.raises(ValueError):
        validate_selected_config(dict(config(), **change))


def detection(points):
    return dict(source_index=0, text='a', score=.8, geometry=dict(boundary_points=points),
                raw=dict(recognition_class_probabilities=[[1.]+[0.]*37]))


def test_no_gt_inference_keeps_geometry_confidence_and_invalid_output():
    ds = [detection([[1.1, 2.2], [5.1, 2.2], [5.1, 6.2], [1.1, 6.2]])]
    p = raw_predictions(ds, [20, 10], 1)[0]
    assert p['bbox'] == [1, 2, 6, 7] and p['recognition_confidence'] == 1.
    assert p['ignored'] is False and p['text'] == 'a'
    ds[0]['geometry']['boundary_points'] = [[30, 2], [40, 2], [40, 6], [30, 6]]
    p = raw_predictions(ds, [20, 10], 1)[0]
    assert p['bbox'] is None and p['invalid_envelope']['treatment'] == 'retain_as_unmatchable_FP'


def test_legacy_export_and_inference_remain_python38_parseable():
    root = Path(__file__).resolve().parents[2]
    for name in ('scripts/run_deepsolo_word_f1_study.py', 'scripts/infer_deepsolo_word_f1.py',
                 'recognition/deepsolo_word_f1.py'):
        ast.parse((root/name).read_text(), feature_version=(3, 8))
