import copy
import json

import pytest

from scripts.run_deepsolo_textocr_comparison import model_input_shape, read_shard, validate_response
from scripts.run_clip_high_recall_admission import payload_digest


def response():
    return {'success': True, 'artifact': {'candidate': {'artifact_id': 'new'},
        'checkpoint_sha256': 'checkpoint', 'config_sha256': 'config',
        'load_validation': {'strict_compatibility_check': True, 'missing_keys': [],
                            'unexpected_keys': [], 'incorrect_shapes': []},
        'resolved_inference_threshold': .4,
        'inference_options': {'device': 'cuda:0', 'score_threshold': .4, 'lexicon_type': None}},
        'placement': {'device': 'cuda:0'}}


def test_new_candidate_does_not_require_old_candidates_qualified_digest():
    # New candidates have no globally qualified_config_sha256 until real qualification.
    value = response()
    assert validate_response(value, 'new', 'checkpoint', 'config', .4) == value


@pytest.mark.parametrize('change', ['candidate', 'weight', 'config', 'threshold', 'missing', 'cpu', 'lexicon'])
def test_changed_scientific_identity_is_rejected(change):
    value = copy.deepcopy(response())
    if change == 'candidate': value['artifact']['candidate']['artifact_id'] = 'old'
    if change == 'weight': value['artifact']['checkpoint_sha256'] = 'other'
    if change == 'config': value['artifact']['config_sha256'] = 'other'
    if change == 'threshold': value['artifact']['resolved_inference_threshold'] = .25
    if change == 'missing': value['artifact']['load_validation']['missing_keys'] = ['backbone']
    if change == 'cpu': value['placement']['device'] = 'cpu'
    if change == 'lexicon': value['artifact']['inference_options']['lexicon_type'] = 1
    with pytest.raises(ValueError): validate_response(value, 'new', 'checkpoint', 'config', .4)


def test_resume_binds_image_and_configuration_and_preserves_response(tmp_path):
    path = tmp_path/'sample.json'
    contract = {'sample_id': 'total_text:1', 'image_sha256': 'image', 'plan_sha256': 'plan'}
    value = {'contract': contract, 'response': response()}
    value['payload_sha256'] = payload_digest(value)
    path.write_text(json.dumps(value))
    assert read_shard(path, contract) == value
    with pytest.raises(ValueError): read_shard(path, {**contract, 'image_sha256': 'other'})
    value['response']['artifact']['checkpoint_sha256'] = 'changed'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError): read_shard(path, contract)


def test_native_resize_exif_and_vitae_padding():
    assert model_input_shape([100, 100], 1, 'r50_textocr') == [3, 1000, 1000]
    assert model_input_shape([4000, 1000], 1, 'r50_textocr') == [3, 473, 1892]
    assert model_input_shape([4000, 1000], 6, 'r50_textocr') == [3, 1892, 473]
    assert model_input_shape([4000, 1000], 1, 'vitae_textocr') == [3, 480, 1920]
    assert model_input_shape([100, 100], 1, 'vitae_textocr') == [3, 1024, 1024]
