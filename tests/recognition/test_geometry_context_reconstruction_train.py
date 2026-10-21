import copy
import json
from pathlib import Path

import pytest

from diagnostics.geometry_context_reconstruction_train import (
    bbox_geometry, inspect_generation, interpret_text, make_prompt,
    path_for, text_metrics, validate_config, validate_population, variants,
)
from diagnostics.caption_routing_train import seal_write
from diagnostics.lavis_captioner_candidates import digest, sha


@pytest.fixture
def config():
    return json.loads(Path('configs/approximate/geometry_context_reconstruction_train_v1.json').read_text())


def block(cid, bbox, text='TEST'):
    return {'candidate_id': cid, 'bbox': bbox, 'text': text}


@pytest.fixture
def sample():
    blocks = [block('outer', [0, 0, 20, 10]), block('inner', [5, 2, 10, 8], 'E')]
    return {'sample_id': '100', 'blocks': blocks, 'baseline': 'TEST E',
            'captions': {k: k+' caption' for k in ('instructblip', 'florence2-base-ft', 'blip2-vitl')},
            'geometry': bbox_geometry(blocks, [['outer', 'inner']], 4),
            'gt_text': 'SECRET_TARGET', 'gt_bbox': [999, 999, 1000, 1000]}


def test_nested_geometry(sample):
    p = sample['geometry'][0]
    assert p['B_inside_A'] and not p['A_inside_B']
    assert p['intersection_fraction_B'] == 1
    assert p['intersection_fraction_A'] == .15
    assert p['horizontal_gap_over_mean_height'] == 0
    assert p['vertical_overlap_over_min_height'] == 1
    assert p['center_y_distance_over_mean_height'] == 0


def test_geometry_scale_translation_and_no_text_dependency(sample):
    shifted = copy.deepcopy(sample['blocks'])
    for b in shifted:
        b['bbox'] = [3*x+17 for x in b['bbox']]
        b['text'] = 'ARBITRARY'
    assert bbox_geometry(shifted, [['outer', 'inner']], 4) == sample['geometry']


def test_disjoint_touching_and_lines():
    blocks = [block('a', [0, 0, 10, 10]), block('b', [15, 0, 25, 10]),
              block('c', [25, 10, 35, 20])]
    p = bbox_geometry(blocks, [['a', 'b'], ['c']], 4)
    assert p[0]['horizontal_gap_over_mean_height'] == .5
    assert p[0]['A_strictly_left_of_B'] and p[0]['same_frozen_line']
    assert p[2]['vertical_overlap_over_min_height'] == 0
    assert p[2]['horizontal_gap_over_mean_height'] == 0
    assert not p[2]['same_frozen_line']
    assert bbox_geometry([], [], 4) == []
    assert bbox_geometry(blocks[:1], [['a']], 4) == []


@pytest.mark.parametrize('blocks,lines', [
    ([block('a', [0, 0, 0, 10])], [['a']]),
    ([block('a', [0, 0, float('nan'), 10])], [['a']]),
    ([block('a', [0, 0, 10, 10])], []),
    ([block('a', [0, 0, 10, 10]), block('a', [0, 0, 10, 10])], [['a', 'a']]),
])
def test_invalid_geometry_fails(blocks, lines):
    with pytest.raises(ValueError):
        bbox_geometry(blocks, lines, 4)


def test_eight_conditions_no_gt_no_hidden_caption(sample, config):
    validate_config(config)
    assert len(variants(config)) == 8
    for v in variants(config):
        prompt = make_prompt(sample, v, config)
        assert 'SECRET_TARGET' not in prompt and '999' not in prompt
        data = json.loads(prompt.split('OBSERVATIONS:\n')[1].split('\nTRANSCRIPTION:')[0])
        assert [b['id'] for b in data['observations_in_frozen_reading_order']] == ['outer', 'inner']
        assert ('geometry' in data) == v.startswith('EXPLICIT_GEOMETRY')
        caption = v.split('/')[1]
        if caption == 'none':
            assert 'scene_caption_fallible' not in data
        else:
            assert data['scene_caption_fallible'] == caption+' caption'


@pytest.mark.parametrize('raw,expected', [
    ('Two words\nTHREE', 'Two words THREE'),
    ("(A120) O'NEIL CO-OP", "(A120) O'NEIL CO-OP"),
    ('SURE', 'SURE'), ('Milk Milk', 'Milk Milk'),
])
def test_allwords_not_singleword_parser(raw, expected):
    assert interpret_text(raw, 'BASE') == {'status': 'valid', 'reason': None, 'text': expected}


@pytest.mark.parametrize('raw', ['', '<ABSTAIN>', 'Here is the text: SHOP',
                                   'Transcription: SHOP', '["SHOP"]', '- SHOP\n- OPEN'])
def test_invalid_output_falls_back_without_salvaging(raw):
    result = interpret_text(raw, 'EXACT BASE')
    assert result['status'] == 'baseline_fallback' and result['text'] == 'EXACT BASE'


def test_truncation_never_accepted():
    assert interpret_text('STORE', 'original', False)['text'] == 'original'


def test_matching_casefold_duplicates_and_fuzzy_one_to_one():
    m = text_metrics('SHOP SHOP cat', ['shop', 'CAT', 'other'])
    assert m['matches'] == 2 and m['predicted'] == 3 and m['fuzzy_ge_0_8'] == 2
    assert text_metrics('SHOPS', ['shop', 'shop'])['fuzzy_ge_0_8'] == 1
    assert text_metrics('', ['shop'])['matches'] == 0


def test_train_separation(config):
    config['split'] = 'TEST'
    with pytest.raises(ValueError): validate_config(config)
    with pytest.raises(RuntimeError): validate_population([{'sample_id': '1', 'blocks': []}])
    samples = [{'sample_id': str(i), 'blocks': []} for i in range(100, 329)]
    with pytest.raises(RuntimeError): validate_population(samples)


def test_cache_request_and_parser_integrity(tmp_path, sample, config):
    v = 'BBOX_ONLY/none'; contract = 'contract'
    seal_write(tmp_path/'model.json', {'contract_sha256': contract})
    row = {'contract_sha256': contract, 'sample_id': '100', 'variant': v,
           'prompt_sha256': digest(make_prompt(sample, v, config)),
           'model_metadata_sha256': sha(tmp_path/'model.json'),
           'raw_output': 'TWO WORDS', 'ended': True,
           **interpret_text('TWO WORDS', sample['baseline'])}
    f = path_for(tmp_path, '100', v)
    seal_write(f, row)
    assert inspect_generation(tmp_path, sample, v, config, contract)['text'] == 'TWO WORDS'
    with pytest.raises(RuntimeError): seal_write(f, {**row, 'raw_output': 'changed'})
    changed = copy.deepcopy(sample); changed['blocks'][0]['text'] = 'CHANGED'
    with pytest.raises(RuntimeError): inspect_generation(tmp_path, changed, v, config, contract)

