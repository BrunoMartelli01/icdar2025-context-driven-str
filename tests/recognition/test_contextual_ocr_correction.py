import copy
import json
from pathlib import Path

import pytest

from diagnostics.contextual_ocr_correction import (
    build_prompt, parse_output, interpret, lexical_counts, word_scores,
    summarize_evaluation, validate_config, load_run,
)
from diagnostics.caption_routing_train import seal_write
from diagnostics.lavis_captioner_candidates import digest


@pytest.fixture
def cfg():
    return json.loads(Path('configs/approximate/contextual_ocr_correction_train_v1.json').read_text())


@pytest.fixture
def sample():
    return {'sample_id': '100', 'caption': 'A store named Bookshop.',
            'gt_text': 'SECRET_GT', 'image_path': 'SECRET_PATH',
            'blocks': [{'id': 0, 'bbox': [0, 0, 10, 10], 'text': 'Bokshop', 'candidate_id': 'v1:100:1'},
                       {'id': 1, 'bbox': [12, 0, 22, 10], 'text': 'OPEN', 'candidate_id': 'v1:100:2'}]}


def output():
    return {'blocks': [{'id': 0, 'text': 'Bookshop', 'action': 'correct'},
                       {'id': 1, 'text': 'OPEN', 'action': 'keep'}]}


def test_prompt_isolation_and_full_caption(cfg, sample):
    for mode in cfg['variants']:
        prompt = build_prompt(sample, mode, cfg)
        assert 'SECRET_GT' not in prompt and 'SECRET_PATH' not in prompt and 'v1:100:1' not in prompt
    assert sample['caption'] not in build_prompt(sample, 'ocr_only', cfg)
    assert sample['caption'] in build_prompt(sample, 'ocr_caption', cfg)
    assert 'Bokshop' not in build_prompt(sample, 'caption_only', cfg)


def test_parser_reorders_ids_without_changing_assignment(sample):
    value = output()
    value['blocks'].reverse()
    assert parse_output(json.dumps(value), sample, 'ocr_only') == output()
    assert parse_output('```json\n'+json.dumps(value)+'\n```', sample, 'ocr_caption') == output()


@pytest.mark.parametrize('mutation', ['unknown', 'duplicate', 'missing', 'keep_changed', 'uncertain_changed',
                                     'empty_correction', 'unchanged_correction', 'extra_key', 'bool_id', 'bad_action'])
def test_parser_rejects_unsafe_schema(sample, mutation):
    value = output()
    row = value['blocks'][0]
    if mutation == 'unknown': row['id'] = 99
    if mutation == 'duplicate': value['blocks'].append(dict(row))
    if mutation == 'missing': value['blocks'].pop()
    if mutation == 'keep_changed': row['action'] = 'keep'
    if mutation == 'uncertain_changed': row['action'] = 'uncertain'
    if mutation == 'empty_correction': row['text'] = ''
    if mutation == 'unchanged_correction': row['text'] = 'Bokshop'
    if mutation == 'extra_key': row['confidence'] = .9
    if mutation == 'bool_id': row['id'] = False
    if mutation == 'bad_action': row['action'] = 'invent'
    with pytest.raises(ValueError): parse_output(json.dumps(value), sample, 'ocr_only')


def test_empty_original_cannot_be_invented(sample):
    sample['blocks'][0]['text'] = ''
    with pytest.raises(ValueError): parse_output(json.dumps(output()), sample, 'ocr_caption')


def test_duplicate_json_keys_rejected(sample):
    with pytest.raises(ValueError): parse_output('{"text":"a","text":"b"}', sample, 'caption_only')


def test_invalid_output_explicitly_abstains(sample):
    result = interpret('narration not JSON', sample, 'ocr_caption')
    assert result['status'] == 'invalid_output_abstention' and result['error']
    assert [b['text'] for b in result['parsed']['blocks']] == ['Bokshop', 'OPEN']
    assert interpret('{"text":"hello"}', sample, 'caption_only', truncated=True)['parsed'] == {'text': ''}


def test_empty_image_blocks_valid(sample):
    sample['blocks'] = []
    assert parse_output('{"blocks":[]}', sample, 'ocr_only') == {'blocks': []}


def test_lexical_duplicate_cap_and_surplus():
    result = lexical_counts(['Cat cat cat DOG extra'], ['cat', 'dog'])
    assert result == {'matches': 2, 'predicted': 5, 'gt': 2, 'surplus': 3, 'missing': 0}


def test_exact_recovery_damage_and_missing_keep_denominator(sample):
    gt = [{'sample_id': '100', 'gt_index': 0, 'crop_bbox': [0, 0, 10, 10], 'available': True,
           'parseq_text': 'Bokshop', 'gt_text': 'Bookshop', 'exact': False},
          {'sample_id': '100', 'gt_index': 1, 'crop_bbox': [12, 0, 22, 10], 'available': True,
           'parseq_text': 'OPEN', 'gt_text': 'open', 'exact': True},
          {'sample_id': '100', 'gt_index': 2, 'crop_bbox': None, 'available': False,
           'parseq_text': None, 'gt_text': 'missed', 'exact': False}]
    rows = word_scores(sample, gt, {0: 'Bookshop', 1: 'CLOSED'})
    summary = summarize_evaluation(rows, [lexical_counts(['Bookshop CLOSED'], ['Bookshop', 'open', 'missed'])], [])
    m = summary['historical_gt_association']
    assert (m['gt_words'], m['exact'], m['corrected_errors'], m['damaged_correct']) == (3, 1, 1, 1)
    assert rows[2]['after'] is None


def test_no_gt_best_match_reselection_after_correction(sample):
    gt = [{'sample_id': '100', 'gt_index': 0, 'crop_bbox': [0, 0, 10, 10], 'available': True,
           'parseq_text': 'Bokshop', 'gt_text': 'OPEN', 'exact': False}]
    assert not word_scores(sample, gt, {0: 'wrong', 1: 'OPEN'})[0]['exact']


def test_training_and_test_and_mutable_revision_fail(cfg):
    validate_config(cfg)
    for key in ('training', 'test_loaded', 'fallback_executed'):
        other = copy.deepcopy(cfg); other[key] = True
        with pytest.raises(ValueError): validate_config(other)
    other = copy.deepcopy(cfg); other['scope']['split'] = 'TEST'
    with pytest.raises(ValueError): validate_config(other)
    cfg['models']['flan-t5-base']['revision'] = 'main'
    with pytest.raises(ValueError): validate_config(cfg)


def test_sealed_run_resume_and_corruption(tmp_path, cfg):
    samples = [{'sample_id': '100', 'blocks': [], 'caption': 'sign'}]
    protocol = {'config': cfg, 'code_sha256': {}, 'samples_sha256': digest(samples)}
    contract = digest(protocol)
    seal_write(tmp_path/'protocol.json', {**protocol, 'contract_sha256': contract})
    value = {'contract_sha256': contract, 'samples': samples, 'samples_sha256': digest(samples)}
    seal_write(tmp_path/'inputs.json', value)
    seal_write(tmp_path/'inputs.json', value)
    assert load_run(tmp_path)[1] == samples
    broken = json.loads((tmp_path/'inputs.json').read_text()); broken['samples'][0]['caption'] = 'tampered'
    (tmp_path/'inputs.json').write_text(json.dumps(broken))
    with pytest.raises(RuntimeError): load_run(tmp_path)
