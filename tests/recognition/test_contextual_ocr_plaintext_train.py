"""CPU-only protocol tests; no model loading or inference."""
import json
from pathlib import Path

import pytest

from diagnostics.contextual_ocr_plaintext_train import (
    validate_config, select_samples, make_prompt, interpret_text, decode_ids,
)


def config():
    return json.loads(Path('configs/approximate/contextual_ocr_plaintext_train_v2.json').read_text())


def test_full_train_population():
    cfg = config()
    validate_config(cfg)
    samples = [{'sample_id': str(i)} for i in reversed(range(100, 329))]
    assert [s['sample_id'] for s in select_samples(samples, 229)] == [str(i) for i in range(100, 329)]
    with pytest.raises(ValueError):
        select_samples(samples, 20)
    with pytest.raises(ValueError):
        select_samples(samples[:-1], 229)
    cfg['split'] = 'TEST'
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_prompt_excludes_gt_and_keeps_full_optional_caption():
    target = {'id': 0, 'text': 'Bokshop'}
    sample = {'sample_id': '100', 'gt_text': 'SECRET_GT', 'image_path': 'SECRET_PATH',
              'caption': 'A bookshop next to a station.',
              'blocks': [target, {'id': 1, 'text': 'OPEN'}]}
    plain = make_prompt(sample, target, 'ocr_only', config())
    captioned = make_prompt(sample, target, 'ocr_caption', config())
    assert sample['caption'] not in plain and sample['caption'] in captioned
    assert 'Other OCR readings (reading order): OPEN' in plain
    assert 'SECRET_GT' not in captioned and 'SECRET_PATH' not in captioned
    assert plain.count('Bokshop') == 1


@pytest.mark.parametrize('raw', ['', 'one\ntwo', '```word'])
def test_invalid_output_only_preserves_affected_target(raw):
    bad = interpret_text(raw, 'Bokshop')
    good = interpret_text('OPEN', '0PEN')
    assert bad['status'] == 'target_abstention' and bad['text'] == 'Bokshop'
    assert good['status'] == 'valid' and good['text'] == 'OPEN'


def test_no_hidden_punctuation_or_narration_cleanup():
    assert interpret_text('  (A120)  ', '(A')['text'] == '(A120)'
    assert interpret_text('"hello"', 'helo')['text'] == '"hello"'
    assert interpret_text('The answer is hello', 'helo')['text'] == 'The answer is hello'


def test_truncation_empty_target_and_decode_boundaries():
    assert interpret_text('OPEN', '0PEN', True)['text'] == '0PEN'
    assert interpret_text('invented', '')['status'] == 'empty_target_preserved'
    assert decode_ids([0, 7, 1], 'seq2seq', 99) == [7, 1]
    assert decode_ids([2, 3, 7, 1], 'causal', 2) == [7, 1]
