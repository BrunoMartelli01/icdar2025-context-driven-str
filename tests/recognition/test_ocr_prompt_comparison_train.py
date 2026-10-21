import json
from pathlib import Path
import pytest
from diagnostics.ocr_prompt_comparison_train import validate_config, make_prompt, interpret_text


def config():
    return json.loads(Path('configs/approximate/ocr_prompt_comparison_train_v1.json').read_text())


def sample():
    return {'sample_id': '100', 'fold': 0, 'image_sha256': 'target',
            'caption': 'A shop.', 'gt_text': 'FORBIDDEN', 'blocks': [{'id': 0, 'text': 'OPFN'}],
            'demonstrations': [{'source_fold': 1, 'image_sha256': 'example',
                               'target': 'HEL0', 'answer': 'HELO', 'caption': 'A sign.'}]}


def test_predeclared_matrix_and_scope():
    cfg = config()
    validate_config(cfg)
    assert len(cfg['variants']) == 8
    cfg['split'] = 'TEST'
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_only_instruction_changes_for_same_condition():
    cfg, s = config(), sample()
    tails = []
    for name in cfg['prompts']:
        message = make_prompt(s, s['blocks'][0], name+'/ocr_caption', cfg)
        assert message.startswith(cfg['prompts'][name])
        tails.append(message[len(cfg['prompts'][name]):])
        assert 'FORBIDDEN' not in message
        assert s['caption'] in message
    assert len(set(tails)) == 1


def test_no_caption_in_ocr_only_and_no_target_gt():
    cfg, s = config(), sample()
    message = make_prompt(s, s['blocks'][0], 'P1_SHORT/ocr_only', cfg)
    assert s['caption'] not in message and s['demonstrations'][0]['caption'] not in message
    assert 'FORBIDDEN' not in message and 'HEL0' in message and 'HELO' in message
    s['demonstrations'][0]['source_fold'] = 0
    with pytest.raises(RuntimeError):
        make_prompt(s, s['blocks'][0], 'P1_SHORT/ocr_only', cfg)


@pytest.mark.parametrize('raw', ['OPEN NOW', 'OPEN\tNOW', 'OPEN\nNOW', 'OPEN\u00a0NOW'])
def test_uniform_multiword_fallback(raw):
    parsed = interpret_text(raw, 'OPFN')
    assert parsed['text'] == 'OPFN' and parsed['error'] == 'whitespace_output'


def test_no_semantic_or_punctuation_salvage():
    assert interpret_text('TEL', 'TEL:')['text'] == 'TEL'
    assert interpret_text('"OPEN"', 'OPFN')['text'] == '"OPEN"'
    assert interpret_text('OPEN', 'OPFN', True)['text'] == 'OPFN'
