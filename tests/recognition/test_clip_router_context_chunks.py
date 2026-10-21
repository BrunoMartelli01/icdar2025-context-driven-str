import pytest

from recognition.clip_router_context_chunks import prompt_chunks
from scripts import run_clip_router_literature_v3 as runner


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        assert kwargs == {'add_special_tokens': True, 'truncation': False}
        return {'input_ids': list(range(len(text) + 2))}


def test_space_just_beyond_fitting_prefix_does_not_overflow():
    tokenizer = CharacterTokenizer()
    chunks = prompt_chunks('{text}', 'abc def', tokenizer, 5)
    assert ''.join(chunks) == 'abc def'
    assert chunks[0] == 'abc'
    assert all(len(tokenizer(c, add_special_tokens=True, truncation=False)['input_ids']) <= 5 for c in chunks)


def test_nonmonotonic_token_lengths_validate_whitespace_boundary():
    class Tokenizer:
        def __call__(self, text, **kwargs):
            sizes = {'': 1, 'a': 2, 'a ': 6, 'a b': 4, 'a bc': 6,
                     ' ': 1, ' b': 2, ' bc': 3, 'b': 1, 'bc': 2, 'c': 1}
            return {'input_ids': [0] * sizes[text]}
    chunks = prompt_chunks('{text}', 'a bc', Tokenizer(), 4)
    assert chunks == ['a b', 'c']
    assert ''.join(chunks) == 'a bc'


def test_unicode_suffix_and_all_characters_are_preserved():
    template = 'text: {text}!'
    text = 'ÉCOLE  東京\nTEST  '
    chunks = prompt_chunks(template, text, CharacterTokenizer(), 13)
    assert ''.join(c[len('text: '):-1] for c in chunks) == text
    assert len(chunks) > 1
    assert all(len(CharacterTokenizer()(c, add_special_tokens=True, truncation=False)['input_ids']) <= 13 for c in chunks)


def test_impossible_prefix_fails_without_truncation():
    with pytest.raises(ValueError, match='prefix'):
        prompt_chunks('long prefix: {text}', 'word', CharacterTokenizer(), 5)


def test_literature_callback_restored_after_inference_failure(tmp_path, monkeypatch):
    from types import SimpleNamespace
    spec = {'revision': 'a' * 40, 'weight_file': 'model.safetensors',
            'weight_sha256': 'b' * 64, 'image_size': 224}
    paths = {'models': {}, 'download_root': tmp_path, 'output_root': tmp_path / 'out'}
    monkeypatch.setattr(runner, 'protocol', lambda args: (
        {'models': {'m': spec}}, paths, {}, None, None))
    before = runner.frozen.prompt_chunks

    def fail(args):
        assert runner.frozen.prompt_chunks is prompt_chunks
        raise RuntimeError('inference failure')

    monkeypatch.setattr(runner.frozen, 'run', fail)
    args = SimpleNamespace(models=['m'], download=False, check=False, limit=2)
    with pytest.raises(RuntimeError, match='inference failure'):
        runner.run(args)
    assert runner.frozen.prompt_chunks is before
