import pytest

from recognition.contextual_quality_features import (
    explicit_caption_evidence, quoted_caption_features, caption_evidence_features,
    contrastive_transcript_variants, other_image_controls, validate_token_budget)
from scripts.run_yolo_context_quality import caption_controls


def test_attributed_literal_caption_can_expose_partial_word():
    caption = 'a sign with the word petrosains on it'
    evidence = explicit_caption_evidence(caption)
    assert [t['text'] for t in evidence['tokens']] == ['petrosains']
    span = evidence['spans'][0]
    assert caption[span['start']:span['end']] == span['text']
    assert span['ambiguous'] is True
    features = caption_evidence_features(['ROSAINS'], evidence, minimum_edit_similarity=.8)
    assert features['exact_caption_recovery'] == 0
    assert features['can_certify_completeness'] is False


def test_generic_prose_and_contractions_do_not_invent_text():
    for caption in ['a sign with text on it', "there's a sign in the dogs' park"]:
        evidence = explicit_caption_evidence(caption)
        assert evidence['tokens'] == []
        assert caption_evidence_features(['OPEN'], evidence,
            minimum_edit_similarity=.8)['exact_caption_recovery'] is None


def test_duplicate_words_use_one_to_one_caption_evidence():
    f = quoted_caption_features('A sign reads "GO GO STOP".', ['go', 'stop'], minimum_edit_similarity=.8)
    assert f['exact_matches'] == 2 and f['exact_caption_recovery'] == pytest.approx(2/3)
    assert f['unmatched_caption_evidence_tokens'] == 1
    f = quoted_caption_features('A sign reads "GO".', ['go', 'go'], minimum_edit_similarity=.8)
    assert f['exact_matches'] == f['edit_matches'] == 1
    assert f['exact_ocr_support'] == .5


def test_raw_apostrophe_spelling_and_punctuation_preserved():
    f = quoted_caption_features('The sign says "L\'école OPEN!".', ["L'école", 'OPEN'], minimum_edit_similarity=1.)
    assert f['exact_matches'] == 1
    assert f['evidence']['tokens'][1]['text'] == 'OPEN!'


def test_probe_limits_preserve_full_original():
    words = ['W'+str(i) for i in range(15)]
    r = contrastive_transcript_variants(words, max_words=10, max_variants=4, max_characters=1000)
    assert r['variants'][0]['text'] == ' '.join(words)
    assert 'UNAVAILABLE' in r['status'] and len(r['variants']) == 1
    r = contrastive_transcript_variants(words, max_words=20, max_variants=4, max_characters=1000)
    assert len(r['variants']) <= 4 and r['variants'][0]['text'] == ' '.join(words)


def test_negative_controls_skip_identical_text_and_stay_in_train_reference():
    pool = {'a':'a shop', 'b':'a shop', 'c':'a boat', 'd':'a street', 'e':'a park'}
    controls = caption_controls('test:image', 'a shop', pool, 3)
    assert len(controls) == 3
    assert all(c['sample_id'] in pool and c['text'] != 'a shop' for c in controls)
    assert caption_controls('test:image','a shop',pool,3) == controls
    controls = other_image_controls({'a':'OPEN', 'b':'OPEN', 'c':'STOP'},controls_per_image=2)
    assert [r['text'] for r in controls['a']] == ['STOP']


def test_token_overflow_raises_instead_of_silently_truncating():
    with pytest.raises(ValueError, match='refusing truncation'):
        validate_token_budget(['A B C'],token_counter=lambda s:len(s.split()),token_limit=2)
