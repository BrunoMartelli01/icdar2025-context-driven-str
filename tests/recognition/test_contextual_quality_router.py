import pytest
from recognition.contextual_quality_router import (
    FAMILIES, REFERENCE_FIELDS, fit_reference, image_scores,
    lower_quantile, spelling_margin, word_variants,
)


def raw(literal=None):
    return {'eligible': True, 'source_scores': {'all_checks_softcrop': .64,
            'coverage_baseline': .7}, 'signals': {**dict.fromkeys(REFERENCE_FIELDS, .2),
            'literal_recall': literal, 'paper_caption_control': .3}}


def test_no_label_reference_is_exact_declared_population():
    source = {'train': raw(), 'test': raw()}
    source['test']['signals']['mpnet_contrast'] = 100
    reference = fit_reference(source, ['train'])
    assert reference['fields']['mpnet_contrast'] == [.2]
    assert reference['labels_used'] is False
    with pytest.raises(ValueError):
        fit_reference(source, ['train', 'train'])


def test_missing_caption_evidence_neutral_only_for_combination():
    r = raw(); ref = fit_reference({'train': r}, ['train'])
    scores = image_scores(r, ref)
    assert scores['literal_only'] is None
    assert scores['base_literal'] == .64
    assert image_scores(raw(0), ref)['base_literal'] == 0
    assert set(scores) == set(FAMILIES)
    r['eligible'] = False
    assert set(image_scores(r, ref).values()) == {None}


def test_spelling_controls_never_need_lexicon_or_annotations():
    variants = word_variants('PETROSAINS')
    assert variants[0] == 'PETROSAINS'
    assert len(set(variants)) == len(variants) >= 3
    assert spelling_margin([.4, .35, .3]) == pytest.approx(.05)
    assert word_variants('') == []
    assert lower_quantile([0., .5, 1.], .2) == pytest.approx(.2)
    with pytest.raises(ValueError):
        spelling_margin([.2])
