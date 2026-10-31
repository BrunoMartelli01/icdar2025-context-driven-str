from recognition.deepsolo_lexicon import DeepSoloLexicon
from scripts.prepare_ic13_complement import complement
from scripts.prepare_ic13_word_risk import aggregate_risk


def row(name, text, box, confidence):
    return dict(candidate_id=name, raw_text=text, bbox=box, confidence=confidence,
                detector_score=.9)


def test_complement_adds_only_confident_spatially_novel_unet_word():
    dictionary = DeepSoloLexicon(['HELLO', 'WORLD', 'NEW'],
        {'HELLO': 'hello', 'WORLD': 'world', 'NEW': 'new'}, distance_exclusive=1.5)
    yolo = [row('y0', 'HELLO', [0, 0, 10, 10], .8)]
    unet = [row('u0', 'HEL1O', [0, 0, 10, 10], 1.),
            row('u1', 'WORLD', [20, 0, 30, 10], .95),
            row('u2', 'NEW', [40, 0, 50, 10], .99)]
    selected, added = complement(yolo, unet, dictionary, .99, .1)
    assert added == 1
    assert [r['candidate_id'] for r in selected] == ['y0', 'u2']
    assert yolo[0]['raw_text'] == 'HELLO' and unet[2]['raw_text'] == 'NEW'


def test_word_risk_excludes_discarded_duplicate_and_keeps_novel_word():
    raw = [row('a', 'hello', [0, 0, 10, 10], .9),
           row('b', 'HELLO', [0, 0, 10, 10], .8),
           row('c', 'world', [20, 0, 30, 10], .7)]
    selected = [raw[0]]
    values = aggregate_risk(raw, [.8, .7, .6], selected, [100, 100], .5)
    assert len(values) == 12
    assert values[0] == .8
    assert values[4] == 1-.8
    assert values[6] == .6
    assert values[10] == 1-.6
