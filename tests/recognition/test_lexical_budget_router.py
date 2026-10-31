from recognition.deepsolo_lexicon import DeepSoloLexicon
from recognition.lexical_budget_router import correct_rows


def test_correction_is_numeric_and_confidence_guarded_and_preserves_source():
    lex=DeepSoloLexicon(['HELLO','123'],{'HELLO':'Hello','123':'123'},distance_exclusive=1.5)
    rows=[dict(raw_text='HEL1O',confidence=.5),dict(raw_text='HEL1O',confidence=.99),dict(raw_text='124',confidence=.2),dict(raw_text='QQQQ',confidence=.2)]
    p=dict(enabled=True,confidence_cap=.9,protect_numeric=True,unmatched_policy='source_reject')
    out,trace=correct_rows(rows,lex,p)
    assert [r['raw_text'] for r in out]==['Hello','HEL1O','124']
    assert [r['source_row'] for r in out]==[0,1,2]
    assert rows[0]['raw_text']=='HEL1O' and not trace[-1]['accepted']


def test_dictionary_evidence_survives_disabled_edit_policy():
    from recognition.lexical_budget_router import lexical_features
    lex=DeepSoloLexicon(['HELLO'],{'HELLO':'Hello'},distance_exclusive=1.5)
    rows=[dict(raw_text='HELLO',confidence=.99)]
    out,trace=correct_rows(rows,lex,dict(enabled=False,confidence_cap=0.,protect_numeric=True,unmatched_policy='keep_unmatched'))
    assert out[0]['raw_text']=='HELLO' and not trace[0]['changed']
    assert lexical_features(rows,out,trace)[2]==1.
