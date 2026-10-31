"""Explicit empty-read rejection and raw observation preservation."""
from scripts.finish_icdar13_word_refit import emit


def test_empty_reading_rejection_is_explicit_and_preserves_raw_rows():
    raw=[dict(raw_text='',sequence_confidence_native=.99,token_probabilities=[],token_log_probabilities=[],detector_score=.8,bbox=[0,0,2,2]),
         dict(raw_text='A',sequence_confidence_native=.9,token_probabilities=[.9],token_log_probabilities=[-.10536051565782628],detector_score=.8,bbox=[0,0,2,2])]
    settings=dict(minimum_transcript_probability=.7)
    assert len(emit(raw,settings,dict(reject_empty_transcriptions=False)))==2
    assert [r['raw_text'] for r in emit(raw,settings,dict(reject_empty_transcriptions=True))]==['A']
    assert len(raw)==2 and raw[0]['raw_text']==''
