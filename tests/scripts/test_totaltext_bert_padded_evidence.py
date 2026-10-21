import pytest

from scripts.evaluate_totaltext_bert_padded_evidence import config,decisions,expanded_box
from pathlib import Path


def test_padding_uses_height_ceil_and_clamps_all_sides():
    assert expanded_box([3,2,12,9],[15,10],.25,1)==([1,0,14,10],2)
    assert expanded_box([0,0,1,1],[10,10],.25,1)==([0,0,2,2],1)


@pytest.mark.parametrize('fraction,minimum',[(0,1),(-.1,1),(float('nan'),1),(.25,0)])
def test_invalid_padding_fails(fraction,minimum):
    with pytest.raises(ValueError):expanded_box([0,0,10,10],[12,12],fraction,minimum)


def test_original_box_outside_image_fails():
    with pytest.raises(ValueError,match='escapes'):
        expanded_box([0,0,13,10],[12,12],.25,1)


def test_controls_keep_previous_anchor_and_native_word_without_gt():
    cfg=config(Path('configs/approximate/totaltext_bert_padded_evidence_v1.json'))
    b={'candidate_pool':[{'text':'BRIDG'},{'text':'BRIDGE'}],'anchor_index':1,'native_index':1,
       'visual_scores':[{'sum_log_probability':-4},{'sum_log_probability':-1}]}
    scores={c:[{'mean_log_probability':-4},{'mean_log_probability':-1}] for c in ['local_ocr','local_ocr_caption']}
    result=decisions(b,scores,cfg)
    assert len(result)==5 and all(r['text']=='BRIDGE' for r in result.values())
