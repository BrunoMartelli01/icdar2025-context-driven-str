from scripts.select_adaspot_gap_ocr import selection_key


def candidate(exact, hmean, epoch, threshold, decoder="cut"):
    return {"ocr":{"aggregate":{"exact_casefold_iou_ge_50":exact,"hmean":hmean}},
            "epoch":epoch,"threshold":threshold,"decoder":decoder}


def test_actual_read_word_count_precedes_hmean_and_conservative_ties():
    baseline=candidate(500,.8,0,None)
    better=candidate(501,.79,10,.5)
    assert selection_key(better)>selection_key(baseline)
    assert selection_key(baseline)>selection_key(candidate(500,.79,1,.5))
    assert selection_key(baseline)>selection_key(candidate(500,.8,1,.99))
    assert selection_key(candidate(501,.79,1,.9))>selection_key(candidate(501,.79,1,.5))
    assert selection_key(candidate(501,.79,1,.9))>selection_key(candidate(501,.79,1,.9,"preserve"))
