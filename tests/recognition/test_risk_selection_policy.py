from scripts.select_risk_admission_gate import selection_key


def test_conservative_tie_prefers_fewer_discarded_words_with_equal_inference_cost():
    common = dict(additional_map_models=0, extra_reader_crops=0, target_name='native')
    def value(score, removed, threshold):
        return dict(common, winner=dict(selection_score=score, removed_words=removed,
            admission_threshold=threshold, heuristic='expected_errors', guard=False))
    unfiltered = value(.90, 0, 0.)
    filtered = value(.905, 100, .9)
    assert min([unfiltered, filtered], key=lambda v: selection_key(v, True)) == unfiltered
    assert min([unfiltered, filtered], key=lambda v: selection_key(v, False)) == filtered
