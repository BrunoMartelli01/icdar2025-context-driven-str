"""Independent population/filter replay includes empty outputs and all GT."""

from scripts.audit_icdar2015_target70 import independently_emitted,aggregate_metric


def test_independent_policy_retains_empty_raw_output_at_zero_gate():
    empty={"text":"","sequence_confidence":.9,"mean_token_log_probability":None}
    policy={"confidence":"mean_token_confidence","threshold":0.,"minimum_alphanumeric_length":0}
    assert independently_emitted(empty,policy)
    assert not independently_emitted(empty,{**policy,"threshold":.1})


def test_independent_normalization_and_threshold_boundary():
    prediction={"text":"A!ß1","sequence_confidence":.8,"mean_token_log_probability":None}
    policy={"confidence":"sequence_confidence","threshold":.8,"minimum_alphanumeric_length":4}
    assert independently_emitted(prediction,policy) # casefold -> ass1
    assert not independently_emitted(prediction,{**policy,"threshold":.800001})


def test_rejecting_predictions_does_not_remove_original_ground_truth():
    metric=aggregate_metric(2,2,4)
    assert metric=={"tp":2,"fp":0,"fn":2,"precision":1.,"recall":.5,"f1":2/3}
    assert aggregate_metric(0,0,4)["fn"]==4
