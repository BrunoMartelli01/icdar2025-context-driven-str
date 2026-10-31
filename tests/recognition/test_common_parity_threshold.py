"""Scientific boundaries of validation-only scalar F1 parity calibration."""
import json
import math

import pytest

from recognition.common_parity_threshold import (
    accepted_ids, aggregate, at_least, calibrate, evaluate, route_observations,
)


def ct(tp, predicted=1, truth=1):
    return dict(tp=tp, fp=predicted-tp, fn=truth-tp, detected_words=predicted,
                gt_words=truth, all_outputs=predicted, ignored_outputs=0)


def test_exhaustive_cut_can_recover_after_an_infeasible_prefix():
    branches = {"a": dict(light=ct(0), deepsolo=ct(1)),
                "b": dict(light=ct(1), deepsolo=ct(0)),
                "c": dict(light=ct(0), deepsolo=ct(1))}
    scores = dict(a=3., b=2., c=1.)
    result = calibrate(scores, branches, scores)
    assert [v["parity"] for v in result["trials"]] == [True, False, True, False]
    assert result["winner"]["accepted_ids"] == ["a", "b"]
    assert result["winner"]["threshold"] == dict(kind="score_ge", value=2.)


def test_equal_scores_cannot_be_split_to_create_a_better_budget():
    branches = {"a": dict(light=ct(1), deepsolo=ct(1)),
                "b": dict(light=ct(0), deepsolo=ct(1))}
    result = calibrate(dict(a=1., b=1.), branches, ["a", "b"])
    assert len(result["trials"]) == 2
    assert result["winner"]["threshold"] == dict(kind="accept_none")
    assert result["winner"]["deepsolo_images"] == 2


def test_ge_boundary_transfers_without_reranking_or_test_labels():
    threshold = dict(kind="score_ge", value=.1)
    scores = dict(a=math.nextafter(.1, -math.inf), b=.1, c=.2)
    assert accepted_ids(scores, scores, threshold) == ["b", "c"]
    assert accepted_ids(scores, ["a", "b"], threshold) == ["b"]
    assert accepted_ids(scores, scores, dict(kind="accept_none")) == []


def test_guard_is_preserved_and_deployment_needs_no_teacher_or_gt():
    obs = {"a": dict(score=4., eligible=False, rows=[]),
           "b": dict(score=2., eligible=True, rows=[dict(raw_text="word")])}
    threshold = dict(kind="score_ge", value=1.)
    out = route_observations(obs, threshold, guard=True)
    assert out["accepted_ids"] == ["b"] and out["fallback_ids"] == ["a"]
    assert (out["deepsolo_images"], out["total_images"], out["deepsolo_percentage"]) == (1, 2, 50.)
    assert route_observations(obs, threshold, guard=False)["deepsolo_images"] == 0


def test_exact_parity_does_not_round_a_small_regression_into_a_pass():
    n = 10**18
    reference = aggregate([ct(n, 2*n, 2*n)])
    worse = aggregate([ct(n-1, 2*n, 2*n)])
    assert worse["f1"] == reference["f1"]  # Float precision alone cannot decide.
    assert not at_least(worse, reference)
    assert at_least(reference, reference)


def test_empty_word_denominators_and_false_positives_remain_in_population():
    branches = dict(a=dict(light=ct(0, 0, 0), deepsolo=ct(0, 1, 0)),
                    b=dict(light=ct(0, 0, 2), deepsolo=ct(1, 1, 2)))
    result = evaluate(branches, ["a"])
    assert result["combined"]["images"] == result["total_images"] == 2
    assert result["combined"]["gt_words"] == 2
    assert result["combined"]["f1"] == 2/3
    assert aggregate([ct(0, 0, 0)])["f1"] == 0.


def test_invalid_scores_counts_and_populations_fail_explicitly():
    branches = dict(a=dict(light=ct(1), deepsolo=ct(1)))
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="finite"):
            calibrate(dict(a=bad), branches, ["a"])
    with pytest.raises(ValueError, match="population mismatch"):
        calibrate(dict(b=1.), branches, ["b"])
    with pytest.raises(ValueError, match="eligible"):
        calibrate(dict(a=1.), branches, ["b"])
    with pytest.raises(ValueError, match="denominator"):
        aggregate([dict(ct(1), fp=1)])
    with pytest.raises(ValueError, match="integers"):
        aggregate([dict(ct(1), tp=1.)])
    with pytest.raises(ValueError, match="ground-truth"):
        evaluate(dict(a=dict(light=ct(1), deepsolo=ct(1, 1, 2))), [])
    with pytest.raises(ValueError, match="Nonempty"):
        accepted_ids({}, [], dict(kind="accept_none"))


def test_calibration_freezes_all_cases_without_reading_test_labels(tmp_path, monkeypatch):
    from scripts import calibrate_common_parity_threshold as runner
    source = tmp_path / "source"
    root = tmp_path / "run"
    root.mkdir()
    runner.write(root / "resolved_configuration.json", {})
    cases = {}
    for name in ("dataset_a", "dataset_b"):
        folder = source / "models" / name
        folder.mkdir(parents=True)
        runner.write(folder / "scores.json", dict(scores=dict(val=1., test=1000.), eligible=dict(val=True, test=True)))
        runner.write(folder / "selection_frozen.json", dict(winner=dict(guard=False)))
        runner.write(folder / "train_branches.json", dict(val=dict(light=ct(1), deepsolo=ct(1))))
        cases[name] = dict(partitions=dict(calibration=["val"], test=["test"]))
    plan = dict(config=dict(cases=list(cases)))
    monkeypatch.setattr(runner, "load_plan", lambda root: (plan, source, cases))
    original_read = runner.read
    def read_without_test(path):
        assert "test" not in path.name
        return original_read(path)
    monkeypatch.setattr(runner, "read", read_without_test)
    runner.calibration_stage(root)
    frozen = json.loads((root / "thresholds_frozen.json").read_text())
    assert set(frozen["cases"]) == set(cases)
    assert frozen["test_labels_opened"] is frozen["test_used_for_selection"] is False
    assert all(v["threshold"] == dict(kind="score_ge", value=1.) for v in frozen["cases"].values())


def test_strict_and_cross_namespace_configs_fail_before_reads():
    from pathlib import Path
    from scripts.calibrate_common_parity_threshold import prepare
    with pytest.raises(ValueError, match="APPROXIMATE"):
        prepare(Path("configs/strict/missing.json"), "strict")
    with pytest.raises(ValueError, match="namespace"):
        prepare(Path("configs/demo/missing.json"), "approximate")


def test_original_source_schema_is_qualified_by_explicit_roles(tmp_path):
    from scripts import calibrate_common_parity_threshold as runner
    folder = tmp_path / "models" / "case"
    folder.mkdir(parents=True)
    runner.write(tmp_path / "audit.json", dict(status="PASS"))
    runner.write(tmp_path / "deployment_qualification.json", dict(
        status="PASS", audit_path="audit.json", audit_sha256=runner.sha(tmp_path / "audit.json"), artifact_sha256={}))
    # The real original common-v1 run predates the validation_policy config key.
    runner.write(tmp_path / "resolved_configuration.json", dict(mode="approximate", smoke=False, config={}))
    parts = dict(fit=["f"], selection=["s"], calibration=["c"], test=["t"])
    keys = dict(f="rgb1", s="rgb2", c="rgb3", t="rgb4")
    runner.write(tmp_path / "cases.json", dict(case=dict(partitions=parts, tie_keys=keys)))
    runner.write(folder / "selection_frozen.json", dict(
        test_used_for_selection=False, fit_ids=["f"], selection_ids=["s"], calibration_ids=["c"]))
    config = dict(source_root=str(tmp_path), cases=["case"], pinned_inputs={})
    assert runner.verify_inputs(config)[0] == tmp_path
    (tmp_path / "cases.json").write_text(json.dumps(dict(case=dict(partitions=parts, tie_keys=dict(keys, c="rgb2")))))
    with pytest.raises(ValueError, match="RGB roles"):
        runner.verify_inputs(config)
