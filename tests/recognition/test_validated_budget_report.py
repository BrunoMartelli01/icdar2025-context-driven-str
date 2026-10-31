import pytest

from scripts.report_validated_budget_variant import verify_decisions


def test_report_enforces_prefallback_boundary_and_exact_unique_quota():
    valid = dict(fallback_ids=['a'], accepted_ids=['b', 'c'],
                 ground_truth_used=False, fallback_outputs_loaded=False)
    verify_decisions(valid, 3, 1)
    for changed in (dict(ground_truth_used=True), dict(fallback_outputs_loaded=True),
                    dict(fallback_ids=['a', 'a']), dict(accepted_ids=['a', 'b'])):
        with pytest.raises(ValueError, match='pre-fallback'):
            verify_decisions(dict(valid, **changed), 3, 1)
