import pytest

from scripts.audit_fixed_retention_test import audit_cases


def test_retention_audit_case_subset_must_be_declared_and_unique():
    assert audit_cases(['a', 'b'], None) == ['a', 'b']
    assert audit_cases(['a', 'b'], ['b']) == ['b']
    for requested in ([], ['a', 'a'], ['c']):
        with pytest.raises(ValueError, match='nonempty subset'):
            audit_cases(['a', 'b'], requested)
