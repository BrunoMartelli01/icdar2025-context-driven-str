"""Instantiate the explicit approximate budget-gate deployment contract."""

from pathlib import Path

from recognition.fixed_expert_deployment import FixedExpertPipeline
from recognition.risk_admission_deployment import RiskAdmissionPipeline
from scripts.icdar_budget_data import read


def load_budget_pipeline(root, *, verify_qualification=True):
    plan = read(Path(root)/'resolved_configuration.json')
    cls = RiskAdmissionPipeline if plan.get('risk_admission', False) else FixedExpertPipeline
    return cls(root, verify_qualification=verify_qualification)
