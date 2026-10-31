"""Resolve frozen approximate fusion and budget-gate inference contracts."""

from pathlib import Path

from recognition.budget_deployment_factory import load_budget_pipeline
from recognition.risk_fusion_deployment import RiskFusionPipeline
from scripts.icdar_budget_data import read


def load_fusion_budget_pipeline(root, *, verify_qualification=True):
    plan = read(Path(root)/'resolved_configuration.json')
    if plan.get('risk_fusion', False):
        return RiskFusionPipeline(root, verify_qualification=verify_qualification)
    return load_budget_pipeline(root, verify_qualification=verify_qualification)
