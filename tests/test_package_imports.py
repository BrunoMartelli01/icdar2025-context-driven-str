from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_all_repository_packages_import() -> None:
    for package in (
        "caption", "cli", "deepsolo", "localize", "models", "pipeline", "scoring",
        "segmentation",
    ):
        importlib.import_module(package)


def test_cli_help_does_not_construct_or_load_models() -> None:
    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-m", "cli.main", "--help"],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "--config" in completed.stdout
    assert "--images" in completed.stdout
