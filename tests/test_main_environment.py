from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def test_detectron2_is_not_installed_in_main_environment() -> None:
    assert importlib.util.find_spec("detectron2") is None


def test_deepsolo_is_not_importable_in_main_environment() -> None:
    assert importlib.util.find_spec("adet") is None


def test_main_cuda_smoke_validation_passes() -> None:
    assert Path(sys.prefix).name == "context-str-main"
    assert torch.cuda.is_available()
    assert "RTX 3070" in torch.cuda.get_device_name(0)
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")
    right = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="cuda")
    assert (left @ right).cpu().tolist() == [[19.0, 22.0], [43.0, 50.0]]


def test_captured_final_environment_validation_is_successful() -> None:
    report = json.loads((ROOT / "environments" / "main" / "validation.json").read_text())
    assert report["status"] == "passed"
    assert report["cuda"]["available"] is True
    assert report["parseq"]["strict_load"] is True
    assert report["parseq"]["inference"] == "passed"
