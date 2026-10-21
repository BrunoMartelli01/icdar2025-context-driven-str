from __future__ import annotations

import ast
import importlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_segmentation_runtime_sources_parse_as_python38_and_import() -> None:
    paths = [
        ROOT / "models/unet_adv.py",
        ROOT / "models/ag_unet_approx.py",
        ROOT / "segmentation/data.py",
        ROOT / "segmentation/transforms.py",
        ROOT / "segmentation/training.py",
        ROOT / "segmentation/train.py",
        ROOT / "segmentation/config.py",
    ]
    for path in paths:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 8))
    for module in (
        "models.unet_adv",
        "models.ag_unet_approx",
        "segmentation.data",
        "segmentation.transforms",
        "segmentation.training",
        "segmentation.train",
        "segmentation.config",
    ):
        importlib.import_module(module)
