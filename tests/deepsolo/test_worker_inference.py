from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import deepsolo.worker as worker


class IdentityTransform:
    def apply_image(self, image):
        return image


class ResizeShortestEdge:
    def __init__(self, short_edges, max_size):
        self.short_edges = short_edges
        self.max_size = max_size

    def get_transform(self, image):
        return IdentityTransform()


def install_fake_detectron2_transforms(monkeypatch, transforms):
    detectron2 = types.ModuleType("detectron2")
    data = types.ModuleType("detectron2.data")
    detectron2.data = data
    data.transforms = transforms
    monkeypatch.setitem(sys.modules, "detectron2", detectron2)
    monkeypatch.setitem(sys.modules, "detectron2.data", data)
    monkeypatch.setitem(sys.modules, "detectron2.data.transforms", transforms)


def test_resnet_prediction_path_never_imports_adet_data(monkeypatch):
    transforms = types.ModuleType("detectron2.data.transforms")
    transforms.ResizeShortestEdge = ResizeShortestEdge
    install_fake_detectron2_transforms(monkeypatch, transforms)
    imported = []
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        imported.append(name)
        if name == "adet.data" or name.startswith("adet.data."):
            raise AssertionError("ResNet-50 inference must not import adet.data")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    cfg = type("Cfg", (), {
        "INPUT": type("Input", (), {
            "FORMAT": "BGR", "MIN_SIZE_TEST": 32, "MAX_SIZE_TEST": 64,
        })(),
    })()
    received = {}

    def model(batch):
        received["shape"] = tuple(batch[0]["image"].shape)
        return [{"instances": "fixture"}]

    result = worker._predict(
        model, cfg, np.zeros((17, 23, 3), dtype=np.uint8), is_vitae=False
    )
    assert result == {"instances": "fixture"}
    assert received["shape"] == (3, 17, 23)
    assert not any(name == "adet.data" or name.startswith("adet.data.") for name in imported)


@pytest.mark.parametrize(
    ("height", "width", "pad_h", "pad_w"),
    ((64, 96, 0, 0), (65, 96, 31, 0), (64, 97, 0, 31), (65, 97, 31, 31)),
)
def test_vitae_padding_matches_frozen_upstream_semantics(
    height: int, width: int, pad_h: int, pad_w: int,
):
    calls = []

    class PadTransform:
        def __init__(self, x0, y0, x1, y1, *, pad_value):
            calls.append((x0, y0, x1, y1, pad_value))
            self.pad_w, self.pad_h, self.pad_value = x1, y1, pad_value

        def apply_image(self, image):
            return np.pad(
                image,
                ((0, self.pad_h), (0, self.pad_w), (0, 0)),
                constant_values=self.pad_value,
            )

    transforms = type("Transforms", (), {"PadTransform": PadTransform})()
    image = np.ones((height, width, 3), dtype=np.uint8)
    padded = worker._apply_frozen_vitae_padding(image, transforms)
    assert calls == [(0, 0, pad_w, pad_h, 0)]
    assert padded.shape == (height + pad_h, width + pad_w, 3)
    np.testing.assert_array_equal(padded[:height, :width], image)
    if pad_h:
        assert not padded[height:, :].any()
    if pad_w:
        assert not padded[:, width:].any()


def test_worker_main_accepts_json_arguments_without_reparsing_by_upstream_cli(
    monkeypatch, tmp_path: Path,
):
    request = tmp_path / "request.json"
    response = tmp_path / "response.json"
    manifest = tmp_path / "runtime.json"
    captured = {}

    def run(request_path, response_path, manifest_path):
        captured["paths"] = (request_path, response_path, manifest_path)
        return 0

    monkeypatch.setattr(worker, "run", run)
    monkeypatch.setattr(sys, "argv", [
        "context_str_m6_deepsolo_json_worker.py",
        "--request", str(request),
        "--response", str(response),
        "--runtime-manifest", str(manifest),
    ])
    assert worker.main() == 0
    assert captured["paths"] == (request, response, manifest)
