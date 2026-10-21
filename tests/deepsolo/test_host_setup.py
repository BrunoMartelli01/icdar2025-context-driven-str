from __future__ import annotations

from pathlib import Path

import pytest

from scripts.prepare_deepsolo_host import (
    DEFAULT_SPEC,
    MAIN_VENV,
    _build_environment,
    _load_install_spec,
    _partition_install_requirements,
    _validate_paths,
)
from scripts.check_deepsolo_environment import EXPECTED


def test_host_spec_reuses_all_frozen_environment_pins() -> None:
    pip_version, arguments = _load_install_spec(DEFAULT_SPEC)

    assert pip_version == "23.3.2"
    assert arguments[:4] == [
        "--find-links",
        "https://download.pytorch.org/whl/torch_stable.html",
        "--find-links",
        "https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html",
    ]
    assert "torch==1.9.0+cu111" in arguments
    assert "torchvision==0.10.0+cu111" in arguments
    assert "detectron2==0.6" in arguments
    assert "setuptools==59.5.0" in arguments
    assert "wheel==0.37.1" in arguments
    assert "numpy==1.20.0" in arguments
    assert "opencv-python==4.5.5.62" in arguments
    assert EXPECTED["detectron2"] == "0.6+cu111"


def test_legacy_sdist_build_prerequisites_are_explicit_and_pinned() -> None:
    _, arguments = _load_install_spec(DEFAULT_SPEC)
    index_arguments, requirements = _partition_install_requirements(arguments)

    assert index_arguments == arguments[:4]
    assert "setuptools==59.5.0" in requirements
    assert "wheel==0.37.1" in requirements
    assert "numpy==1.20.0" in requirements
    assert "scikit-image==0.15.0" in requirements


def test_host_setup_refuses_main_environment_before_any_mutation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="context-str-main"):
        _validate_paths(
            MAIN_VENV,
            tmp_path / "missing-source",
            tmp_path / "missing-checkpoint",
            tmp_path / "report.json",
        )


def test_host_build_environment_is_explicitly_cuda_111_and_sm86() -> None:
    environment = _build_environment(Path("/usr/local/cuda-11.1"))

    assert environment["CUDA_HOME"] == "/usr/local/cuda-11.1"
    assert environment["TORCH_CUDA_ARCH_LIST"] == "8.6"
    assert environment["FORCE_CUDA"] == "1"
    assert environment["CC"] == "/usr/bin/gcc"
    assert environment["CXX"] == "/usr/bin/g++"
