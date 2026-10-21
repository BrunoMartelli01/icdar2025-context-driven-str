"""Test-local isolation for process-global PyTorch backend settings."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def restore_pytorch_backend_state() -> None:
    """Restore deterministic/cuDNN flags changed by segmentation smoke helpers."""

    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
