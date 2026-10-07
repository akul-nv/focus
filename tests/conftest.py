# SPDX-License-Identifier: Apache-2.0
"""Shared pytest fixtures for focus tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Ensure the package is importable when running tests from the repo root
# without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from focus import FocusConfig  # noqa: E402


@pytest.fixture
def default_config() -> FocusConfig:
    """Default FocusConfig with patch mode."""
    return FocusConfig()


@pytest.fixture
def frame_config() -> FocusConfig:
    """FocusConfig in frame mode with replace strategy."""
    return FocusConfig(mode="frame", reference_update_strategy="replace")


@pytest.fixture
def random_embedding() -> torch.Tensor:
    """A single random embedding (64 patches, 128-dim)."""
    return torch.randn(64, 128)


@pytest.fixture
def make_embedding():
    """Factory for embeddings with controllable similarity."""

    def _make(base: torch.Tensor, noise_scale: float = 0.0) -> torch.Tensor:
        if noise_scale == 0.0:
            return base.clone()
        return base + torch.randn_like(base) * noise_scale

    return _make
