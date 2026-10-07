# SPDX-License-Identifier: Apache-2.0
"""Tests for FocusConfig validation."""

import pytest

from focus import FocusConfig


class TestFocusConfig:
    def test_defaults(self):
        c = FocusConfig()
        assert c.mode == "patch"
        assert c.threshold == 0.3
        assert c.burst_size == 8
        assert c.burst_enabled is True

    def test_frame_mode(self):
        c = FocusConfig(mode="frame", reference_update_strategy="ema", ema_alpha=0.9)
        assert c.mode == "frame"
        assert c.ema_alpha == 0.9

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode"):
            FocusConfig(mode="invalid")

    def test_threshold_out_of_range(self):
        with pytest.raises(ValueError, match="threshold"):
            FocusConfig(threshold=1.5)
        with pytest.raises(ValueError, match="threshold"):
            FocusConfig(threshold=-0.1)

    def test_invalid_strategy(self):
        with pytest.raises(ValueError, match="reference_update_strategy"):
            FocusConfig(reference_update_strategy="unknown")

    def test_ema_alpha_bounds(self):
        with pytest.raises(ValueError, match="ema_alpha"):
            FocusConfig(ema_alpha=0.0)
        with pytest.raises(ValueError, match="ema_alpha"):
            FocusConfig(ema_alpha=1.5)

    def test_burst_size_minimum(self):
        with pytest.raises(ValueError, match="burst_size"):
            FocusConfig(burst_size=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_gate_frames": 0},
        {"reference_update_interval": 0},
        {"dissimilarity_method": "median"},
    ],
)
def test_invalid_additional_options(kwargs):
    with pytest.raises(ValueError):
        FocusConfig(**kwargs)
