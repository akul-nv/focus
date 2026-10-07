# SPDX-License-Identifier: Apache-2.0
"""Tests for SpatialAttentionAnalyzer focus prior functionality."""

from __future__ import annotations

import pytest
import torch

from focus.vlm.spatial import SpatialAttentionAnalyzer


@pytest.fixture
def analyzer() -> SpatialAttentionAnalyzer:
    return SpatialAttentionAnalyzer(enabled=True)


@pytest.fixture
def dissimilarity_map() -> torch.Tensor:
    """A 4x4 dissimilarity map with a hot spot in the upper-left."""
    dmap = torch.zeros(16)
    dmap[:4] = 0.8
    return dmap


@pytest.fixture
def attention_prior() -> torch.Tensor:
    """A 4x4 normalised attention map with focus on upper-left."""
    prior = torch.zeros(4, 4)
    prior[0, 0] = 1.0
    prior[0, 1] = 0.7
    prior[1, 0] = 0.6
    prior[1, 1] = 0.3
    return prior


class TestGetFocusPrior:
    def test_none_without_attention_prior(self, analyzer, dissimilarity_map):
        analyzer.update(dissimilarity_map, grid_h=4, grid_w=4)
        assert analyzer.get_focus_prior() is None

    def test_none_when_disabled(self, dissimilarity_map, attention_prior):
        sa = SpatialAttentionAnalyzer(enabled=False)
        sa.update(dissimilarity_map, grid_h=4, grid_w=4)
        sa.set_attention_prior(attention_prior)
        assert sa.get_focus_prior() is None

    def test_returns_tensor_with_prior(self, analyzer, dissimilarity_map, attention_prior):
        analyzer.update(dissimilarity_map, grid_h=4, grid_w=4)
        analyzer.set_attention_prior(attention_prior)
        prior = analyzer.get_focus_prior()
        assert prior is not None
        assert prior.shape == (4, 4)
        assert prior.min() >= 0.0
        assert prior.max() <= 1.0

    def test_focus_prior_is_cpu(self, analyzer, dissimilarity_map, attention_prior):
        analyzer.update(dissimilarity_map, grid_h=4, grid_w=4)
        analyzer.set_attention_prior(attention_prior)
        prior = analyzer.get_focus_prior()
        assert prior.device == torch.device("cpu")


class TestResetPreservesPrior:
    def test_prior_survives_reset(self, analyzer, dissimilarity_map, attention_prior):
        analyzer.update(dissimilarity_map, grid_h=4, grid_w=4)
        analyzer.set_attention_prior(attention_prior)
        analyzer.reset()
        assert analyzer._attention_prior is not None

    def test_accumulated_cleared_on_reset(self, analyzer, dissimilarity_map, attention_prior):
        analyzer.update(dissimilarity_map, grid_h=4, grid_w=4)
        analyzer.set_attention_prior(attention_prior)
        analyzer.reset()
        assert analyzer._accumulated is None

    def test_focus_prior_after_reset_and_new_update(
        self, analyzer, dissimilarity_map, attention_prior
    ):
        analyzer.update(dissimilarity_map, grid_h=4, grid_w=4)
        analyzer.set_attention_prior(attention_prior)
        analyzer.reset()
        analyzer.update(dissimilarity_map, grid_h=4, grid_w=4)
        prior = analyzer.get_focus_prior()
        assert prior is not None
        assert prior.shape == (4, 4)


class TestCombinedMap:
    def test_without_prior_returns_normalised_dmap(self, analyzer, dissimilarity_map):
        analyzer.update(dissimilarity_map, grid_h=4, grid_w=4)
        combined = analyzer._compute_combined_map()
        assert combined is not None
        assert combined.shape == (4, 4)
        assert combined.max() <= 1.0 + 1e-6
        assert combined.min() >= -1e-6

    def test_with_prior_multiplicative(self, analyzer, dissimilarity_map, attention_prior):
        analyzer.update(dissimilarity_map, grid_h=4, grid_w=4)
        analyzer.set_attention_prior(attention_prior)
        combined = analyzer._compute_combined_map()
        assert combined is not None
        assert combined.shape == (4, 4)

    def test_none_without_accumulation(self, analyzer):
        assert analyzer._compute_combined_map() is None


class TestSummary:
    def test_summary_keys(self, analyzer, dissimilarity_map, attention_prior):
        analyzer.update(dissimilarity_map, grid_h=4, grid_w=4)
        analyzer.set_attention_prior(attention_prior)
        analyzer.get_focus_prior()
        s = analyzer.get_summary()
        assert s["enabled"] is True
        assert s["has_attention_prior"] is True
        assert s["total_updates"] == 1
        assert s["total_focus_priors"] == 1
        assert s["fusion_mode"] == "multiplicative"
