# SPDX-License-Identifier: Apache-2.0
"""Tests for gating strategies using synthetic embeddings."""

import pytest
import torch

from focus import (
    FocusConfig,
    FrameLevelGating,
    PatchLevelGating,
    compute_patch_dissimilarity,
    create_gating,
)


class TestComputePatchDissimilarity:
    def test_identical_embeddings(self):
        emb = torch.randn(64, 128)
        assert compute_patch_dissimilarity(emb, emb, "mean") == pytest.approx(0.0, abs=1e-5)

    def test_orthogonal_embeddings(self):
        a = torch.zeros(1, 128)
        a[0, 0] = 1.0
        b = torch.zeros(1, 128)
        b[0, 1] = 1.0
        assert compute_patch_dissimilarity(a, b, "mean") == pytest.approx(1.0, abs=1e-5)

    def test_methods(self):
        a = torch.randn(64, 128)
        b = a + torch.randn_like(a) * 0.1
        mean_val = compute_patch_dissimilarity(a, b, "mean")
        max_val = compute_patch_dissimilarity(a, b, "max")
        ratio_val = compute_patch_dissimilarity(a, b, "ratio")
        assert 0.0 <= mean_val <= 1.0
        assert max_val >= mean_val
        assert 0.0 <= ratio_val <= 1.0


class TestPatchLevelGating:
    def test_first_frame_always_captured(self, default_config, random_embedding):
        g = PatchLevelGating(default_config)
        use, dissim, meta = g.process_frame(random_embedding)
        assert use is True
        assert meta["reason"] == "first_frame"

    def test_identical_frames_gated(self, random_embedding):
        config = FocusConfig(burst_enabled=False, threshold=0.1)
        g = PatchLevelGating(config)
        g.process_frame(random_embedding)
        use, _, meta = g.process_frame(random_embedding.clone())
        assert use is False
        assert meta["reason"] == "watching"

    def test_different_frames_captured(self, random_embedding):
        config = FocusConfig(burst_enabled=False, threshold=0.01)
        g = PatchLevelGating(config)
        g.process_frame(random_embedding)
        different = torch.randn_like(random_embedding)
        use, _, meta = g.process_frame(different)
        assert use is True
        assert meta["reason"] == "change_detected"

    def test_burst_capture(self):
        config = FocusConfig(burst_enabled=True, burst_size=3, threshold=0.01)
        g = PatchLevelGating(config)

        emb = torch.randn(64, 128)
        results = []
        for i in range(10):
            if i == 0:
                use, _, _ = g.process_frame(emb)
            elif i < 3:
                use, _, _ = g.process_frame(emb + torch.randn_like(emb) * 0.001)
            else:
                use, _, _ = g.process_frame(emb.clone())
            results.append(use)

        assert results[:3] == [True, True, True]

    def test_max_gate_forced_capture(self):
        config = FocusConfig(burst_enabled=False, threshold=0.99, max_gate_frames=5)
        g = PatchLevelGating(config)
        emb = torch.randn(64, 128)
        g.process_frame(emb)

        for _ in range(4):
            use, _, _ = g.process_frame(emb.clone())
            assert use is False

        use, _, meta = g.process_frame(emb.clone())
        assert use is True
        assert meta["reason"] == "max_gate"

    def test_stats(self, default_config, random_embedding):
        g = PatchLevelGating(default_config)
        g.process_frame(random_embedding)
        stats = g.get_stats()
        assert stats["mode"] == "patch"
        assert stats["total_frames"] == 1
        assert stats["captured_frames"] == 1

    def test_reset(self, default_config, random_embedding):
        g = PatchLevelGating(default_config)
        g.process_frame(random_embedding)
        g.reset()
        assert g.total_frames == 0
        assert g.previous_embedding is None


class TestFrameLevelGating:
    def test_first_frame_always_captured(self, frame_config, random_embedding):
        g = FrameLevelGating(frame_config)
        use, _, meta = g.process_frame(random_embedding)
        assert use is True
        assert meta["reason"] == "first_frame"

    def test_similar_frames_gated(self, random_embedding):
        config = FocusConfig(mode="frame", burst_enabled=False, threshold=0.1)
        g = FrameLevelGating(config)
        g.process_frame(random_embedding)
        use, _, meta = g.process_frame(random_embedding.clone())
        assert use is False
        assert meta["reason"] == "similar_to_reference"

    def test_ema_strategy(self, random_embedding):
        config = FocusConfig(
            mode="frame",
            burst_enabled=False,
            threshold=0.01,
            reference_update_strategy="ema",
            ema_alpha=0.5,
        )
        g = FrameLevelGating(config)
        g.process_frame(random_embedding)
        different = torch.randn_like(random_embedding)
        g.process_frame(different)
        assert g.reference_embedding is not None
        assert not torch.allclose(g.reference_embedding, different)

    def test_stats_include_strategy(self, frame_config, random_embedding):
        g = FrameLevelGating(frame_config)
        g.process_frame(random_embedding)
        stats = g.get_stats()
        assert stats["mode"] == "frame"
        assert "reference_strategy" in stats


class TestCreateGating:
    def test_patch_mode(self):
        g = create_gating(FocusConfig(mode="patch"))
        assert isinstance(g, PatchLevelGating)

    def test_frame_mode(self):
        g = create_gating(FocusConfig(mode="frame"))
        assert isinstance(g, FrameLevelGating)


@pytest.mark.parametrize("mode", ["patch", "frame"])
def test_single_frame_bursts_and_repeated_forced_inclusion(mode):
    gating = create_gating(FocusConfig(mode=mode, burst_size=1, max_gate_frames=3))
    embedding = torch.tensor([[1.0, 0.0]])
    kept = [i for i in range(10) if gating.process_frame(embedding)[0]]
    assert kept == [0, 3, 6, 9]
    assert gating.get_stats()["burst_count"] == 4


@pytest.mark.parametrize("mode", ["patch", "frame"])
def test_forced_inclusion_resets_without_bursts(mode):
    gating = create_gating(FocusConfig(mode=mode, burst_enabled=False, max_gate_frames=3))
    embedding = torch.tensor([[1.0, 0.0]])
    kept = [i for i in range(10) if gating.process_frame(embedding)[0]]
    assert kept == [0, 3, 6, 9]
