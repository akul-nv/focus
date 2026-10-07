# SPDX-License-Identifier: Apache-2.0
"""Tests for FocusController."""

import pytest
import torch

from focus import FocusConfig, FocusController


class TestFocusController:
    def test_default_creation(self):
        ctrl = FocusController()
        assert ctrl.mode == "patch"
        assert ctrl.threshold == 0.3

    def test_creation_with_kwargs(self):
        ctrl = FocusController(threshold=0.5, mode="frame")
        assert ctrl.threshold == 0.5
        assert ctrl.mode == "frame"

    def test_creation_with_config(self):
        config = FocusConfig(threshold=0.2, burst_size=16)
        ctrl = FocusController(config=config)
        assert ctrl.threshold == 0.2
        assert ctrl.burst_size == 16

    def test_threshold_setter(self):
        ctrl = FocusController()
        ctrl.threshold = 0.5
        assert ctrl.threshold == 0.5

    def test_threshold_validation(self):
        ctrl = FocusController()
        with pytest.raises(ValueError):
            ctrl.threshold = -0.1
        with pytest.raises(ValueError):
            ctrl.threshold = 1.5

    def test_mode_setter_resets_state(self):
        ctrl = FocusController()
        emb = torch.randn(64, 128)
        ctrl.process_frame(emb)
        assert ctrl.get_stats()["total_frames"] == 1

        ctrl.mode = "frame"
        assert ctrl.mode == "frame"
        assert ctrl.get_stats()["total_frames"] == 0

    def test_process_frame(self):
        ctrl = FocusController()
        emb = torch.randn(64, 128)
        use, dissim, meta = ctrl.process_frame(emb)
        assert use is True
        assert meta["reason"] == "first_frame"

    def test_paused_captures_all(self):
        ctrl = FocusController(burst_enabled=False)
        emb = torch.randn(64, 128)
        ctrl.process_frame(emb)
        ctrl.paused = True

        for _ in range(5):
            use, _, meta = ctrl.process_frame(emb.clone())
            assert use is True
            assert meta["reason"] == "paused"

    def test_force_capture(self):
        ctrl = FocusController(burst_enabled=False, threshold=0.99)
        emb = torch.randn(64, 128)
        ctrl.process_frame(emb)

        use, _, _ = ctrl.process_frame(emb.clone())
        assert use is False

        ctrl.force_capture()
        use, _, meta = ctrl.process_frame(emb.clone())
        assert use is True
        assert meta["reason"] == "forced_capture"

    def test_update_config(self):
        ctrl = FocusController()
        changes = ctrl.update_config(threshold=0.5, burst_size=16)
        assert changes == {"threshold": 0.3, "burst_size": 8}
        assert ctrl.threshold == 0.5
        assert ctrl.burst_size == 16

    def test_update_config_unknown_param(self):
        ctrl = FocusController()
        with pytest.raises(ValueError, match="Unknown"):
            ctrl.update_config(nonexistent=42)

    def test_callback(self):
        ctrl = FocusController()
        changes = []
        ctrl.add_callback(lambda p, o, n: changes.append((p, o, n)))
        ctrl.threshold = 0.5
        assert changes == [("threshold", 0.3, 0.5)]

    def test_remove_callback(self):
        ctrl = FocusController()
        changes = []

        def cb(p, o, n):
            changes.append((p, o, n))

        ctrl.add_callback(cb)
        ctrl.remove_callback(cb)
        ctrl.threshold = 0.5
        assert changes == []

    def test_context_manager(self):
        with FocusController() as ctrl:
            emb = torch.randn(64, 128)
            ctrl.process_frame(emb)
            assert ctrl.get_stats()["total_frames"] == 1
        assert ctrl.get_stats()["total_frames"] == 0

    def test_reset_keep_config(self):
        ctrl = FocusController(threshold=0.5)
        emb = torch.randn(64, 128)
        ctrl.process_frame(emb)
        ctrl.reset(keep_config=True)
        assert ctrl.threshold == 0.5
        assert ctrl.get_stats()["total_frames"] == 0

    def test_reset_defaults(self):
        ctrl = FocusController(threshold=0.5)
        ctrl.reset(keep_config=False)
        assert ctrl.threshold == 0.3

    def test_get_state(self):
        ctrl = FocusController()
        emb = torch.randn(64, 128)
        ctrl.process_frame(emb)
        state = ctrl.get_state()
        assert "mode" in state
        assert "gating_state" in state
        assert state["total_frames"] == 1

    def test_repr(self):
        ctrl = FocusController(threshold=0.4, mode="frame")
        r = repr(ctrl)
        assert "FocusController" in r
        assert "0.4" in r
        assert "frame" in r
