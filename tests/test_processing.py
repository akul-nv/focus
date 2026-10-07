# SPDX-License-Identifier: Apache-2.0
"""End-to-end streaming lifecycle using model-free video and VLM adapters."""

import numpy as np
import pytest
import torch

from focus import FocusConfig
from focus.encoders import BaseEncoder
from focus.video import processing
from focus.vlm import BaseVLMPipeline


class ToyEncoder(BaseEncoder):
    device = "cpu"
    hidden_size = 2
    last_grid_hw = (2, 2)

    def __call__(self, frame):
        value = float(frame[0, 0, 0])
        return torch.tensor([[1.0, value], [1.0, value * 2], [1.0, value * 3], [1.0, value * 4]])


class ToyVLM(BaseVLMPipeline):
    device = "cpu"
    last_grid_hw = (2, 2)

    def __init__(self):
        self.calls = []

    def encode_frame(self, frame):
        return ToyEncoder()(frame)

    def generate_from_frames(self, frames, prompt, max_new_tokens, spatial_prior, do_sample):
        self.calls.append((len(frames), spatial_prior, prompt))
        if self.extract_attention:
            self.last_attention_map = torch.tensor([[0.0, 0.2], [0.5, 1.0]])
        return f"observation {len(self.calls)}", 1.0


@pytest.fixture
def fake_video(monkeypatch):
    frames = iter([np.full((32, 32, 3), i, dtype=np.uint8) for i in range(5)])

    class Capture:
        released = False

        def isOpened(self):
            return True

        def get(self, prop):
            return {
                processing.cv2.CAP_PROP_FPS: 24,
                processing.cv2.CAP_PROP_FRAME_COUNT: 5,
                processing.cv2.CAP_PROP_FRAME_WIDTH: 32,
                processing.cv2.CAP_PROP_FRAME_HEIGHT: 32,
            }[prop]

        def read(self):
            frame = next(frames, None)
            return frame is not None, frame

        def release(self):
            self.released = True

    class Writer:
        released = False

        def isOpened(self):
            return True

        def write(self, frame):
            pass

        def release(self):
            self.released = True

    capture, writer = Capture(), Writer()
    monkeypatch.setattr(processing.cv2, "VideoCapture", lambda _: capture)
    monkeypatch.setattr(processing.cv2, "VideoWriter", lambda *args: writer)
    return capture, writer


def test_sequential_feedback_keeps_tail_and_supports_callable_encoder(fake_video, tmp_path):
    vlm = ToyVLM()
    result = processing.process_video_with_vlm(
        "input.mp4",
        str(tmp_path / "output.mp4"),
        FocusConfig(threshold=0, burst_size=2),
        vlm,
        encoder=ToyEncoder(),
        vlm_prompt="Look for the red object.",
        compress=False,
        save_log=False,
    )
    assert [call[0] for call in vlm.calls] == [2, 2, 1]
    assert vlm.calls[0][1] is None
    assert vlm.calls[1][1] is not None
    assert vlm.calls[2][1] is not None
    assert "Look for the red object." in vlm.calls[1][2]
    assert "observation 1" in vlm.calls[1][2]
    assert result["focus"]["total_frames"] == 5
    assert result["focus"]["captured_frames"] == 5
    assert result["vlm"]["total_inferences"] == 3
    assert result["vlm"]["dropped_requests"] == 0
    assert result["vlm_responses"][-1]["frame_end"] == 4
    assert all(handle.released for handle in fake_video)


def test_model_failure_propagates_and_releases_video(fake_video, tmp_path):
    class BrokenVLM(ToyVLM):
        def generate_from_frames(self, *args, **kwargs):
            raise RuntimeError("model unavailable")

    with pytest.raises(RuntimeError, match="model unavailable"):
        processing.process_video_with_vlm(
            "input.mp4",
            str(tmp_path / "output.mp4"),
            FocusConfig(burst_size=2),
            BrokenVLM(),
            compress=False,
            save_log=False,
        )
    assert all(handle.released for handle in fake_video)


def test_explicit_cli_burst_size_is_preserved(monkeypatch):
    import sys

    from focus.vlm import model_pool

    monkeypatch.setattr(sys, "argv", ["focus-video", "input.mp4", "--burst-size", "8"])
    monkeypatch.setattr(model_pool, "load_vlm_pipeline", lambda *args, **kwargs: ToyVLM())
    seen = {}
    monkeypatch.setattr(processing, "process_video_with_vlm", lambda **kwargs: seen.update(kwargs))
    processing.focus_video_cli()
    assert seen["config"].burst_size == 8
