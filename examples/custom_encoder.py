#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Runnable CPU adapter example with deterministic image patch features.

This small pixel encoder illustrates the interface; it is not a semantic
VLM encoder or a paper baseline. Replace its body with your backbone's
patch features, preserving spatial order and a fixed grid between frames.
"""

import numpy as np
import torch
import torch.nn.functional as F

from focus import FocusController
from focus.encoders import BaseEncoder


class PatchColorEncoder(BaseEncoder):
    def __init__(self, grid_size: int = 8):
        self.device = "cpu"
        self.hidden_size = 3
        self.last_grid_hw = (grid_size, grid_size)

    def __call__(self, frame: np.ndarray) -> torch.Tensor:
        pixels = torch.from_numpy(frame.copy()).float().permute(2, 0, 1) / 255
        patches = F.adaptive_avg_pool2d(pixels.unsqueeze(0), self.last_grid_hw)
        return patches[0].permute(1, 2, 0).reshape(-1, self.hidden_size)


def main() -> None:
    encoder = PatchColorEncoder()
    controller = FocusController(threshold=0.4, burst_size=3)
    for index in range(16):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:, :, 0 if index < 8 else 2] = 255
        use, _, metadata = controller.process_frame(encoder(frame))
        print(f"Frame {index:02d}: {'USE ' if use else 'GATE'} {metadata['reason']}")
    assert controller.get_stats()["captured_frames"] == 6


if __name__ == "__main__":
    main()
