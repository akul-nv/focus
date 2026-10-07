#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Download-free CPU smoke example: a static scene, a change, and burst capture.

Run after ``pip install -e .`` with ``python examples/basic_usage.py``.
Use ``focus-gate video.mp4`` for a real video and Qwen encoder.
"""

import torch

from focus import FocusController


def main() -> None:
    controller = FocusController(threshold=0.4, burst_size=3, max_gate_frames=10)
    # Orthogonal embeddings stand in for two different semantic scenes.
    scenes = [torch.tensor([[1.0, 0.0]]).repeat(4, 1), torch.tensor([[0.0, 1.0]]).repeat(4, 1)]
    for index in range(16):
        embedding = scenes[index >= 8]
        use, dissimilarity, metadata = controller.process_frame(embedding)
        print(
            f"Frame {index:02d}: {'USE ' if use else 'GATE'} "
            f"dissimilarity={dissimilarity:.2f} {metadata['reason']}"
        )
    stats = controller.get_stats()
    assert stats["captured_frames"] == 6
    assert stats["burst_count"] == 2
    print(f"Captured {stats['captured_frames']}/{stats['total_frames']} frames in 2 bursts.")


if __name__ == "__main__":
    main()
