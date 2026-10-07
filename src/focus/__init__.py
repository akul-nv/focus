# SPDX-License-Identifier: Apache-2.0
"""FOCUS -- Online temporal redundancy detection for VLMs.

Quick start::

    from focus import FocusConfig, FocusController

    controller = FocusController(threshold=0.3, mode="patch")
    for frame in video_frames:
        embedding = encoder(frame)
        should_use, dissimilarity, metadata = controller.process_frame(embedding)
        if should_use:
            process(frame)

See :mod:`focus.encoders` and :mod:`focus.vlm`
for vision encoder and VLM pipeline abstractions.
"""

from .config import FocusConfig
from .controller import FocusController
from .gating import (
    FrameLevelGating,
    PatchLevelGating,
    compute_patch_dissimilarity,
    create_gating,
)
from .monitor import PerformanceMonitor

__version__ = "0.1.0"

__all__ = [
    "FocusConfig",
    "FocusController",
    "FrameLevelGating",
    "PatchLevelGating",
    "PerformanceMonitor",
    "compute_patch_dissimilarity",
    "create_gating",
]
