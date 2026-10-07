# SPDX-License-Identifier: Apache-2.0
"""Configuration dataclass for FOCUS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class FocusConfig:
    """Configuration for FOCUS.

    Supports two gating modes:

    - ``"patch"``: Burst-based capture comparing patches to the **previous** frame
      When a significant change is detected, a burst of *N* consecutive
      frames is captured before returning to the watch state.
    - ``"frame"``: Reference-based gating comparing patches to a **reference** frame
      that is updated according to a configurable strategy (replace / EMA / periodic).

    Both modes support burst capture and forced-include intervals.

    Attributes:
        mode: Gating mode -- ``"patch"`` or ``"frame"``.
        threshold: Dissimilarity threshold in ``[0, 1]``.  Lower values are more
            sensitive (keep more frames); higher values gate more aggressively.
        max_gate_frames: Force a capture after this many consecutive gated frames.
        burst_enabled: Whether to capture a burst of frames on each trigger.
        burst_size: Number of frames per burst.
        reference_update_strategy: How the reference frame is updated in frame mode.
        reference_update_interval: Update interval for the ``"periodic"`` strategy.
        ema_alpha: Weight given to the new frame in EMA updates (``(0, 1]``).
        dissimilarity_method: Aggregation over per-patch dissimilarity scores --
            ``"mean"`` (average), ``"max"`` (most-changed patch), or ``"ratio"``
            (fraction of patches exceeding 0.5 dissimilarity).
    """

    mode: Literal["patch", "frame"] = "patch"

    threshold: float = 0.3
    max_gate_frames: int = 300

    burst_enabled: bool = True
    burst_size: int = 8

    reference_update_strategy: Literal["replace", "ema", "periodic"] = "replace"
    reference_update_interval: int = 10
    ema_alpha: float = 0.7

    dissimilarity_method: Literal["mean", "max", "ratio"] = "mean"

    def __post_init__(self) -> None:
        if self.mode not in ("patch", "frame"):
            raise ValueError(f"mode must be 'patch' or 'frame', got {self.mode!r}")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")
        if self.reference_update_strategy not in ("replace", "ema", "periodic"):
            raise ValueError(
                f"reference_update_strategy must be 'replace', 'ema', or 'periodic', "
                f"got {self.reference_update_strategy!r}"
            )
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1], got {self.ema_alpha}")
        if self.max_gate_frames < 1:
            raise ValueError("max_gate_frames must be >= 1")
        if self.reference_update_interval < 1:
            raise ValueError("reference_update_interval must be >= 1")
        if self.dissimilarity_method not in ("mean", "max", "ratio"):
            raise ValueError("dissimilarity_method must be 'mean', 'max', or 'ratio'")
        if self.burst_size < 1:
            raise ValueError(f"burst_size must be >= 1, got {self.burst_size}")
