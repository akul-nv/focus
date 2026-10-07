# SPDX-License-Identifier: Apache-2.0
"""Core gating strategies for FOCUS.

This module contains the two gating strategies (patch-level and frame-level),
the shared dissimilarity computation, and the factory function.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .config import FocusConfig


def compute_patch_dissimilarity_map(
    current: torch.Tensor,
    reference: torch.Tensor,
    method: str = "mean",
) -> tuple[float, torch.Tensor]:
    """Compute per-patch dissimilarity and an aggregated scalar.

    Args:
        current: Current frame patches ``(num_patches, hidden_size)``.
        reference: Reference frame patches ``(num_patches, hidden_size)``.
        method: Aggregation strategy (``"mean"``, ``"max"``, or ``"ratio"``).

    Returns:
        A tuple of ``(scalar_dissimilarity, per_patch_dissimilarity)`` where
        the per-patch tensor has shape ``(num_patches,)`` with values in
        ``[0, 2]`` (cosine dissimilarity).
    """
    if current.ndim != 2 or current.shape != reference.shape or current.numel() == 0:
        raise ValueError("Embeddings must have the same nonempty (patches, hidden_size) shape")
    if method not in ("mean", "max", "ratio"):
        raise ValueError("method must be 'mean', 'max', or 'ratio'")
    similarity = F.cosine_similarity(current.float(), reference.float(), dim=-1)
    dissimilarity = (1 - similarity).clamp(0, 2)

    if method == "max":
        scalar = dissimilarity.max().item()
    elif method == "ratio":
        scalar = (dissimilarity > 0.5).float().mean().item()
    else:
        scalar = dissimilarity.mean().item()

    return scalar, dissimilarity


def compute_patch_dissimilarity(
    current: torch.Tensor,
    reference: torch.Tensor,
    method: str = "mean",
) -> float:
    """Compute dissimilarity between two sets of patch embeddings.

    Convenience wrapper around :func:`compute_patch_dissimilarity_map` that
    returns only the aggregated scalar.

    Args:
        current: Current frame patches ``(num_patches, hidden_size)``.
        reference: Reference frame patches ``(num_patches, hidden_size)``.
        method: Aggregation strategy:
            - ``"mean"`` -- average dissimilarity across patches.
            - ``"max"`` -- maximum single-patch dissimilarity (very sensitive).
            - ``"ratio"`` -- fraction of patches with dissimilarity > 0.5.

    Returns:
        Scalar dissimilarity score in ``[0, 2]`` (0 = identical, 2 = maximally
        dissimilar). In practice values rarely exceed 1.
    """
    scalar, _ = compute_patch_dissimilarity_map(current, reference, method)
    return scalar


class PatchLevelGating:
    """Patch-level FOCUS.

    Operates in two states:

    - **CAPTURE** -- collecting ``burst_size`` consecutive frames.
    - **WATCH** -- gating frames until significant change is detected by
      comparing each frame to the **previous** frame.

    When a change exceeding ``config.threshold`` is detected (or the maximum
    gate duration is reached), the state transitions back to CAPTURE.
    """

    def __init__(self, config: FocusConfig) -> None:
        self.config = config

        self.previous_embedding: Optional[torch.Tensor] = None
        self.state: str = "CAPTURE" if config.burst_enabled else "WATCH"
        self.burst_remaining: int = config.burst_size if config.burst_enabled else 0
        self.frames_since_burst: int = 0

        self.total_frames: int = 0
        self.captured_frames: int = 0
        self.burst_count: int = 0

    def reset(self) -> None:
        """Reset all internal state."""
        self.previous_embedding = None
        self.state = "CAPTURE" if self.config.burst_enabled else "WATCH"
        self.burst_remaining = self.config.burst_size if self.config.burst_enabled else 0
        self.frames_since_burst = 0
        self.total_frames = 0
        self.captured_frames = 0
        self.burst_count = 0

    def process_frame(self, embedding: torch.Tensor) -> tuple[bool, float, dict]:
        """Decide whether to USE or GATE a frame.

        Args:
            embedding: Patch embeddings ``(num_patches, hidden_size)``.

        Returns:
            A tuple of ``(should_use, dissimilarity, metadata)`` where
            metadata includes a ``"dissimilarity_map"`` key holding the
            raw per-patch dissimilarity tensor ``(num_patches,)``.
        """
        self.total_frames += 1
        dissimilarity = 0.0
        dissimilarity_map: torch.Tensor | None = None

        if self.previous_embedding is None:
            self.previous_embedding = embedding.detach().clone()
            self.captured_frames += 1
            if self.config.burst_enabled:
                self.burst_remaining -= 1
                if self.burst_remaining == 0:
                    self.state = "WATCH"
                    self.burst_count += 1
            return True, 0.0, self._metadata("first_frame", 0.0)

        dissimilarity, dissimilarity_map = compute_patch_dissimilarity_map(
            embedding, self.previous_embedding, self.config.dissimilarity_method
        )

        if self.config.burst_enabled and self.state == "CAPTURE":
            should_use = True
            self.captured_frames += 1
            self.burst_remaining -= 1
            reason = (
                f"burst_{self.config.burst_size - self.burst_remaining}/{self.config.burst_size}"
            )
            if self.burst_remaining <= 0:
                self.state = "WATCH"
                self.frames_since_burst = 0
                self.burst_count += 1
        else:
            self.frames_since_burst += 1

            if dissimilarity > self.config.threshold:
                should_use = True
                self.captured_frames += 1
                reason = "change_detected"
                if self.config.burst_enabled:
                    self.state = "CAPTURE"
                    self.burst_remaining = self.config.burst_size - 1
                    if self.burst_remaining == 0:
                        self.state = "WATCH"
                        self.burst_count += 1
            elif self.frames_since_burst >= self.config.max_gate_frames:
                should_use = True
                self.captured_frames += 1
                reason = "max_gate"
                if self.config.burst_enabled:
                    self.state = "CAPTURE"
                    self.burst_remaining = self.config.burst_size - 1
                    if self.burst_remaining == 0:
                        self.state = "WATCH"
                        self.burst_count += 1
            else:
                should_use = False
                reason = "watching"

        if should_use:
            self.frames_since_burst = 0
        self.previous_embedding = embedding.detach().clone()
        meta = self._metadata(reason, dissimilarity)
        meta["dissimilarity_map"] = dissimilarity_map
        return should_use, dissimilarity, meta

    def get_stats(self) -> dict:
        """Return gating statistics."""
        return {
            "mode": "patch",
            "total_frames": self.total_frames,
            "captured_frames": self.captured_frames,
            "gated_frames": self.total_frames - self.captured_frames,
            "gate_ratio": (self.total_frames - self.captured_frames) / max(1, self.total_frames),
            "capture_ratio": self.captured_frames / max(1, self.total_frames),
            "burst_count": self.burst_count,
            "burst_enabled": self.config.burst_enabled,
            "dissimilarity_method": self.config.dissimilarity_method,
        }

    def _metadata(self, reason: str, dissimilarity: float) -> dict:
        return {
            "mode": "patch",
            "state": self.state,
            "reason": reason,
            "dissimilarity": dissimilarity,
            "threshold": self.config.threshold,
            "dissimilarity_method": self.config.dissimilarity_method,
            "burst_remaining": self.burst_remaining,
            "frames_since_burst": self.frames_since_burst,
            "burst_count": self.burst_count,
        }


class FrameLevelGating:
    """Frame-level FOCUS.

    Similar to :class:`PatchLevelGating` but compares each frame to a
    **reference** frame rather than the immediately preceding one.  The
    reference is updated according to ``config.reference_update_strategy``:

    - ``"replace"`` -- overwrite with the latest kept frame.
    - ``"ema"`` -- exponential moving average of patch embeddings.
    - ``"periodic"`` -- overwrite every *N* frames.

    Supports the same burst capture mechanism as patch mode.
    """

    def __init__(self, config: FocusConfig) -> None:
        self.config = config

        self.reference_embedding: Optional[torch.Tensor] = None
        self.frames_since_reference: int = 0
        self.frames_since_forced: int = 0

        self.state: str = "CAPTURE" if config.burst_enabled else "WATCH"
        self.burst_remaining: int = config.burst_size if config.burst_enabled else 0
        self.frames_since_burst: int = 0

        self.total_frames: int = 0
        self.captured_frames: int = 0
        self.burst_count: int = 0

    def reset(self) -> None:
        """Reset all internal state."""
        self.reference_embedding = None
        self.frames_since_reference = 0
        self.frames_since_forced = 0
        self.state = "CAPTURE" if self.config.burst_enabled else "WATCH"
        self.burst_remaining = self.config.burst_size if self.config.burst_enabled else 0
        self.frames_since_burst = 0
        self.total_frames = 0
        self.captured_frames = 0
        self.burst_count = 0

    def process_frame(self, embedding: torch.Tensor) -> tuple[bool, float, dict]:
        """Decide whether to USE or GATE a frame.

        Args:
            embedding: Patch embeddings ``(num_patches, hidden_size)``.

        Returns:
            A tuple of ``(should_use, dissimilarity, metadata)`` where
            metadata includes a ``"dissimilarity_map"`` key holding the
            raw per-patch dissimilarity tensor ``(num_patches,)``.
        """
        self.total_frames += 1

        if self.reference_embedding is None:
            self.reference_embedding = embedding.detach().clone()
            self.captured_frames += 1
            if self.config.burst_enabled:
                self.burst_remaining -= 1
                if self.burst_remaining == 0:
                    self.state = "WATCH"
                    self.burst_count += 1
            return True, 0.0, self._metadata("first_frame", 0.0)

        dissimilarity, dissimilarity_map = compute_patch_dissimilarity_map(
            embedding, self.reference_embedding, self.config.dissimilarity_method
        )

        if self.config.burst_enabled and self.state == "CAPTURE":
            should_use = True
            self.captured_frames += 1
            self.burst_remaining -= 1
            self._update_reference(embedding)
            self.frames_since_forced = 0
            reason = (
                f"burst_{self.config.burst_size - self.burst_remaining}/{self.config.burst_size}"
            )
            if self.burst_remaining <= 0:
                self.state = "WATCH"
                self.frames_since_burst = 0
                self.burst_count += 1
        else:
            self.frames_since_burst += 1
            self.frames_since_forced += 1

            if self.frames_since_forced >= self.config.max_gate_frames:
                should_use = True
                self.captured_frames += 1
                self._update_reference(embedding, force=True)
                self.frames_since_forced = 0
                reason = "forced_inclusion"
                if self.config.burst_enabled:
                    self.state = "CAPTURE"
                    self.burst_remaining = self.config.burst_size - 1
                    if self.burst_remaining == 0:
                        self.state = "WATCH"
                        self.burst_count += 1
            elif dissimilarity > self.config.threshold:
                should_use = True
                self.captured_frames += 1
                self._update_reference(embedding)
                self.frames_since_forced = 0
                reason = "change_detected"
                if self.config.burst_enabled:
                    self.state = "CAPTURE"
                    self.burst_remaining = self.config.burst_size - 1
                    if self.burst_remaining == 0:
                        self.state = "WATCH"
                        self.burst_count += 1
            else:
                should_use = False
                reason = "similar_to_reference"

        meta = self._metadata(reason, dissimilarity)
        meta["dissimilarity_map"] = dissimilarity_map
        return should_use, dissimilarity, meta

    def get_stats(self) -> dict:
        """Return gating statistics."""
        return {
            "mode": "frame",
            "total_frames": self.total_frames,
            "captured_frames": self.captured_frames,
            "gated_frames": self.total_frames - self.captured_frames,
            "gate_ratio": (self.total_frames - self.captured_frames) / max(1, self.total_frames),
            "capture_ratio": self.captured_frames / max(1, self.total_frames),
            "burst_count": self.burst_count,
            "burst_enabled": self.config.burst_enabled,
            "reference_strategy": self.config.reference_update_strategy,
            "dissimilarity_method": self.config.dissimilarity_method,
        }

    def _update_reference(self, embedding: torch.Tensor, force: bool = False) -> None:
        if force:
            self.reference_embedding = embedding.detach().clone()
            self.frames_since_reference = 0
            return

        strategy = self.config.reference_update_strategy
        if strategy == "replace":
            self.reference_embedding = embedding.detach().clone()
            self.frames_since_reference = 0
        elif strategy == "ema":
            alpha = self.config.ema_alpha
            self.reference_embedding = alpha * embedding + (1 - alpha) * self.reference_embedding
            self.frames_since_reference = 0
        elif strategy == "periodic":
            self.frames_since_reference += 1
            if self.frames_since_reference >= self.config.reference_update_interval:
                self.reference_embedding = embedding.detach().clone()
                self.frames_since_reference = 0

    def _metadata(self, reason: str, dissimilarity: float) -> dict:
        return {
            "mode": "frame",
            "state": self.state,
            "reason": reason,
            "dissimilarity": dissimilarity,
            "threshold": self.config.threshold,
            "dissimilarity_method": self.config.dissimilarity_method,
            "strategy": self.config.reference_update_strategy,
            "burst_remaining": self.burst_remaining,
            "frames_since_burst": self.frames_since_burst,
            "burst_count": self.burst_count,
        }


def create_gating(config: FocusConfig) -> PatchLevelGating | FrameLevelGating:
    """Factory: instantiate the gating strategy specified by *config*.

    Args:
        config: A :class:`FocusConfig` whose ``mode`` selects the strategy.

    Returns:
        A :class:`PatchLevelGating` or :class:`FrameLevelGating` instance.
    """
    if config.mode == "patch":
        return PatchLevelGating(config)
    return FrameLevelGating(config)
