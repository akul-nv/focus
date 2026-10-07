# SPDX-License-Identifier: Apache-2.0
"""Dynamic runtime controller for FOCUS.

The :class:`FocusController` wraps a gating strategy and exposes every
configuration knob as a Python property, enabling live parameter changes
from an outer control loop (e.g. adjusting threshold based on scene
complexity).
"""

from __future__ import annotations

from typing import Optional

import torch

from .config import FocusConfig
from .gating import create_gating


class FocusController:
    """Dynamic controller for FOCUS.

    Provides a clean interface for runtime modification of gating parameters,
    designed for integration with larger video-processing pipelines.

    Example::

        controller = FocusController(threshold=0.3, mode="patch")
        encoder = Qwen2VLEncoder(device="cuda")

        for frame in video_frames:
            embedding = encoder(frame)
            should_use, dissim, meta = controller.process_frame(embedding)

            if scene_complexity > 0.8:
                controller.threshold = 0.2   # more sensitive
            else:
                controller.threshold = 0.4   # less sensitive

    Register callbacks to react to config changes::

        def on_change(param, old, new):
            print(f"{param}: {old} -> {new}")

        controller.add_callback(on_change)
    """

    def __init__(self, config: Optional[FocusConfig] = None, **kwargs) -> None:
        """Create a FOCUS controller.

        Args:
            config: A :class:`FocusConfig` instance.  If *None*, a default
                config is created from **kwargs**.
            **kwargs: Override fields on *config* (or supply them when
                *config* is *None*).
        """
        if config is None:
            config = FocusConfig(**kwargs)
        elif kwargs:
            config_dict = {
                f.name: getattr(config, f.name) for f in config.__dataclass_fields__.values()
            }
            config_dict.update(kwargs)
            config = FocusConfig(**config_dict)

        self._config = config
        self._gating = create_gating(config)
        self._callbacks: list = []
        self._paused = False
        self._force_next_capture = False

    @property
    def config(self) -> FocusConfig:
        """Current configuration (read-only snapshot)."""
        return self._config

    @property
    def mode(self) -> str:
        return self._config.mode

    @mode.setter
    def mode(self, value: str) -> None:
        """Set gating mode.  **Resets** internal gating state."""
        if value not in ("patch", "frame"):
            raise ValueError("mode must be 'patch' or 'frame'")
        if value != self._config.mode:
            old = self._config.mode
            config_dict = {
                f.name: getattr(self._config, f.name)
                for f in self._config.__dataclass_fields__.values()
            }
            config_dict["mode"] = value
            self._config = FocusConfig(**config_dict)
            self._gating = create_gating(self._config)
            self._notify("mode", old, value)

    @property
    def threshold(self) -> float:
        return self._config.threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if value != self._config.threshold:
            old = self._config.threshold
            self._config.threshold = value
            self._gating.config.threshold = value
            self._notify("threshold", old, value)

    @property
    def burst_enabled(self) -> bool:
        return self._config.burst_enabled

    @burst_enabled.setter
    def burst_enabled(self, value: bool) -> None:
        if value != self._config.burst_enabled:
            old = self._config.burst_enabled
            self._config.burst_enabled = value
            self._gating.config.burst_enabled = value
            self._notify("burst_enabled", old, value)

    @property
    def burst_size(self) -> int:
        return self._config.burst_size

    @burst_size.setter
    def burst_size(self, value: int) -> None:
        if value < 1:
            raise ValueError("burst_size must be >= 1")
        if value != self._config.burst_size:
            old = self._config.burst_size
            self._config.burst_size = value
            self._gating.config.burst_size = value
            self._notify("burst_size", old, value)

    @property
    def max_gate_frames(self) -> int:
        return self._config.max_gate_frames

    @max_gate_frames.setter
    def max_gate_frames(self, value: int) -> None:
        if value < 1:
            raise ValueError("max_gate_frames must be >= 1")
        if value != self._config.max_gate_frames:
            old = self._config.max_gate_frames
            self._config.max_gate_frames = value
            self._gating.config.max_gate_frames = value
            self._notify("max_gate_frames", old, value)

    @property
    def ema_alpha(self) -> float:
        return self._config.ema_alpha

    @ema_alpha.setter
    def ema_alpha(self, value: float) -> None:
        if not 0.0 < value <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if value != self._config.ema_alpha:
            old = self._config.ema_alpha
            self._config.ema_alpha = value
            self._gating.config.ema_alpha = value
            self._notify("ema_alpha", old, value)

    @property
    def reference_update_strategy(self) -> str:
        return self._config.reference_update_strategy

    @reference_update_strategy.setter
    def reference_update_strategy(self, value: str) -> None:
        if value not in ("replace", "ema", "periodic"):
            raise ValueError("reference_update_strategy must be 'replace', 'ema', or 'periodic'")
        if value != self._config.reference_update_strategy:
            old = self._config.reference_update_strategy
            self._config.reference_update_strategy = value
            self._gating.config.reference_update_strategy = value
            self._notify("reference_update_strategy", old, value)

    @property
    def reference_update_interval(self) -> int:
        return self._config.reference_update_interval

    @reference_update_interval.setter
    def reference_update_interval(self, value: int) -> None:
        if value < 1:
            raise ValueError("reference_update_interval must be >= 1")
        if value != self._config.reference_update_interval:
            old = self._config.reference_update_interval
            self._config.reference_update_interval = value
            self._gating.config.reference_update_interval = value
            self._notify("reference_update_interval", old, value)

    @property
    def dissimilarity_method(self) -> str:
        return self._config.dissimilarity_method

    @dissimilarity_method.setter
    def dissimilarity_method(self, value: str) -> None:
        if value not in ("mean", "max", "ratio"):
            raise ValueError("dissimilarity_method must be 'mean', 'max', or 'ratio'")
        if value != self._config.dissimilarity_method:
            old = self._config.dissimilarity_method
            self._config.dissimilarity_method = value
            self._gating.config.dissimilarity_method = value
            self._notify("dissimilarity_method", old, value)

    @property
    def paused(self) -> bool:
        """When *True*, all frames are captured (gating disabled)."""
        return self._paused

    @paused.setter
    def paused(self, value: bool) -> None:
        if value != self._paused:
            old = self._paused
            self._paused = value
            self._notify("paused", old, value)

    def update_config(self, **kwargs) -> dict:
        """Update multiple parameters at once.

        Returns:
            Dict mapping changed parameter names to their **previous** values.
        """
        changes: dict = {}
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise ValueError(f"Unknown config parameter: {key}")
            old = getattr(self, key)
            if old != value:
                setattr(self, key, value)
                changes[key] = old
        return changes

    def process_frame(self, embedding: torch.Tensor) -> tuple[bool, float, dict]:
        """Process a frame through the gating logic.

        Args:
            embedding: Patch embeddings from an encoder
                ``(num_patches, hidden_size)``.

        Returns:
            ``(should_use, dissimilarity, metadata)``
        """
        if self._paused:
            self._gating.total_frames += 1
            self._gating.captured_frames += 1
            return (
                True,
                0.0,
                {
                    "mode": self._config.mode,
                    "state": "PAUSED",
                    "reason": "paused",
                    "dissimilarity": 0.0,
                    "threshold": self._config.threshold,
                },
            )

        if self._force_next_capture:
            self._force_next_capture = False
            should_use, dissimilarity, metadata = self._gating.process_frame(embedding)
            if not should_use:
                self._gating.captured_frames += 1
                metadata["reason"] = "forced_capture"
            return True, dissimilarity, metadata

        return self._gating.process_frame(embedding)

    def force_capture(self) -> None:
        """Force the next frame to be captured regardless of dissimilarity."""
        self._force_next_capture = True

    def reset(self, keep_config: bool = True) -> None:
        """Reset gating state.

        Args:
            keep_config: If *False*, resets configuration to defaults too.
        """
        if keep_config:
            self._gating.reset()
        else:
            self._config = FocusConfig()
            self._gating = create_gating(self._config)
        self._paused = False
        self._force_next_capture = False

    def get_stats(self) -> dict:
        """Return current gating statistics."""
        stats = self._gating.get_stats()
        stats["paused"] = self._paused
        stats["controller_config"] = {
            "threshold": self.threshold,
            "burst_enabled": self.burst_enabled,
            "burst_size": self.burst_size,
            "mode": self.mode,
            "dissimilarity_method": self.dissimilarity_method,
        }
        return stats

    def get_state(self) -> dict:
        """Return internal state snapshot (useful for debugging)."""
        state = {
            "mode": self._config.mode,
            "paused": self._paused,
            "force_next_capture": self._force_next_capture,
            "total_frames": self._gating.total_frames,
            "captured_frames": self._gating.captured_frames,
            "gated_frames": self._gating.total_frames - self._gating.captured_frames,
            "gating_state": self._gating.state,
            "burst_remaining": self._gating.burst_remaining,
            "frames_since_burst": self._gating.frames_since_burst,
        }
        if hasattr(self._gating, "frames_since_reference"):
            state["frames_since_reference"] = self._gating.frames_since_reference
        return state

    def add_callback(self, callback) -> None:
        """Register ``callback(param_name, old_value, new_value)``."""
        self._callbacks.append(callback)

    def remove_callback(self, callback) -> None:
        """Unregister a previously registered callback."""
        self._callbacks.remove(callback)

    def clear_callbacks(self) -> None:
        """Remove all registered callbacks."""
        self._callbacks.clear()

    def _notify(self, param: str, old, new) -> None:
        for cb in self._callbacks:
            try:
                cb(param, old, new)
            except Exception as exc:
                print(f"Warning: callback error for {param}: {exc}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.reset()
        return False

    def __repr__(self) -> str:
        return (
            f"FocusController(mode={self.mode!r}, threshold={self.threshold}, "
            f"burst_enabled={self.burst_enabled}, burst_size={self.burst_size})"
        )
