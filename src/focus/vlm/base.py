# SPDX-License-Identifier: Apache-2.0
"""Abstract base class for Vision-Language Model pipelines.

To integrate a different locally hosted VLM, subclass
:class:`BaseVLMPipeline` and implement :meth:`encode_frame` and
:meth:`generate_from_frames`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch


class BaseVLMPipeline(ABC):
    """Abstract base for VLM pipelines that combine vision encoding with
    language generation.

    Subclasses must set :attr:`device` and implement both
    :meth:`encode_frame` (for FOCUS embedding extraction) and
    :meth:`generate_from_frames` (for text generation from a burst of
    frames). Gating-only adapters may omit spatial internals. Full FOCUS
    requires decoder attention and writable value-cache access; a hosted
    text/image API without these internals cannot implement calibration.

    Generation must clear ``last_attention_map`` on every call and populate
    it with a finite, normalized CPU map when ``extract_attention`` is true.
    Honor ``spatial_prior`` in the local cache adapter, or raise a clear
    unsupported-feature error rather than silently ignoring it.
    """

    device: str
    """Device the model resides on."""

    last_grid_hw: tuple[int, int] | None = None
    """Spatial grid dimensions ``(grid_h, grid_w)`` from the most recent
    :meth:`encode_frame` call, or *None* if not yet available."""

    last_attention_map: torch.Tensor | None = None
    """Spatial attention map ``(grid_h, grid_w)`` extracted from the most
    recent :meth:`generate_from_frames` call, or *None* if attention
    extraction is not supported / disabled."""

    extract_attention: bool = False
    """When *True*, :meth:`generate_from_frames` should extract an attention
    map and store it in :attr:`last_attention_map`.  Toggled by external
    callers (e.g. the inference queue)."""

    @abstractmethod
    def encode_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Extract patch embeddings from a single frame.

        Args:
            frame: BGR image ``(H, W, 3)`` as a uint8 NumPy array.

        Returns:
            Patch embeddings ``(num_patches, hidden_size)``.
        """
        ...

    @abstractmethod
    def generate_from_frames(
        self,
        frames: list[np.ndarray],
        prompt: str = "Describe what is happening in these video frames briefly.",
        max_new_tokens: int = 100,
        spatial_prior: Optional[torch.Tensor] = None,
        do_sample: bool = False,
    ) -> tuple[str, float]:
        """Generate a text description from a burst of frames.

        Args:
            frames: List of BGR frames (the current burst).
            prompt: Text prompt / instruction.
            max_new_tokens: Maximum tokens to generate.
            spatial_prior: Optional normalised ``(grid_h, grid_w)`` heatmap
                for value-cache calibration during generation.

        Returns:
            ``(generated_text, inference_time_ms)``
        """
        ...
