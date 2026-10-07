# SPDX-License-Identifier: Apache-2.0
"""Abstract base class for vision encoders.

To add support for a new VLM encoder (e.g. CLIP, SigLIP, InternViT),
subclass :class:`BaseEncoder` and implement :meth:`__call__`.

Example::

    class CLIPEncoder(BaseEncoder):
        def __init__(self, device="cuda"):
            self.device = device
            self.hidden_size = 768
            # ... load CLIP model ...

        def __call__(self, frame):
            # ... extract patch embeddings ...
            return embeddings  # (num_patches, hidden_size)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch


class BaseEncoder(ABC):
    """Abstract base for vision encoders that extract patch embeddings.

    Subclasses must set :attr:`device` and :attr:`hidden_size`, and
    implement :meth:`__call__`.
    """

    device: str
    """Device the model resides on (e.g. ``"cuda"`` or ``"cpu"``)."""

    hidden_size: int
    """Dimensionality of each patch embedding vector."""

    last_grid_hw: tuple[int, int] | None = None
    """Spatial grid dimensions ``(grid_h, grid_w)`` from the most recent
    encoding call, or *None* if not yet available.  Updated by subclasses
    that know their spatial layout."""

    @abstractmethod
    def __call__(self, frame: np.ndarray) -> torch.Tensor:
        """Extract patch embeddings from a single video frame.

        Args:
            frame: BGR image as a NumPy array of shape ``(H, W, 3)``
                with dtype ``uint8``.

        Returns:
            Patch embeddings tensor of shape ``(num_patches, hidden_size)``.
        """
        ...
