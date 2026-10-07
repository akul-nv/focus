# SPDX-License-Identifier: Apache-2.0
"""Qwen2-VL vision encoder for patch embedding extraction.

Requires the ``transformers`` and ``accelerate`` packages
(install with ``pip install focus-vlm[qwen]``).
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from .base import BaseEncoder

try:
    import cv2
except ImportError as exc:
    raise ImportError("opencv-python is required: pip install opencv-python") from exc


class Qwen2VLEncoder(BaseEncoder):
    """Extract semantic patch embeddings using the Qwen2-VL vision tower.

    This loads the full Qwen2-VL model but only calls the vision encoder
    (``model.visual``).  The language head is not used.

    Args:
        model_name: HuggingFace model identifier.
        device: Torch device string.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.device = device

        print(f"Loading Qwen2-VL encoder: {model_name}")
        self._full_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            device_map=device,
        )
        self._full_model.eval()

        self._model = self._full_model.visual
        self._processor = AutoProcessor.from_pretrained(model_name)

        vision_cfg = self._full_model.config.vision_config
        self.hidden_size: int = vision_cfg.hidden_size
        self.patch_size: int = vision_cfg.patch_size
        self.spatial_merge_size: int = vision_cfg.spatial_merge_size

        print(f"  hidden_size={self.hidden_size}, patch_size={self.patch_size}")

    @torch.no_grad()
    def __call__(self, frame: np.ndarray) -> torch.Tensor:
        """Extract patch embeddings from a BGR frame.

        Args:
            frame: BGR image ``(H, W, 3)`` as a NumPy array.

        Returns:
            Patch embeddings ``(num_patches, hidden_size)``.
        """
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        messages = [{"role": "user", "content": [{"type": "image", "image": image}]}]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text=[text], images=[image], return_tensors="pt", padding=True)

        pixel_values = inputs["pixel_values"].to(self.device)
        grid_thw = inputs.get("image_grid_thw")
        if grid_thw is not None:
            grid_thw = grid_thw.to(self.device)
        else:
            raise ValueError("Qwen processor must return image_grid_thw for spatial alignment")

        _, h, w = grid_thw[0].tolist()
        self.last_grid_hw = (
            int(h) // self.spatial_merge_size,
            int(w) // self.spatial_merge_size,
        )

        return self._model(pixel_values, grid_thw=grid_thw)
