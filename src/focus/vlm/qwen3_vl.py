# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL pipeline -- vision encoding **and** text generation.

Requires ``transformers`` and ``accelerate``
(install with ``pip install focus-vlm[qwen]``).
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from .attention import AttentionMapExtractor
from .base import BaseVLMPipeline
from .focus import SpatialFocusCalibrator


class Qwen3VLPipeline(BaseVLMPipeline):
    """Full Qwen3-VL pipeline for embedding extraction and text generation.

    Loads the complete Qwen3-VL model.  Use :meth:`encode_frame` for FOCUS
    and :meth:`generate_from_frames` for VLM inference on bursts.

    Args:
        model_name: HuggingFace model identifier.
        device: Torch device string.
        attention_layers: How many final decoder layers to use when
            extracting spatial attention maps.
        focus_alpha: Value-cache scale factor for spatial focus calibration.
            Set to ``1.0`` to disable.
        focus_range: ``(low, high)`` prior-value window for token selection.
            Only vision tokens with prior in ``[low, high]`` are scaled.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        attention_layers: int = 4,
        focus_alpha: float = 1.02,
        focus_range: tuple[float, float] = (0.0, 1.0),
        focus_suppress: float = 0.0,
    ) -> None:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.device = device
        self.model_name = model_name

        print(f"Loading Qwen3-VL: {model_name}")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            device_map=device,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.vision_encoder = self.model.visual

        vision_cfg = self.model.config.vision_config
        self.hidden_size: int = vision_cfg.out_hidden_size
        self.patch_size: int = vision_cfg.patch_size
        self.spatial_merge_size: int = vision_cfg.spatial_merge_size
        self.image_token_id: int = self.model.config.image_token_id

        self.attention_extractor = AttentionMapExtractor(
            num_layers=attention_layers,
            enabled=True,
        )
        self.focus_alpha = focus_alpha
        self.focus_range = focus_range
        self.focus_suppress = focus_suppress

        print(f"  hidden_size={self.hidden_size}, patch_size={self.patch_size}")

    @torch.no_grad()
    def encode_frame(self, frame: np.ndarray) -> torch.Tensor:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        messages = [{"role": "user", "content": [{"type": "image", "image": image}]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding=True)

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

        embeddings, _deepstack = self.vision_encoder(pixel_values, grid_thw=grid_thw)
        return embeddings

    @torch.no_grad()
    def generate_from_frames(
        self,
        frames: list[np.ndarray],
        prompt: str = "Describe what is happening in these video frames briefly.",
        max_new_tokens: int = 100,
        spatial_prior: Optional[torch.Tensor] = None,
        do_sample: bool = False,
    ) -> tuple[str, float]:
        if not frames:
            raise ValueError("frames must contain at least one BGR image")
        self.last_attention_map = None
        start = time.perf_counter()

        all_images: list[Image.Image] = []
        for f in frames:
            all_images.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))

        content: list[dict] = []
        for img in all_images:
            content.append({"type": "image", "image": img})

        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=all_images, return_tensors="pt", padding=True)
        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
        }

        calibrator: SpatialFocusCalibrator | None = None
        if spatial_prior is not None and (self.focus_alpha > 1.0 or self.focus_suppress > 0.0):
            calibrator = SpatialFocusCalibrator(
                alpha=self.focus_alpha,
                focus_range=self.focus_range,
                suppress=self.focus_suppress,
            )
            calibrator.apply(
                self.model,
                inputs["input_ids"],
                self.image_token_id,
                spatial_prior,
            )

        try:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
            )
        finally:
            if calibrator is not None:
                calibrator.remove()

        input_len = inputs["input_ids"].shape[1]
        response = self.processor.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)[
            0
        ].strip()

        self.last_attention_map = None
        if self.extract_attention:
            attn_model = getattr(self, "attention_model_override", None) or self.model
            attn_processor = getattr(self, "attention_processor_override", None) or self.processor
            attn_merge = (
                getattr(self, "attention_spatial_merge_override", None) or self.spatial_merge_size
            )
            self.last_attention_map = self._extract_attention_single_frame(
                frames[len(frames) // 2],
                response,
                attn_model,
                attn_processor,
                attn_merge,
            )

        return response, (time.perf_counter() - start) * 1000

    def _extract_attention_single_frame(
        self,
        frame: np.ndarray,
        response_text: str,
        model: torch.nn.Module | None = None,
        processor=None,
        spatial_merge_size: int | None = None,
    ) -> torch.Tensor | None:
        model = model or self.model
        processor = processor or self.processor
        spatial_merge_size = spatial_merge_size or self.spatial_merge_size

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        content: list[dict] = [
            {"type": "image", "image": image},
            {"type": "text", "text": response_text},
        ]
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        attn_inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        device = next(model.parameters()).device
        attn_inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in attn_inputs.items()
        }

        grid_thw = attn_inputs.get("image_grid_thw")
        if grid_thw is None:
            return None

        _, gh, gw = grid_thw[0].tolist()
        grid_h = int(gh) // spatial_merge_size
        grid_w = int(gw) // spatial_merge_size

        image_token_id = getattr(model.config, "image_token_id", self.image_token_id)

        return self.attention_extractor.extract_from_model(
            model=model,
            inputs=attn_inputs,
            image_token_id=image_token_id,
            grid_h=grid_h,
            grid_w=grid_w,
            num_images=1,
        )
