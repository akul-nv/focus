"""BOLT-style CLIP scoring and inverse-CDF frame selection within each burst.

The sampling function adapts sming256/BOLT/select_frames.py (MIT); see
THIRD_PARTY_NOTICES.md for attribution and the license.

Adapted from the baseline on this repository's ``baselines`` branch. The scoring
and sampling rule follows BOLT (Liu et al., CVPR 2025, arXiv:2503.21483), but this
runner selects within an arriving burst, rather than over the complete video.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch
from PIL import Image


def inverse_transform_sampling(score: np.ndarray, n: int, power: float = -1) -> np.ndarray:
    """Return BOLT inverse-CDF sample indices; duplicate samples are possible.

    The original quantiles ``linspace(1/n, 1-1/n, n)`` are preserved. A one-frame
    request uses the median quantile. Equal scores fall back to uniform sampling.
    """
    score = np.asarray(score, dtype=float)
    if score.ndim != 1 or not score.size or not np.isfinite(score).all():
        raise ValueError("score must be a nonempty, finite, one-dimensional array")
    if n < 1 or (power != -1 and (not np.isfinite(power) or power <= 0)):
        raise ValueError("n must be positive and power must be -1 or positive")
    score = score - score.min()
    if score.max() > 0:
        score = score / score.max()
    if power != -1:
        score = score**power
    if score.sum() == 0:
        return np.linspace(0, len(score) - 1, n, dtype=int)
    quantiles = np.array([0.5]) if n == 1 else np.linspace(1 / n, 1 - 1 / n, n)
    return np.clip(np.searchsorted(np.cumsum(score / score.sum()), quantiles), 0, len(score) - 1)


class CLIPScorer:
    """Optional Hugging Face CLIP scorer, loaded only for the BOLT baseline."""

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", device: str = "cuda"):
        from transformers import CLIPModel, CLIPProcessor

        self.device = torch.device(device)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.inference_mode()
    def __call__(self, frames: list[np.ndarray], query: str) -> np.ndarray:
        images = [Image.fromarray(frame[:, :, ::-1]) for frame in frames]
        inputs = self.processor(
            text=[query],
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        scores = self.model(**inputs).logits_per_text
        return scores[0].float().cpu().numpy()


class BoltPipeline:
    """Wrap any generation pipeline with query-aware selection.

    ``scorer(frames, query)`` returns one score per BGR frame. Injecting a scorer
    makes the selection algorithm testable without loading CLIP.
    """

    def __init__(
        self,
        pipeline,
        *,
        num_frames: int = 16,
        power: float = 3.0,
        scorer: Callable | None = None,
        clip_model: str = "openai/clip-vit-large-patch14",
        query: str | None = None,
    ):
        if num_frames < 1:
            raise ValueError("num_frames must be positive")
        if power != -1 and (not np.isfinite(power) or power <= 0):
            raise ValueError("power must be -1 or positive")
        self.pipeline = pipeline
        self.num_frames = num_frames
        self.power = power
        self.query = query
        self.scorer = scorer if scorer is not None else CLIPScorer(clip_model, pipeline.device)
        self.last_stats: dict = {}

    def generate_from_frames(self, frames, prompt, **kwargs):
        if not frames:
            raise ValueError("At least one frame is required")
        started = time.perf_counter()
        indices = np.arange(len(frames))
        if len(frames) > self.num_frames:
            scores = np.asarray(self.scorer(frames, self.query or prompt))
            if scores.shape != (len(frames),):
                raise ValueError("The CLIP scorer must return one score per frame")
            indices = np.unique(inverse_transform_sampling(scores, self.num_frames, self.power))
        selection_ms = (time.perf_counter() - started) * 1000
        text, generation_ms = self.pipeline.generate_from_frames(
            [frames[int(i)] for i in indices],
            prompt=prompt,
            **kwargs,
        )
        self.last_stats = {
            "selected_indices": indices.tolist(),
            "input_frames": len(frames),
            "selected_frames": len(indices),
            "selection_time_ms": selection_ms,
            "generation_time_ms": generation_ms,
        }
        return text, selection_ms + generation_ms
