"""Repository adaptation of DyCoke (Tao et al., CVPR 2025, arXiv:2411.15024).

Temporal compression drops the second frame in the most similar adjacent pairs.
It does not merge patch embeddings. Decode-time cache suppression zeros selected
vision keys and values; it does not shrink the cache or guarantee speedups. These
are the approximations used by the existing repository baseline.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .dycoke_cache import install_dycoke_hooks, remove_hooks


class DyCokePipeline:
    """Compose the repository's temporal selection and cache suppression with a VLM.

    Temporal selection needs ``encode_frame``. Cache suppression additionally
    needs accessible transformer decoder self-attention modules and image token
    IDs. Set ``cache_pruning=False`` for a temporal-only ablation.
    """

    def __init__(
        self,
        pipeline,
        *,
        drop_fraction: float = 0.3,
        keep_fraction: float = 0.8,
        start_layer: int = 3,
        similarity_threshold: float = 0.9,
        cache_pruning: bool = True,
    ):
        if not 0 <= drop_fraction < 1 or not 0 <= keep_fraction <= 1:
            raise ValueError("drop_fraction must be in [0,1); keep_fraction must be in [0,1]")
        if not -1 <= similarity_threshold <= 1 or start_layer < 0:
            raise ValueError("Invalid similarity_threshold or start_layer")
        self.pipeline = pipeline
        self.drop_fraction = drop_fraction
        self.keep_fraction = keep_fraction
        self.start_layer = start_layer
        self.similarity_threshold = similarity_threshold
        self.cache_pruning = cache_pruning
        self.last_stats: dict = {}

    @torch.inference_mode()
    def select_indices(self, frames) -> list[int]:
        if not frames:
            raise ValueError("At least one frame is required")
        n_drop = int(len(frames) * self.drop_fraction)
        if n_drop == 0:
            return list(range(len(frames)))
        embeddings = [self.pipeline.encode_frame(frame).float() for frame in frames]
        pairs = []
        for i, (previous, current) in enumerate(zip(embeddings, embeddings[1:])):
            length = min(len(previous), len(current))
            if length == 0:
                raise ValueError("Vision encoder returned no patch embeddings")
            similarity = F.cosine_similarity(previous[:length], current[:length], dim=-1).mean()
            pairs.append((i + 1, float(similarity)))
        dropped = {i for i, _ in sorted(pairs, key=lambda pair: -pair[1])[:n_drop]}
        return [i for i in range(len(frames)) if i not in dropped]

    def generate_from_frames(self, frames, prompt, **kwargs):
        started = time.perf_counter()
        indices = self.select_indices(frames)
        selection_ms = (time.perf_counter() - started) * 1000
        handles = []
        tracker = None
        try:
            if self.cache_pruning and self.keep_fraction < 1:
                tracker, handles = install_dycoke_hooks(
                    self.pipeline.model,
                    image_token_id=self.pipeline.image_token_id,
                    prune_ratio=1 - self.keep_fraction,
                    sim_threshold=self.similarity_threshold,
                    start_layer=self.start_layer,
                )
            text, generation_ms = self.pipeline.generate_from_frames(
                [frames[i] for i in indices],
                prompt=prompt,
                **kwargs,
            )
        finally:
            remove_hooks(handles, tracker)
        self.last_stats = {
            "selected_indices": indices,
            "input_frames": len(frames),
            "selected_frames": len(indices),
            "selection_time_ms": selection_ms,
            "generation_time_ms": generation_ms,
            "zeroed_kv_positions_across_layers": tracker.total_pruned if tracker else 0,
            "cache_shape_reduced": False,
        }
        return text, selection_ms + generation_ms
