"""CPU coverage for the published baseline adaptations; no model downloads."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from focus.baselines import BoltPipeline, DyCokePipeline, inverse_transform_sampling
from focus.baselines.dycoke_cache import (
    PruningTracker,
    install_dycoke_hooks,
    remove_hooks,
)


class Generator:
    device = "cpu"

    def generate_from_frames(self, frames, prompt, **kwargs):
        self.frames = frames
        return "caption", 10.0

    def encode_frame(self, frame):
        return torch.tensor([[1.0, 0.0]]) if frame < 2 else torch.tensor([[0.0, 1.0]])


def test_bolt_uniform_ties_and_single_sample():
    assert inverse_transform_sampling(np.ones(5), 3).tolist() == [0, 2, 4]
    assert inverse_transform_sampling(np.array([0, 0, 1]), 1).tolist() == [2]
    with pytest.raises(ValueError):
        inverse_transform_sampling(np.array([]), 1)
    with pytest.raises(ValueError):
        inverse_transform_sampling(np.array([np.nan]), 1)


def test_bolt_selection_retains_order_and_reports_deduplication():
    base = Generator()
    pipeline = BoltPipeline(base, num_frames=3, scorer=lambda frames, query: [0, 0, 0, 1, 0])
    text, elapsed = pipeline.generate_from_frames(list(range(5)), "caption")
    assert text == "caption" and elapsed >= 10
    assert base.frames == [3]
    assert pipeline.last_stats["selected_indices"] == [3]
    assert pipeline.last_stats["input_frames"] == 5
    assert pipeline.last_stats["selected_frames"] == 1


def test_dycoke_temporal_selection_drops_redundant_neighbor_without_merging():
    base = Generator()
    pipeline = DyCokePipeline(base, drop_fraction=0.25, cache_pruning=False)
    pipeline.generate_from_frames([0, 1, 2, 3], "caption")
    assert base.frames == [0, 2, 3]
    assert pipeline.last_stats["cache_shape_reduced"] is False


@pytest.mark.parametrize("modern", [True, False])
def test_cache_zeroing_changes_only_visual_positions_and_counts_unique(modern):
    keys = torch.ones(1, 1, 4, 2)
    values = keys.clone()
    cache = (
        SimpleNamespace(layers=[SimpleNamespace(keys=keys, values=values)])
        if modern
        else SimpleNamespace(key_cache=[keys], value_cache=[values])
    )
    tracker = PruningTracker(
        torch.tensor([False, True, True, False]), start_layer=0, prune_ratio=0.5, sim_threshold=0.9
    )
    for scores in ([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]):
        tracker.record_attention_and_prune(0, torch.tensor(scores).reshape(1, 1, 1, 4), cache)
    assert keys.shape == (1, 1, 4, 2)
    assert not keys[0, 0, 1].any()
    assert keys[0, 0, [0, 2, 3]].all()
    assert tracker.total_pruned == 1
    zero = PruningTracker(torch.tensor([True] * 4), start_layer=0, prune_ratio=0)
    zero.record_attention_and_prune(0, torch.ones(1, 1, 1, 4), cache)
    assert zero.total_pruned == 0


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_idx = 0
        self.config = SimpleNamespace(_attn_implementation="sdpa")

    def forward(self, hidden, past_key_values=None):
        return hidden, torch.ones(1, 1, 1, 4)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attention()

    def forward(self, input_ids):
        return self.self_attn(input_ids)


def test_cache_hooks_restore_backend_even_when_generation_fails():
    base = Generator()
    base.model = Model()
    base.image_token_id = 9

    def fail(*args, **kwargs):
        assert base.model.self_attn.config._attn_implementation == "eager"
        raise RuntimeError("generation failed")

    base.model.self_attn.forward = fail
    base.generate_from_frames = lambda *args, **kwargs: base.model(input_ids=torch.tensor([[9]]))
    pipeline = DyCokePipeline(base, drop_fraction=0, start_layer=0)
    with pytest.raises(RuntimeError, match="generation failed"):
        pipeline.generate_from_frames([0], "caption")
    assert base.model.self_attn.config._attn_implementation == "sdpa"
    assert not base.model._forward_pre_hooks
    assert not base.model.self_attn._forward_hooks


def test_cache_hook_captures_mask_once_and_rejects_unsupported_layers():
    model = Model()
    tracker, handles = install_dycoke_hooks(model, image_token_id=9, start_layer=0)
    try:
        model(input_ids=torch.tensor([[0, 9, 9, 0]]))
        model(input_ids=torch.tensor([[3]]))
        assert tracker.vision_token_mask.tolist() == [False, True, True, False]
    finally:
        remove_hooks(handles, tracker)
    with pytest.raises(ValueError, match="No supported"):
        install_dycoke_hooks(model, image_token_id=9, start_layer=1)
