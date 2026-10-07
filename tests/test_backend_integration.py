# SPDX-License-Identifier: Apache-2.0
"""No-download integration checks against actual, tiny Qwen backends.

These use randomly initialized models and a tensor-only processor stand-in;
they validate Transformers APIs, not checkpoint quality or tokenizer behavior.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

from focus.vlm.attention import AttentionMapExtractor  # noqa: E402
from focus.vlm.focus import (  # noqa: E402
    SpatialFocusCalibrator,
    _get_decoder_layers,
)


@pytest.fixture(params=[2, 3], ids=["qwen2", "qwen3"])
def backend(request):
    """Construct both supported architectures without fetching any artifacts."""
    generation = request.param
    torch.manual_seed(7)
    text_config = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 128,
        "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]},
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    vision_config = {
        "depth": 1,
        "num_heads": 2,
        "patch_size": 2,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    }
    if generation == 2:
        vision_config.update(embed_dim=16, hidden_size=32, mlp_ratio=2)
    else:
        text_config["head_dim"] = 8
        vision_config.update(
            hidden_size=16,
            out_hidden_size=32,
            intermediate_size=32,
            num_position_embeddings=16,
            deepstack_visual_indexes=[0],
        )
    config_class = getattr(transformers, f"Qwen{generation}VLConfig")
    model_class = getattr(transformers, f"Qwen{generation}VLForConditionalGeneration")
    config = config_class(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=3,
        video_token_id=6,
        vision_start_token_id=4,
        vision_end_token_id=5,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = model_class(config).eval()
    processor = TensorProcessor()
    return generation, model, processor


class TensorProcessor:
    """Provide valid tiny multimodal inputs while leaving model execution real."""

    def __init__(self):
        self.pixels = torch.randn(16, 24)
        self.image_counts = []

    def apply_chat_template(self, messages, **kwargs):
        return "synthetic prompt"

    def __call__(self, *, images, **kwargs):
        count = len(images)
        self.image_counts.append(count)
        ids = [1] + [4, 3, 3, 3, 3, 5] * count + [7, 8]
        return transformers.BatchFeature(
            data={
                "input_ids": torch.tensor([ids]),
                "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                "pixel_values": self.pixels.repeat(count, 1),
                "image_grid_thw": torch.tensor([[1, 4, 4]] * count),
            }
        )

    def batch_decode(self, token_ids, **kwargs):
        return ["synthetic response"] * len(token_ids)


def _patch_loaders(monkeypatch, model, processor):
    monkeypatch.setattr(type(model), "from_pretrained", lambda *args, **kwargs: model)
    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: processor,
    )


def test_real_vision_encoder_contract(backend, monkeypatch):
    generation, model, processor = backend
    _patch_loaders(monkeypatch, model, processor)
    module = importlib.import_module(f"focus.encoders.qwen{generation}_vl")
    encoder = getattr(module, f"Qwen{generation}VLEncoder")("unused-local-model", device="cpu")

    embedding = encoder(np.zeros((8, 8, 3), dtype=np.uint8))

    assert isinstance(embedding, torch.Tensor)
    assert embedding.shape == (4, 32)
    assert encoder.hidden_size == embedding.shape[1]
    assert encoder.last_grid_hw == (2, 2)
    assert torch.isfinite(embedding).all()


def test_real_pipeline_generation_and_feedback(backend, monkeypatch):
    generation, model, processor = backend
    _patch_loaders(monkeypatch, model, processor)
    module = importlib.import_module(f"focus.vlm.qwen{generation}_vl")
    pipeline = getattr(module, f"Qwen{generation}VLPipeline")(
        "unused-local-model",
        device="cpu",
        attention_layers=2,
        focus_alpha=1.05,
        focus_range=(0.75, 1.0),
        focus_suppress=0.04,
    )
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    embedding = pipeline.encode_frame(frame)
    assert isinstance(embedding, torch.Tensor)
    assert pipeline.hidden_size == embedding.shape[1]
    pipeline.extract_attention = True
    response, elapsed_ms = pipeline.generate_from_frames(
        [frame, frame],
        "Describe the activity.",
        max_new_tokens=3,
        spatial_prior=torch.tensor([[0.0, 0.25], [0.75, 1.0]]),
    )
    assert response == "synthetic response"
    assert elapsed_ms >= 0
    assert pipeline.last_attention_map is not None
    assert pipeline.last_attention_map.shape == (2, 2)
    assert torch.isfinite(pipeline.last_attention_map).all()
    assert all(not layer.self_attn._forward_pre_hooks for layer in _get_decoder_layers(model))


def test_real_dynamic_cache_scales_only_vision_values(backend):
    _, model, processor = backend
    inputs = processor(images=[None])
    prior = torch.tensor([[0.0, 0.25], [0.75, 1.0]])
    calibrator = SpatialFocusCalibrator(
        alpha=1.05,
        focus_range=(0.75, 1.0),
        suppress=0.04,
    )
    with torch.no_grad():
        baseline = model(**inputs, use_cache=True).past_key_values
        calibrator.apply(model, inputs["input_ids"], model.config.image_token_id, prior)
        try:
            cache = model(**inputs, use_cache=True).past_key_values
            for expected, actual in zip(baseline.layers, cache.layers):
                torch.testing.assert_close(actual.values, expected.values)
                torch.testing.assert_close(actual.keys, expected.keys)
            initial_values = [layer.values.clone() for layer in cache.layers]
            initial_keys = [layer.keys.clone() for layer in cache.layers]
            prompt_length = inputs["input_ids"].shape[1]
            scales = torch.ones(prompt_length)
            scales[2:6] = torch.tensor([0.96, 0.97, 1.0375, 1.05])
            for step in (1, 2):
                position = prompt_length + step - 1
                model(
                    input_ids=torch.tensor([[9]]),
                    position_ids=torch.full((3, 1, 1), position),
                    attention_mask=torch.ones(1, position + 1, dtype=torch.long),
                    cache_position=torch.tensor([position]),
                    past_key_values=cache,
                    use_cache=True,
                )
                for index, layer in enumerate(cache.layers):
                    expected_values = initial_values[index] * scales.pow(step)[None, None, :, None]
                    torch.testing.assert_close(layer.values[:, :, :prompt_length], expected_values)
                    torch.testing.assert_close(
                        layer.keys[:, :, :prompt_length], initial_keys[index]
                    )
        finally:
            calibrator.remove()
    assert all(not layer.self_attn._forward_pre_hooks for layer in _get_decoder_layers(model))


def test_real_attention_extraction_restores_backend(backend):
    _, model, processor = backend
    inputs = processor(images=[None, None])
    module = importlib.import_module(type(_get_decoder_layers(model)[0].self_attn).__module__)
    eager_before = module.eager_attention_forward
    config = model.config.text_config
    implementation_before = config._attn_implementation
    extractor = AttentionMapExtractor(num_layers=2)

    attention_map = extractor.extract_from_model(
        model,
        inputs,
        image_token_id=3,
        grid_h=2,
        grid_w=2,
        num_images=2,
    )

    assert attention_map is not None
    assert attention_map.shape == (2, 2)
    assert torch.isfinite(attention_map).all()
    assert attention_map.min() >= 0 and attention_map.max() <= 1
    assert config._attn_implementation == implementation_before
    assert module.eager_attention_forward is eager_before


def test_real_dycoke_decode_hooks_track_and_restore_attention(backend):
    from focus.baselines.dycoke_cache import install_dycoke_hooks, remove_hooks

    _, model, processor = backend
    inputs = processor(images=[None])
    config = model.config.text_config
    original_implementation = config._attn_implementation
    tracker, handles = install_dycoke_hooks(
        model,
        image_token_id=3,
        start_layer=0,
        prune_ratio=0.5,
        sim_threshold=1.0,
    )
    try:
        with torch.no_grad():
            cache = model(**inputs, use_cache=True).past_key_values
            assert not tracker._prev_attn  # Prefill does not request a quadratic attention map.
            assert config._attn_implementation == original_implementation
            prompt_length = inputs["input_ids"].shape[1]
            for step in range(3):
                position = prompt_length + step
                model(
                    input_ids=torch.tensor([[9 + step]]),
                    position_ids=torch.full((3, 1, 1), position),
                    attention_mask=torch.ones(1, position + 1, dtype=torch.long),
                    cache_position=torch.tensor([position]),
                    past_key_values=cache,
                    use_cache=True,
                )
                assert config._attn_implementation == original_implementation
        assert tracker.vision_token_mask.tolist() == [
            False,
            False,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
        ]
        assert len(tracker._prev_attn) == 2
        assert tracker.total_pruned > 0
        for layer_idx, positions in tracker._pruned.items():
            selected = list(positions)
            assert not cache.layers[layer_idx].keys[:, :, selected].any()
            assert not cache.layers[layer_idx].values[:, :, selected].any()
    finally:
        remove_hooks(handles, tracker)
    assert all(not layer.self_attn._forward_hooks for layer in _get_decoder_layers(model))
    assert all(not layer.self_attn._forward_pre_hooks for layer in _get_decoder_layers(model))
    assert config._attn_implementation == original_implementation


def test_real_dycoke_pipeline_uses_generator_encoder_and_finishes(backend, monkeypatch):
    from focus.baselines import DyCokePipeline

    generation, model, processor = backend
    _patch_loaders(monkeypatch, model, processor)
    module = importlib.import_module(f"focus.vlm.qwen{generation}_vl")
    base = getattr(module, f"Qwen{generation}VLPipeline")("unused-local-model", device="cpu")
    pipeline = DyCokePipeline(base, start_layer=0, drop_fraction=0.5)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    text, elapsed = pipeline.generate_from_frames([frame] * 4, "Describe.", max_new_tokens=3)
    assert text == "synthetic response" and elapsed >= 0
    assert pipeline.last_stats["selected_frames"] == 2
    assert not model._forward_pre_hooks
    assert all(not layer.self_attn._forward_pre_hooks for layer in _get_decoder_layers(model))
