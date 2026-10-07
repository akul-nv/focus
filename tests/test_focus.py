# SPDX-License-Identifier: Apache-2.0
"""Exercise real attention hooks against tiny writable KV caches."""

from types import SimpleNamespace

import pytest
import torch

from focus.vlm.focus import SpatialFocusCalibrator, _get_decoder_layers


class Attention(torch.nn.Module):
    def forward(self, hidden_states, past_key_values=None):
        return hidden_states


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
        for layer in self.layers:
            layer.self_attn = Attention()


def test_scaling_is_decode_only_compounds_and_preserves_text_and_keys():
    model = Decoder()
    cache = SimpleNamespace(
        layers=[
            SimpleNamespace(values=torch.ones(1, 2, 6, 3), keys=torch.ones(1, 2, 6, 3))
            for _ in model.layers
        ]
    )
    calibrator = SpatialFocusCalibrator(alpha=1.1, focus_range=(0.75, 1), suppress=0.04)
    calibrator.apply(
        model, torch.tensor([[1, 9, 9, 9, 9, 2]]), 9, torch.tensor([[0.0, 0.5], [0.75, 1.0]])
    )
    expected = torch.tensor([1, 0.96, 0.98, 1.075, 1.1, 1])
    for layer, entry in zip(model.layers, cache.layers):
        layer.self_attn(torch.ones(1, 6, 4), past_key_values=cache)
        assert torch.equal(entry.values, torch.ones_like(entry.values))
    for step in (1, 2):
        for layer, entry in zip(model.layers, cache.layers):
            layer.self_attn(torch.ones(1, 1, 4), past_key_values=cache)
            torch.testing.assert_close(entry.values[0, 0, :, 0], expected**step)
            assert torch.equal(entry.keys, torch.ones_like(entry.keys))
    calibrator.remove()
    for layer, entry in zip(model.layers, cache.layers):
        layer.self_attn(torch.ones(1, 1, 4), past_key_values=cache)
        torch.testing.assert_close(entry.values[0, 0, :, 0], expected**2)
    calibrator.remove()


def test_nearest_resampling_and_layer_selection():
    model = Decoder()
    cache = SimpleNamespace(
        layers=[SimpleNamespace(values=torch.ones(1, 1, 4, 1)) for _ in model.layers]
    )
    calibrator = SpatialFocusCalibrator(alpha=2, start_layer=1)
    calibrator.apply(model, torch.tensor([[9, 9, 9, 9]]), 9, torch.tensor([[0.0, 1.0]]))
    for layer in model.layers:
        layer.self_attn(torch.ones(1, 1, 4), past_key_values=cache)
    torch.testing.assert_close(cache.layers[0].values.flatten(), torch.ones(4))
    torch.testing.assert_close(cache.layers[1].values.flatten(), torch.tensor([1.0, 1.0, 2.0, 2.0]))
    calibrator.remove()


def test_no_vision_tokens_does_not_install_hooks():
    model = Decoder()
    calibrator = SpatialFocusCalibrator()
    calibrator.apply(model, torch.tensor([[1, 2]]), 9, torch.ones(2, 2))
    assert not calibrator._applied
    assert not model.layers[0].self_attn._forward_pre_hooks


def test_unsupported_cache_is_an_explicit_error():
    model = Decoder()
    calibrator = SpatialFocusCalibrator()
    calibrator.apply(model, torch.tensor([[9]]), 9, torch.ones(1, 1))
    with pytest.raises(TypeError, match="Unsupported KV cache"):
        model.layers[0].self_attn(torch.ones(1, 1, 4), past_key_values=object())
    calibrator.remove()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 0.5},
        {"alpha": float("nan")},
        {"focus_range": (0.8, 0.2)},
        {"suppress": -0.1},
        {"start_layer": 3, "end_layer": 1},
    ],
)
def test_invalid_calibration_options(kwargs):
    with pytest.raises(ValueError):
        SpatialFocusCalibrator(**kwargs)


def test_incompatible_decoder_is_an_explicit_error():
    with pytest.raises(AttributeError, match="Cannot find decoder layers"):
        _get_decoder_layers(torch.nn.Linear(2, 2))
