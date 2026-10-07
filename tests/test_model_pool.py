# SPDX-License-Identifier: Apache-2.0
"""Verify loader dispatch and cache behavior without downloading models."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from focus.encoders import load_encoder
from focus.vlm import model_pool


@pytest.fixture(autouse=True)
def clear_pool():
    model_pool.clear_pool()
    yield
    model_pool.clear_pool()


def test_cache_respects_constructor_options(monkeypatch):
    auto = Mock()
    auto.from_pretrained.return_value = SimpleNamespace(model_type="qwen2_vl")
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoConfig=auto))
    from focus.vlm import qwen2_vl

    constructor = Mock(side_effect=[object(), object()])
    monkeypatch.setattr(qwen2_vl, "Qwen2VLPipeline", constructor)
    first = model_pool.load_vlm_pipeline("model", "cpu", focus_alpha=1.05)
    assert model_pool.load_vlm_pipeline("model", "cpu", focus_alpha=1.05) is first
    assert model_pool.load_vlm_pipeline("model", "cpu", focus_alpha=1.0) is not first
    assert constructor.call_count == 2


@pytest.mark.parametrize("loader", [load_encoder, model_pool.load_vlm_pipeline])
def test_unsupported_model_has_actionable_error(monkeypatch, loader):
    auto = Mock()
    auto.from_pretrained.return_value = SimpleNamespace(model_type="new_model")
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoConfig=auto))
    with pytest.raises(ValueError, match="Unsupported model_type"):
        loader("model", device="cpu")
