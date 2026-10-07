# SPDX-License-Identifier: Apache-2.0
"""Model pool: load VLM pipelines with auto-detection and caching.

Calling :func:`load_vlm_pipeline` twice with the same ``model_name`` and
``device`` and constructor options returns the same instance, avoiding duplicate GPU memory usage
when multiple pipeline roles share a model.
"""

from __future__ import annotations

from .base import BaseVLMPipeline

_pool: dict[tuple, BaseVLMPipeline] = {}


def load_vlm_pipeline(
    model_name: str,
    device: str = "cuda",
    **kwargs,
) -> BaseVLMPipeline:
    """Load (or retrieve from cache) a VLM pipeline for the given model.

    Auto-detects whether *model_name* is a Qwen2-VL or Qwen3-VL model
    by inspecting the ``model_type`` field in the HuggingFace config.

    Args:
        model_name: HuggingFace model identifier.
        device: Torch device string.
        **kwargs: Forwarded to the pipeline constructor (e.g.
            ``focus_alpha``, ``attention_layers``).

    Returns:
        A :class:`BaseVLMPipeline` subclass instance (cached).
    """
    key = (model_name, device, tuple(sorted(kwargs.items())))
    try:
        hash(key)
    except TypeError as exc:
        raise TypeError("Pipeline constructor options must be hashable for model caching") from exc
    if key in _pool:
        return _pool[key]

    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise ImportError("Install the model backend with: pip install '.[qwen]'") from exc

    config = AutoConfig.from_pretrained(model_name)
    model_type = getattr(config, "model_type", "")

    if model_type == "qwen3_vl":
        from .qwen3_vl import Qwen3VLPipeline

        pipeline = Qwen3VLPipeline(model_name, device=device, **kwargs)
    elif model_type == "qwen2_vl":
        from .qwen2_vl import Qwen2VLPipeline

        pipeline = Qwen2VLPipeline(model_name, device=device, **kwargs)

    else:
        raise ValueError(
            f"Unsupported model_type {model_type!r}. Supported: qwen2_vl, qwen3_vl. "
            "Implement BaseVLMPipeline to add another model family."
        )

    _pool[key] = pipeline
    return pipeline


def clear_pool() -> None:
    """Remove all cached pipelines (for testing / cleanup)."""
    _pool.clear()
