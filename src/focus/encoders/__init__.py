# SPDX-License-Identifier: Apache-2.0
"""Vision encoder implementations.

.. rubric:: Extensibility

Subclass :class:`BaseEncoder` to add support for a new vision backbone.
"""

from __future__ import annotations

from .base import BaseEncoder


def load_encoder(model_name: str, device: str = "cuda") -> BaseEncoder:
    """Load an encoder with auto-detection of model family.

    Inspects the ``model_type`` field in the HuggingFace config to pick
    between Qwen2-VL and Qwen3-VL encoders.
    """
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise ImportError("Install the model backend with: pip install '.[qwen]'") from exc

    config = AutoConfig.from_pretrained(model_name)
    model_type = getattr(config, "model_type", "")

    if model_type == "qwen3_vl":
        from .qwen3_vl import Qwen3VLEncoder

        return Qwen3VLEncoder(model_name=model_name, device=device)

    if model_type == "qwen2_vl":
        from .qwen2_vl import Qwen2VLEncoder

        return Qwen2VLEncoder(model_name=model_name, device=device)
    raise ValueError(
        f"Unsupported model_type {model_type!r}. Supported: qwen2_vl, qwen3_vl. "
        "Implement BaseEncoder to add another vision backbone."
    )


__all__ = ["BaseEncoder", "load_encoder"]
