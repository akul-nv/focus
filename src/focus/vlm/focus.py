# SPDX-License-Identifier: Apache-2.0
"""Spatial-prior-driven value-cache calibration for focused VLM generation.

Inspired by SPARC (Selective Progressive Attention ReCalibration), this
module uses FOCUS's combined dissimilarity + attention spatial prior to
scale the value cache of vision tokens during autoregressive generation.
Instead of SPARC's dynamic per-step token selection via relative activation
scores, the selection is fixed from the spatial prior which is a stronger
signal for continuous video because it incorporates both motion (where pixels
changed) and semantics (what the VLM previously attended to).

The effect: vision tokens in the task-relevant region have their value
representations progressively amplified, causing the VLM's generated text
to focus on that region rather than describing the entire scene.
"""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import Optional

import torch
import torch.nn.functional as F


def _get_decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Resolve the decoder layer list from a Qwen2-VL model.

    Handles both ``Qwen2VLForConditionalGeneration`` (layers at
    ``model.model.language_model.layers``) and direct ``Qwen2VLModel``
    (layers at ``model.language_model.layers``).
    """
    # Qwen2VLForConditionalGeneration wraps Qwen2VLModel in self.model
    inner = getattr(model, "model", model)
    lm = getattr(inner, "language_model", inner)
    layers = getattr(lm, "layers", None)
    if layers is None:
        raise AttributeError(
            f"Cannot find decoder layers on {type(model).__name__}. "
            "Expected model.model.language_model.layers"
        )
    return layers


class SpatialFocusCalibrator:
    """Scale vision-token value cache during generation using a spatial prior.

    At each autoregressive step the value vectors of vision tokens in the
    KV cache are multiplied by a per-token scale factor derived from the
    spatial prior.  The scaling accumulates over generation steps, so
    high-prior tokens receive progressively stronger amplification.

    Lifecycle::

        calibrator = SpatialFocusCalibrator(alpha=1.02)
        calibrator.apply(model, input_ids, image_token_id, spatial_prior)
        output_ids = model.generate(...)
        calibrator.remove()

    Args:
        alpha: Base boost factor for tokens inside ``focus_range``.
            The effective per-token per-step scale is
            ``1.0 + (alpha - 1.0) * prior_value``, so tokens with a
            prior value of 1.0 are scaled by *alpha* each step and tokens
            with prior 0.0 are unaffected.
        focus_range: ``(low, high)`` prior-value window.  Only tokens
            whose prior value falls within ``[low, high]`` are boosted;
            tokens outside are suppressed (if ``suppress > 0``) or left
            at scale 1.0.  Defaults to ``(0.0, 1.0)``.
        suppress: Per-step suppression strength for tokens *outside*
            ``focus_range``.  The scale for a suppressed token with
            prior value ``C_j`` is ``1.0 - suppress * (1.0 - C_j)``,
            so tokens with very low prior decay fastest.  Set to ``0.0``
            (default) for no suppression.
        start_layer: First decoder layer (inclusive) to calibrate.
        end_layer: Last decoder layer (inclusive).  ``-1`` means the final
            layer.
    """

    def __init__(
        self,
        alpha: float = 1.02,
        focus_range: tuple[float, float] = (0.0, 1.0),
        suppress: float = 0.0,
        start_layer: int = 0,
        end_layer: int = -1,
    ) -> None:
        if not isfinite(alpha) or alpha < 1.0:
            raise ValueError("alpha must be finite and >= 1")
        if not 0 <= focus_range[0] <= focus_range[1] <= 1:
            raise ValueError("focus_range must satisfy 0 <= low <= high <= 1")
        if not isfinite(suppress) or suppress < 0:
            raise ValueError("suppress must be finite and >= 0")
        if start_layer < 0 or end_layer < -1 or (end_layer >= 0 and end_layer < start_layer):
            raise ValueError("Invalid decoder layer range")
        self.alpha = alpha
        self.focus_range = focus_range
        self.suppress = suppress
        self.start_layer = start_layer
        self.end_layer = end_layer

        self._handles: list[torch.utils.hooks.RemovableHook] = []
        self._applied = False

    def apply(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        image_token_id: int,
        spatial_prior: torch.Tensor,
    ) -> None:
        """Register forward-pre-hooks on decoder attention layers.

        Args:
            model: ``Qwen2VLForConditionalGeneration`` instance.
            input_ids: Tokenised input ``(1, seq_len)``.
            image_token_id: Token ID used for vision-token placeholders.
            spatial_prior: Normalised ``(grid_h, grid_w)`` heatmap in
                ``[0, 1]`` from :class:`SpatialAttentionAnalyzer`.
        """
        if self._applied:
            self.remove()

        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Spatial calibration supports one prompt at a time")
        if spatial_prior.ndim != 2 or spatial_prior.numel() == 0:
            raise ValueError("spatial_prior must be a nonempty 2-D map")
        if not torch.isfinite(spatial_prior).all():
            raise ValueError("spatial_prior must contain finite values")
        vision_mask = input_ids[0] == image_token_id
        vision_indices = vision_mask.nonzero(as_tuple=True)[0]
        num_vision = vision_indices.numel()
        if num_vision == 0:
            return

        prior_flat = spatial_prior.detach().float().reshape(-1)

        if prior_flat.numel() != num_vision:
            prior_flat = F.interpolate(
                prior_flat.unsqueeze(0).unsqueeze(0),
                size=(num_vision,),
                mode="nearest",
            ).reshape(-1)

        prior_clamped = prior_flat.clamp(0, 1)
        lo, hi = self.focus_range
        inside = (prior_clamped >= lo) & (prior_clamped <= hi)
        outside = ~inside

        per_token_scales = torch.ones_like(prior_clamped)

        if inside.any():
            per_token_scales[inside] = 1.0 + (self.alpha - 1.0) * prior_clamped[inside]

        if self.suppress > 0.0 and outside.any():
            per_token_scales[outside] = 1.0 - self.suppress * (1.0 - prior_clamped[outside])
            per_token_scales.clamp_(min=0.5)

        layers = _get_decoder_layers(model)
        num_layers = len(layers)
        end = self.end_layer if self.end_layer >= 0 else num_layers - 1
        start = max(0, self.start_layer)
        end = min(end, num_layers - 1)

        if start >= num_layers:
            raise ValueError("start_layer exceeds the number of decoder layers")
        for layer in layers[start : end + 1]:
            if not hasattr(layer, "self_attn"):
                raise TypeError("Calibration requires decoder layers exposing self_attn")
        vision_idx_cpu = vision_indices.cpu()

        for layer_idx in range(start, end + 1):
            hook = _make_value_scale_hook(
                layer_idx,
                vision_idx_cpu,
                per_token_scales,
            )
            handle = layers[layer_idx].self_attn.register_forward_pre_hook(
                hook,
                with_kwargs=True,
            )
            self._handles.append(handle)

        self._applied = True

    def remove(self) -> None:
        """Remove all registered hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._applied = False


def _make_value_scale_hook(
    layer_idx: int,
    vision_indices: torch.Tensor,
    per_token_scales: torch.Tensor,
) -> Callable:
    """Create a forward-pre-hook that scales vision-token values in the cache.

    The hook only activates during the generation phase (seq_len == 1),
    not during the prefill pass (seq_len > 1).
    """
    _device_scales: Optional[torch.Tensor] = None

    def hook(module, args, kwargs):
        nonlocal _device_scales

        hidden_states = args[0] if args else kwargs.get("hidden_states")
        if hidden_states is None or hidden_states.shape[1] > 1:
            return

        past_kv = kwargs.get("past_key_values", kwargs.get("past_key_value"))
        if past_kv is None:
            return

        if hasattr(past_kv, "layers"):
            if layer_idx >= len(past_kv.layers):
                return
            values = past_kv.layers[layer_idx].values
        elif hasattr(past_kv, "value_cache"):
            if layer_idx >= len(past_kv.value_cache):
                return
            values = past_kv.value_cache[layer_idx]
        else:
            raise TypeError("Unsupported KV cache: expected layers[].values or value_cache[]")
        if values is None or values.numel() == 0:
            return

        if (
            _device_scales is None
            or _device_scales.device != values.device
            or _device_scales.dtype != values.dtype
        ):
            _device_scales = per_token_scales.to(
                device=values.device,
                dtype=values.dtype,
            )

        idx = vision_indices.to(values.device)
        seq_len = values.shape[2]
        valid = idx < seq_len
        if not valid.all():
            idx = idx[valid]
            scales = _device_scales[valid]
        else:
            scales = _device_scales

        # shape: (1, 1, num_selected, 1) for broadcasting over
        # (batch, heads, seq, head_dim)
        values[:, :, idx] *= scales.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

        # Clamp to prevent fp16 overflow -> NaN when alpha is aggressive
        if values.dtype == torch.float16:
            values.clamp_(-65504.0, 65504.0)

    return hook
