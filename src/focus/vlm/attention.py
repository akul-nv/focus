# SPDX-License-Identifier: Apache-2.0
"""VLM attention map extraction for spatial importance feedback.

:class:`AttentionMapExtractor` runs a lightweight forward pass after VLM
generation to extract attention weights from the last few decoder layers.
The text-to-vision attention is aggregated into a spatial heatmap that
represents *what the VLM looked at* to produce its response -- a top-down,
semantically-aware complement to the bottom-up dissimilarity signal.
"""

from __future__ import annotations

import importlib
import logging
import time

import torch

from .focus import _get_decoder_layers

logger = logging.getLogger(__name__)


class AttentionMapExtractor:
    """Extract spatial attention maps from VLM decoder attention weights.

    After the VLM generates a response, call :meth:`extract_from_model` with
    the original (pre-generation) inputs to run a single forward pass with
    ``output_attentions=True``.  The method identifies vision-token positions
    via ``image_token_id``, aggregates text-to-vision attention across the
    last *N* decoder layers and all heads, and reshapes the result into a
    normalised ``(grid_h, grid_w)`` spatial heatmap.

    Args:
        num_layers: Number of final decoder layers to aggregate over.
        enabled: Master switch.
    """

    def __init__(self, num_layers: int = 4, enabled: bool = True) -> None:
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.num_layers = num_layers
        self.enabled = enabled

        self._total_extractions: int = 0
        self._total_extraction_time_ms: float = 0.0

    def extract_from_model(
        self,
        model: torch.nn.Module,
        inputs: dict,
        image_token_id: int,
        grid_h: int,
        grid_w: int,
        num_images: int,
    ) -> torch.Tensor | None:
        """Run an attention-extraction forward pass and return a spatial map.

        Args:
            model: The ``Qwen2VLForConditionalGeneration`` (or compatible)
                model instance.
            inputs: Tokenised inputs dict as returned by the processor
                (must include ``input_ids``; other keys are forwarded to
                ``model.forward``).
            image_token_id: Token ID used for vision-token placeholders
                (e.g. ``model.config.image_token_id``).
            grid_h: Spatial grid height (patches after merge) per image.
            grid_w: Spatial grid width (patches after merge) per image.
            num_images: Total number of images in the input.

        Returns:
            Normalised attention map of shape ``(grid_h, grid_w)`` with
            values in ``[0, 1]``, or *None* if extraction is disabled,
            the model returns no attention weights, or an error occurs.
        """
        if not self.enabled:
            return None

        start = time.perf_counter()
        try:
            attention_map = self._extract(
                model,
                inputs,
                image_token_id,
                grid_h,
                grid_w,
                num_images,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Attention extraction failed; feedback is unavailable: %s", exc)
            attention_map = None

        elapsed_ms = (time.perf_counter() - start) * 1000
        self._total_extractions += 1
        self._total_extraction_time_ms += elapsed_ms
        return attention_map

    def get_stats(self) -> dict:
        """Return extraction statistics for logging."""
        return {
            "enabled": self.enabled,
            "num_layers": self.num_layers,
            "total_extractions": self._total_extractions,
            "avg_extraction_time_ms": (
                self._total_extraction_time_ms / self._total_extractions
                if self._total_extractions > 0
                else 0.0
            ),
        }

    @staticmethod
    def _get_decoder_attn_config(model: torch.nn.Module):
        """Return the config object used by decoder attention layers.

        Works with both the new composite architecture (text_config sub-config)
        and the older monolithic architecture (top-level config).
        """
        cfg = getattr(model, "config", None)
        if cfg is None:
            return None
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "_attn_implementation"):
            return text_cfg
        if hasattr(cfg, "_attn_implementation"):
            return cfg
        return None

    @staticmethod
    def _patch_eager_attention_fp32(model):
        """Replace the module-level ``eager_attention_forward`` with a
        float32-safe version that prevents fp16 overflow/NaN in the
        ``Q @ K^T`` dot product.

        Returns the original function so it can be restored later, or
        *None* if the module could not be imported.
        """
        try:
            attention = _get_decoder_layers(model)[0].self_attn
            _mod = importlib.import_module(type(attention).__module__)
        except (ImportError, AttributeError, IndexError):
            return None
        if not hasattr(_mod, "eager_attention_forward") or not hasattr(_mod, "repeat_kv"):
            return None
        original = _mod.eager_attention_forward

        def _fp32_safe_eager(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling,
            dropout=0.0,
            **kwargs,
        ):
            key_states = _mod.repeat_kv(key, module.num_key_value_groups)
            value_states = _mod.repeat_kv(value, module.num_key_value_groups)

            # Compute Q @ K^T in float32 to avoid fp16 overflow → NaN.
            attn_weights = torch.matmul(query.float(), key_states.float().transpose(2, 3)) * scaling
            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask.float()

            attn_weights = torch.nn.functional.softmax(
                attn_weights,
                dim=-1,
                dtype=torch.float32,
            )
            attn_weights_f16 = attn_weights.to(query.dtype)
            attn_weights_f16 = torch.nn.functional.dropout(
                attn_weights_f16,
                p=dropout,
                training=module.training,
            )
            attn_output = torch.matmul(attn_weights_f16, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            return attn_output, attn_weights

        _mod.eager_attention_forward = _fp32_safe_eager
        return _mod, original

    @staticmethod
    def _restore_eager_attention(original_fn):
        """Restore the original ``eager_attention_forward``."""
        if original_fn is None:
            return
        module, original = original_fn
        module.eager_attention_forward = original

    def _extract(
        self,
        model: torch.nn.Module,
        inputs: dict,
        image_token_id: int,
        grid_h: int,
        grid_w: int,
        num_images: int,
    ) -> torch.Tensor | None:
        input_ids = inputs["input_ids"]
        vision_mask = input_ids[0] == image_token_id

        num_vision = int(vision_mask.sum().item())
        if num_vision == 0:
            logger.warning("[AttentionMapExtractor] No vision tokens found in input_ids")
            return None

        fwd_inputs = {k: v for k, v in inputs.items() if k != "labels"}

        # Temporarily force eager attention for the extraction pass.
        # In transformers >= 4.50 the unified Qwen2VLAttention dispatches
        # via config._attn_implementation at runtime.  SDPA/Flash kernels
        # never materialise attention weights; eager does.
        attn_cfg = self._get_decoder_attn_config(model)
        orig_impl = None
        if attn_cfg is not None:
            orig_impl = getattr(attn_cfg, "_attn_implementation", None)
            attn_cfg._attn_implementation = "eager"

        # Monkey-patch the eager attention function with a float32-safe
        # version.  The default implementation computes Q @ K^T in fp16
        # which overflows to inf → NaN after softmax.
        orig_eager_fn = self._patch_eager_attention_fp32(model)

        try:
            with torch.no_grad():
                outputs = model(
                    **fwd_inputs, output_attentions=True, return_dict=True, use_cache=False
                )
        finally:
            self._restore_eager_attention(orig_eager_fn)
            if attn_cfg is not None:
                attn_cfg._attn_implementation = orig_impl

        attentions = outputs.attentions
        if attentions is None or len(attentions) == 0:
            logger.warning("[AttentionMapExtractor] Model returned no attention weights.")
            return None
        if any(a is None for a in attentions):
            attentions = tuple(a for a in attentions if a is not None)
            if len(attentions) == 0:
                logger.warning("[AttentionMapExtractor] All returned attention layers were None")
                return None

        n_layers = min(self.num_layers, len(attentions))
        last_layers = attentions[-n_layers:]

        text_mask = ~vision_mask
        text_indices = text_mask.nonzero(as_tuple=True)[0]
        vision_indices = vision_mask.nonzero(as_tuple=True)[0]

        if text_indices.numel() == 0 or vision_indices.numel() == 0:
            logger.warning(
                f"[AttentionMapExtractor] Empty partition: "
                f"text_tokens={text_indices.numel()}, vision_tokens={vision_indices.numel()}"
            )
            return None

        # Only use text tokens that appear AFTER the vision block.
        # Pre-vision tokens (system prompt, user prefix) have zero
        # vision attention due to the causal mask and would dilute
        # the spatial signal.
        vision_end = vision_indices.max().item()
        post_vision = text_indices[text_indices > vision_end]
        if post_vision.numel() == 0:
            post_vision = text_indices

        # Max-pool across heads (preserves the most spatially selective
        # head) then average across layers.
        stacked = torch.stack([a[0] for a in last_layers])  # (L, H, S, S)
        max_heads = stacked.float().max(dim=1).values  # (L, S, S)
        avg_attn = max_heads.mean(dim=0)  # (S, S)

        text_to_vision = avg_attn[post_vision][:, vision_indices]

        # Weight each text token by how much total attention it
        # directs at vision tokens -- tokens that look at the image
        # more contribute more to the spatial map.
        row_sums = text_to_vision.sum(dim=1, keepdim=True).clamp(min=1e-8)
        weights = row_sums / row_sums.sum()
        spatial_attn = (text_to_vision * weights).sum(dim=0)

        expected_per_image = grid_h * grid_w
        if expected_per_image == 0:
            logger.warning(
                f"[AttentionMapExtractor] grid_h={grid_h}, grid_w={grid_w} -> 0 expected tokens"
            )
            return None

        if num_vision == num_images * expected_per_image:
            attention_map = spatial_attn.reshape(num_images, grid_h, grid_w)
            attention_map = attention_map.mean(dim=0)
        elif num_vision == expected_per_image:
            attention_map = spatial_attn.reshape(grid_h, grid_w)
        elif num_vision % expected_per_image == 0:
            n = num_vision // expected_per_image
            attention_map = spatial_attn.reshape(n, grid_h, grid_w).mean(dim=0)
        else:
            # Variable per-image grid sizes (common with dynamic resolution).
            # Pool the attention scores into a fixed spatial grid via
            # interpolation instead of failing.
            grid_thw = inputs.get("image_grid_thw")
            if grid_thw is not None:
                attention_map = self._pool_variable_grids(
                    spatial_attn,
                    grid_thw,
                    grid_h,
                    grid_w,
                )
                if attention_map is None:
                    logger.warning(
                        f"[AttentionMapExtractor] Vision token count mismatch: "
                        f"num_vision={num_vision}, num_images={num_images}, "
                        f"expected_per_image={expected_per_image}"
                    )
                    return None
            else:
                logger.warning(
                    f"[AttentionMapExtractor] Vision token count mismatch and "
                    f"no grid_thw available: num_vision={num_vision}, "
                    f"expected={num_images}*{expected_per_image}"
                )
                return None

        a_min = attention_map.min()
        a_max = attention_map.max()
        if a_max - a_min > 1e-8:
            attention_map = (attention_map - a_min) / (a_max - a_min)
        else:
            attention_map = torch.zeros_like(attention_map)

        return attention_map.cpu()

    @staticmethod
    def _pool_variable_grids(
        spatial_attn: torch.Tensor,
        grid_thw: torch.Tensor,
        target_h: int,
        target_w: int,
    ) -> torch.Tensor | None:
        """Aggregate per-vision-token attention when images have different grids.

        Each row of *grid_thw* describes one image as ``(t, h, w)`` in
        pre-merge patch units.  The total vision tokens should equal the sum
        of ``t*h*w / merge^2`` across images, but the individual grids may
        differ.  This method reshapes each image's tokens into its own
        spatial map, bilinearly resizes to ``(target_h, target_w)``, and
        averages.
        """
        import torch.nn.functional as F

        # Infer the spatial merge factor from the first image's grid
        h0 = int(grid_thw[0][1])
        merge = 1
        if target_h > 0 and h0 > 0:
            merge = max(1, h0 // target_h)

        offset = 0
        maps: list[torch.Tensor] = []
        for row in grid_thw:
            t_i, h_i, w_i = int(row[0]), int(row[1]), int(row[2])
            tokens_i = (t_i * h_i * w_i) // (merge * merge)
            if offset + tokens_i > spatial_attn.numel():
                return None
            chunk = spatial_attn[offset : offset + tokens_i]
            offset += tokens_i

            gh_i = h_i // merge
            gw_i = w_i // merge
            if gh_i * gw_i * t_i == 0:
                continue
            per_frame = gh_i * gw_i
            for fi in range(t_i):
                frame_chunk = chunk[fi * per_frame : (fi + 1) * per_frame]
                img_map = frame_chunk.reshape(1, 1, gh_i, gw_i).float()
                resized = F.interpolate(
                    img_map,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
                maps.append(resized.squeeze(0).squeeze(0))

        if not maps or offset != spatial_attn.numel():
            return None
        return torch.stack(maps).mean(dim=0)
