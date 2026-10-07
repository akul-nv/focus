# SPDX-License-Identifier: Apache-2.0
"""Spatial attention analysis for burst-level action localisation.

:class:`SpatialAttentionAnalyzer` consumes per-patch dissimilarity maps
produced by the gating layer and accumulates them into a spatial attention
heatmap across a burst.  The heatmap is blended with a top-down VLM
attention prior (from the previous burst) to produce a *focus prior* --
a normalised spatial map that drives value-cache calibration during VLM
generation (see :mod:`~focus.vlm.focus`).

Supports two fusion modes for combining bottom-up dissimilarity with the
top-down VLM attention prior:

- **multiplicative** (default): ``dmap * (eps + prior ** gamma)``.
  Attention modulates dissimilarity, suppressing unattended regions.
- **additive** (legacy): ``alpha * dmap + (1 - alpha) * prior``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class SpatialAttentionAnalyzer:
    """Accumulate per-patch dissimilarity maps and produce a focus prior.

    Designed to be used per-burst: call :meth:`update` for every captured
    frame, query :meth:`get_focus_prior` when the burst is submitted to
    the VLM, then :meth:`reset` before the next burst.

    Optionally blends a *bottom-up* dissimilarity signal with a *top-down*
    VLM attention prior (set via :meth:`set_attention_prior`).

    Args:
        activation_threshold_sigma: Standard deviations above the mean to
            consider a patch "active".
        attention_blend_alpha: Additive blend weight (only used when
            ``fusion_mode="additive"``).
        fusion_mode: ``"multiplicative"`` (default) or ``"additive"``.
        fusion_gamma: Exponent controlling attention sharpness in
            multiplicative fusion.  Lower values spread attention more
            evenly; higher values sharpen peaks.
        enabled: Master switch.
    """

    def __init__(
        self,
        activation_threshold_sigma: float = 1.0,
        attention_blend_alpha: float = 0.5,
        fusion_mode: str = "multiplicative",
        fusion_gamma: float = 0.5,
        enabled: bool = True,
    ) -> None:
        self.activation_threshold_sigma = activation_threshold_sigma
        self.attention_blend_alpha = attention_blend_alpha
        self.fusion_mode = fusion_mode
        self.fusion_gamma = fusion_gamma
        self.enabled = enabled

        self._grid_h: int = 0
        self._grid_w: int = 0
        self._accumulated: torch.Tensor | None = None
        self._frame_count: int = 0
        self._total_updates: int = 0

        self._attention_prior: torch.Tensor | None = None
        self._attention_prior_uses: int = 0
        self._total_focus_priors: int = 0

    def update(
        self,
        dissimilarity_map: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> None:
        """Add a per-patch dissimilarity map from one frame to the accumulator.

        Args:
            dissimilarity_map: Flat dissimilarity tensor ``(num_patches,)``
                as produced by :func:`~focus.gating.compute_patch_dissimilarity_map`.
            grid_h: Spatial grid height (patches after merge).
            grid_w: Spatial grid width (patches after merge).
        """
        if not self.enabled:
            return

        spatial = dissimilarity_map.detach().float().reshape(grid_h, grid_w)

        if self._accumulated is None or (grid_h, grid_w) != (self._grid_h, self._grid_w):
            self._grid_h = grid_h
            self._grid_w = grid_w
            self._accumulated = spatial
        else:
            self._accumulated = torch.maximum(self._accumulated, spatial)

        self._frame_count += 1
        self._total_updates += 1

    def set_attention_prior(self, attention_map: torch.Tensor) -> None:
        """Store a VLM-derived spatial attention map as the top-down prior.

        The prior persists across :meth:`reset` calls so it carries the
        VLM's spatial understanding forward to subsequent bursts.

        Args:
            attention_map: Normalised ``(grid_h, grid_w)`` heatmap produced
                by :class:`~focus.vlm.attention.AttentionMapExtractor`.
        """
        if not self.enabled:
            return
        self._attention_prior = attention_map.detach().float()

    def get_focus_prior(self) -> torch.Tensor | None:
        """Return the combined spatial map for value-cache calibration.

        The map is normalised to ``[0, 1]`` and shaped ``(grid_h, grid_w)``.
        Returns *None* on the first burst (before any attention prior is
        available), since there is no top-down signal to blend with yet.
        """
        if not self.enabled:
            return None
        if self._attention_prior is None:
            return None
        combined = self._compute_combined_map()
        if combined is None:
            return None
        self._total_focus_priors += 1
        return combined.detach().clone().cpu()

    def get_maps_snapshot(self) -> dict:
        """Return copies of the current spatial maps for visualization.

        Call this **before** :meth:`reset` to capture the state of the
        current burst.

        Returns:
            A dict with keys ``"accumulated"`` (raw dissimilarity heatmap),
            ``"attention_prior"`` (VLM attention prior, may be *None* for
            the first burst), and ``"combined"`` (the blended map).  All
            tensors are detached CPU copies.
        """
        accumulated = (
            self._accumulated.detach().clone().cpu() if self._accumulated is not None else None
        )
        attention_prior = (
            self._attention_prior.detach().clone().cpu()
            if self._attention_prior is not None
            else None
        )
        combined = self._compute_combined_map()
        if combined is not None:
            combined = combined.detach().clone().cpu()
        return {
            "accumulated": accumulated,
            "attention_prior": attention_prior,
            "combined": combined,
        }

    def reset(self) -> None:
        """Clear accumulated dissimilarity state for the current burst.

        The attention prior is intentionally preserved so the VLM's spatial
        understanding carries forward to the next burst.
        """
        self._accumulated = None
        self._frame_count = 0
        self._grid_h = 0
        self._grid_w = 0

    def get_summary(self) -> dict:
        """Snapshot of analyser state for logging / serialisation."""
        return {
            "enabled": self.enabled,
            "activation_threshold_sigma": self.activation_threshold_sigma,
            "attention_blend_alpha": self.attention_blend_alpha,
            "fusion_mode": self.fusion_mode,
            "fusion_gamma": self.fusion_gamma,
            "has_attention_prior": self._attention_prior is not None,
            "attention_prior_uses": self._attention_prior_uses,
            "total_updates": self._total_updates,
            "total_focus_priors": self._total_focus_priors,
            "current_burst_frames": self._frame_count,
            "current_grid": ((self._grid_h, self._grid_w) if self._grid_h else None),
        }

    def _compute_combined_map(self) -> torch.Tensor | None:
        """Blend bottom-up dissimilarity with the top-down attention prior.

        Uses multiplicative fusion by default: the attention prior modulates
        dissimilarity so that unattended regions are suppressed even when
        they have high pixel-level change.  Falls back to additive blending
        when ``fusion_mode="additive"``.

        Returns a normalised ``(grid_h, grid_w)`` heatmap, or *None* if no
        dissimilarity data has been accumulated.
        """
        if self._accumulated is None:
            return None

        dmap = self._accumulated.float()
        d_min, d_max = dmap.min(), dmap.max()
        if d_max - d_min > 1e-8:
            dmap = (dmap - d_min) / (d_max - d_min)
        else:
            dmap = torch.zeros_like(dmap)

        if self._attention_prior is None:
            return dmap

        prior = self._attention_prior
        if prior.shape != dmap.shape:
            prior = (
                F.interpolate(
                    prior.unsqueeze(0).unsqueeze(0),
                    size=dmap.shape,
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .squeeze(0)
            )

        prior = prior.to(dmap.device)
        self._attention_prior_uses += 1

        if self.fusion_mode == "additive":
            alpha = self.attention_blend_alpha
            combined = alpha * dmap + (1.0 - alpha) * prior
        else:
            eps = 0.01
            combined = dmap * (eps + prior**self.fusion_gamma)

        c_min, c_max = combined.min(), combined.max()
        if c_max - c_min > 1e-8:
            combined = (combined - c_min) / (c_max - c_min)

        return combined
