# SPDX-License-Identifier: Apache-2.0
"""Overlay drawing, text formatting, log-file generation, and heatmap visualizations."""

from __future__ import annotations

import textwrap
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
import torch

if TYPE_CHECKING:
    from ..config import FocusConfig
    from ..vlm.inference import BurstState, VLMResponse


def wrap_text(text: str, max_width: int = 60) -> list[str]:
    """Word-wrap *text* to *max_width* characters per line."""
    return textwrap.wrap(text, width=max(1, max_width), break_long_words=True)


def format_timestamp(seconds: float) -> str:
    """Format *seconds* as ``MM:SS.mmm``."""
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:06.3f}"


def _heatmap_to_frame_size(
    heatmap: torch.Tensor | np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Upscale a small ``(grid_h, grid_w)`` heatmap to ``(height, width)``.

    Uses full min-max normalisation so the coldest patch maps to 0 (blue)
    and the hottest maps to 1 (red), giving maximum contrast across the
    colourmap.  No gamma correction is applied -- the raw spatial
    structure is preserved.
    """
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().float().cpu().numpy()
    heatmap = heatmap.astype(np.float32)

    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max - h_min > 1e-8:
        heatmap = (heatmap - h_min) / (h_max - h_min)
    else:
        heatmap = np.zeros_like(heatmap)

    return cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)


def render_heatmap_overlay(
    frame: np.ndarray,
    heatmap: torch.Tensor | np.ndarray,
    alpha: float = 0.65,
    colormap: int = cv2.COLORMAP_TURBO,
) -> np.ndarray:
    """Blend a spatial heatmap onto a BGR frame.

    Args:
        frame: BGR image ``(H, W, 3)``.
        heatmap: ``(grid_h, grid_w)`` float map (any range -- will be
            normalised to ``[0, 1]``).
        alpha: Blending weight for the coloured heatmap.
        colormap: OpenCV colormap constant.

    Returns:
        Blended BGR image of the same shape as *frame*.
    """
    h, w = frame.shape[:2]
    resized = _heatmap_to_frame_size(heatmap, h, w)
    colored = cv2.applyColorMap((resized * 255).astype(np.uint8), colormap)
    blended = cv2.addWeighted(frame, 1.0 - alpha, colored, alpha, 0)
    return blended


def save_heatmap_overlay(
    frame: np.ndarray,
    heatmap: torch.Tensor | np.ndarray,
    path: str | Path,
    alpha: float = 0.65,
    title: str | None = None,
    colormap: int = cv2.COLORMAP_TURBO,
) -> None:
    """Render a heatmap overlay and write it to disk.

    Args:
        frame: BGR image ``(H, W, 3)``.
        heatmap: ``(grid_h, grid_w)`` spatial map.
        path: Output image file path.
        alpha: Blending weight.
        title: Optional title rendered in a banner at the top.
        colormap: OpenCV colormap constant.
    """
    blended = render_heatmap_overlay(frame, heatmap, alpha=alpha, colormap=colormap)
    if title:
        banner_h = 36
        canvas = np.zeros((blended.shape[0] + banner_h, blended.shape[1], 3), dtype=np.uint8)
        canvas[:banner_h] = (30, 30, 30)
        cv2.putText(
            canvas,
            title,
            (10, banner_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        canvas[banner_h:] = blended
        blended = canvas
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), blended)


def draw_overlay(
    frame: np.ndarray,
    should_use: bool,
    dissimilarity: float,
    metadata: dict,
    config: FocusConfig,
) -> np.ndarray:
    """Render a USE / GATE overlay on a video frame."""
    frame = frame.copy()
    h, w = frame.shape[:2]

    USE_COLOR = (0, 255, 0)
    GATE_COLOR = (0, 0, 255)
    color = USE_COLOR if should_use else GATE_COLOR
    status = "USE" if should_use else "GATE"

    cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.putText(frame, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 3)

    mode = metadata.get("mode", config.mode)
    cv2.putText(
        frame,
        f"[{mode.upper()}]",
        (150, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
    )

    state = metadata.get("state", "")
    reason = metadata.get("reason", "")
    burst_count = metadata.get("burst_count", 0)

    info1 = f"State: {state} | Dissim: {dissimilarity:.3f} | Bursts: {burst_count}"
    cv2.putText(frame, info1, (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    if mode == "frame":
        strategy = metadata.get("strategy", config.reference_update_strategy)
        info2 = f"Reason: {reason} | Strategy: {strategy}"
    else:
        info2 = f"Reason: {reason}"
    cv2.putText(frame, info2, (250, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    bar_y = h - 30
    cv2.rectangle(frame, (20, bar_y), (w - 20, bar_y + 15), (50, 50, 50), -1)
    fill = int((w - 40) * min(1.0, dissimilarity))
    cv2.rectangle(frame, (20, bar_y), (20 + fill, bar_y + 15), color, -1)
    thresh_x = 20 + int((w - 40) * config.threshold)
    cv2.line(frame, (thresh_x, bar_y - 3), (thresh_x, bar_y + 18), (255, 255, 0), 2)
    cv2.putText(
        frame,
        f"thresh={config.threshold:.2f}",
        (thresh_x - 30, bar_y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (255, 255, 0),
        1,
    )

    return frame


def draw_vlm_overlay(
    frame: np.ndarray,
    frame_idx: int,
    should_use: bool,
    dissimilarity: float,
    metadata: dict,
    config: FocusConfig,
    burst_state: BurstState,
    vlm_stats: dict,
) -> np.ndarray:
    """Render a rich overlay including VLM response panel."""
    frame = frame.copy()
    h, w = frame.shape[:2]

    USE_COLOR = (0, 255, 0)
    GATE_COLOR = (0, 0, 255)
    VLM_BG = (40, 40, 40)
    VLM_BORDER = (80, 80, 80)
    VLM_ACTIVE = (0, 200, 100)
    TEXT = (255, 255, 255)
    MUTED = (180, 180, 180)
    ACCENT = (255, 200, 0)
    PROCESSING = (100, 180, 255)

    color = USE_COLOR if should_use else GATE_COLOR
    status = "USE" if should_use else "GATE"

    # -- top banner --
    bh = 60
    cv2.rectangle(frame, (0, 0), (w, bh), (20, 20, 20), -1)
    cv2.putText(frame, status, (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)

    mode = metadata.get("mode", config.mode)
    cv2.putText(frame, f"[{mode.upper()}]", (120, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ACCENT, 2)

    state = metadata.get("state", "")
    reason = metadata.get("reason", "")
    cv2.putText(
        frame,
        f"State: {state} | {reason}",
        (220, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        MUTED,
        1,
    )
    burst_count = metadata.get("burst_count", 0)
    info2 = f"Dissim: {dissimilarity:.3f} | Bursts: {burst_count} | Frame: {frame_idx}"
    if burst_state.is_collecting():
        n = burst_state.collecting_end_frame - burst_state.collecting_start_frame + 1
        info2 += f" | Collecting: {n}/{config.burst_size}"
    cv2.putText(frame, info2, (220, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT, 1)

    # -- VLM response panel --
    pw = min(400, w // 2 - 20)
    px = w - pw - 10
    py = bh + 10
    ph = min(200, h - bh - 80)

    response = burst_state.last_response
    in_range = burst_state.current_frame_in_response_range(frame_idx)
    pending = burst_state.is_pending()

    if in_range and response:
        border = VLM_ACTIVE
    elif pending:
        border = PROCESSING
    else:
        border = VLM_BORDER

    cv2.rectangle(frame, (px - 2, py - 2), (px + pw + 2, py + ph + 2), border, -1)
    cv2.rectangle(frame, (px, py), (px + pw, py + ph), VLM_BG, -1)

    header_y = py + 18
    if in_range and response:
        ht = f"VLM Output [Frames {response.frame_start}-{response.frame_end}]"
        hc = VLM_ACTIVE
    elif pending:
        ht = f"VLM Processing Burst #{burst_state.pending_burst_id}..."
        hc = PROCESSING
    elif burst_state.is_collecting():
        ht = f"Collecting Burst #{burst_state.collecting_burst_id}..."
        hc = ACCENT
    else:
        ht = "VLM Output [Gating]"
        hc = MUTED
    cv2.putText(frame, ht, (px + 8, header_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, hc, 1)

    text_y = header_y + 22
    lh = 16
    mc = pw // 7
    show = response and (in_range or burst_state.is_collecting())

    if show:
        lines = wrap_text(response.text, max_width=mc)
        ml = (ph - 50) // lh
        for i, line in enumerate(lines[:ml]):
            y = text_y + i * lh
            if y + lh > py + ph - 20:
                break
            cv2.putText(frame, line, (px + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT, 1)
        if len(lines) > ml:
            cv2.putText(
                frame, "...", (px + 8, py + ph - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, MUTED, 1
            )
        meta_text = (
            f"Burst #{response.burst_id} | {response.inference_time_ms:.0f}ms | "
            f"Frames {response.frame_start}-{response.frame_end}"
        )
        cv2.putText(
            frame, meta_text, (px + 8, py + ph - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, MUTED, 1
        )
    elif pending:
        cv2.putText(
            frame,
            f"Processing frames {burst_state.pending_start_frame}-{burst_state.pending_end_frame}...",
            (px + 8, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            PROCESSING,
            1,
        )
        cv2.putText(
            frame,
            "VLM inference in progress",
            (px + 8, text_y + lh),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            MUTED,
            1,
        )
    elif burst_state.is_collecting():
        n = burst_state.collecting_end_frame - burst_state.collecting_start_frame + 1
        cv2.putText(
            frame,
            "Collecting burst frames...",
            (px + 8, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            ACCENT,
            1,
        )
        cv2.putText(
            frame,
            f"{n}/{config.burst_size} frames",
            (px + 8, text_y + lh),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            MUTED,
            1,
        )
    else:
        cv2.putText(
            frame,
            "Frame gated - no VLM analysis",
            (px + 8, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            MUTED,
            1,
        )
        if response:
            cv2.putText(
                frame,
                f"Last: Burst #{response.burst_id} (frames {response.frame_start}-{response.frame_end})",
                (px + 8, text_y + lh),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                MUTED,
                1,
            )

    # -- bottom stats --
    sy = h - 55
    cv2.rectangle(frame, (0, sy), (w, h), (20, 20, 20), -1)
    if vlm_stats:
        st = (
            f"VLM: {vlm_stats.get('total_inferences', 0)} inferences | "
            f"Avg: {vlm_stats.get('avg_inference_time_ms', 0):.0f}ms | "
            f"Queue: {vlm_stats.get('queue_size', 0)} | "
            f"Dropped: {vlm_stats.get('dropped_requests', 0)}"
        )
        cv2.putText(frame, st, (15, sy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, MUTED, 1)

    # -- threshold bar --
    by = h - 25
    bm = 15
    bw = w - 2 * bm
    cv2.rectangle(frame, (bm, by), (bm + bw, by + 12), (50, 50, 50), -1)
    fw = int(bw * min(1.0, dissimilarity))
    if fw > 0:
        cv2.rectangle(frame, (bm, by), (bm + fw, by + 12), color, -1)
    tx = bm + int(bw * config.threshold)
    cv2.line(frame, (tx, by - 3), (tx, by + 15), ACCENT, 2)
    cv2.putText(frame, "0", (bm, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, MUTED, 1)
    cv2.putText(
        frame,
        f"T={config.threshold:.2f}",
        (tx - 20, by - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        ACCENT,
        1,
    )
    cv2.putText(frame, "1", (bm + bw - 10, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, MUTED, 1)

    return frame


def save_convergence_figure(
    burst_snapshots: list[dict],
    output_path: str | Path,
    cell_width: int = 240,
    cell_height: int = 180,
    colormap: int = cv2.COLORMAP_TURBO,
    heatmap_alpha: float = 0.65,
) -> None:
    """Create a multi-panel grid showing how the spatial focus evolves.

    Each column represents one burst; rows show (top to bottom):

    1. Representative frame
    2. Bottom-up dissimilarity heatmap overlaid on the frame
    3. Top-down VLM attention prior overlaid on the frame
    4. Combined (blended) map overlaid on the frame

    Args:
        burst_snapshots: List of dicts, one per burst, each containing:
            ``"burst_id"`` (int), ``"frame"`` (BGR ndarray),
            ``"accumulated"`` (tensor or None), ``"attention_prior"``
            (tensor or None), ``"combined"`` (tensor or None).
        output_path: Where to save the figure.
        cell_width: Width of each cell in the grid.
        cell_height: Height of each cell in the grid.
        colormap: OpenCV colormap for heatmaps.
        heatmap_alpha: Blending weight for heatmap overlays.
    """
    if not burst_snapshots:
        return

    n_bursts = len(burst_snapshots)
    row_labels = ["Frame", "Dissimilarity", "VLM Attention", "Combined"]
    n_rows = len(row_labels)

    label_width = 130
    header_height = 32
    canvas_w = label_width + n_bursts * cell_width
    canvas_h = header_height + n_rows * cell_height

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)

    for col, snap in enumerate(burst_snapshots):
        cx = label_width + col * cell_width + cell_width // 2 - 30
        cv2.putText(
            canvas,
            f"Burst {snap['burst_id']}",
            (cx, header_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    for row, label in enumerate(row_labels):
        ly = header_height + row * cell_height + cell_height // 2 + 5
        cv2.putText(
            canvas,
            label,
            (8, ly),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
        )

    for col, snap in enumerate(burst_snapshots):
        frame = snap["frame"]
        thumb = cv2.resize(frame, (cell_width, cell_height), interpolation=cv2.INTER_AREA)

        def _place(row: int, img: np.ndarray) -> None:
            y0 = header_height + row * cell_height
            x0 = label_width + col * cell_width
            canvas[y0 : y0 + cell_height, x0 : x0 + cell_width] = img

        _place(0, thumb)

        for row, key in enumerate(["accumulated", "vlm_attention", "combined"], start=1):
            hmap = snap.get(key)
            if hmap is not None:
                overlay = render_heatmap_overlay(
                    thumb,
                    hmap,
                    alpha=heatmap_alpha,
                    colormap=colormap,
                )
                _place(row, overlay)
            else:
                placeholder = thumb.copy() // 3
                cv2.putText(
                    placeholder,
                    "N/A",
                    (cell_width // 2 - 20, cell_height // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (120, 120, 120),
                    2,
                )
                _place(row, placeholder)

    # Thin grid lines
    for row in range(n_rows + 1):
        y = header_height + row * cell_height
        cv2.line(canvas, (label_width, y), (canvas_w, y), (70, 70, 70), 1)
    for col in range(n_bursts + 1):
        x = label_width + col * cell_width
        cv2.line(canvas, (x, header_height), (x, canvas_h), (70, 70, 70), 1)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def save_vlm_log(
    log_path: str,
    input_path: str,
    output_path: str,
    vlm_responses: list[VLMResponse],
    focus_stats: dict,
    vlm_stats: dict,
    perf_stats: dict,
    frame_skipping: dict,
    resizing: dict,
    config: FocusConfig,
    fps: float,
    total_frames: int,
    vlm_prompt: str,
    context_summary: Optional[dict] = None,
    spatial_summary: Optional[dict] = None,
    attention_summary: Optional[dict] = None,
) -> None:
    """Write a human-readable log of FOCUS + VLM processing results."""
    with open(log_path, "w") as f:
        f.write("=" * 80 + "\n  FOCUS + VLM Processing Log\n")
        if context_summary and context_summary.get("enabled"):
            f.write("  (with Temporal Context Tracking)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        _section(f, "FILES")
        f.write(f"Input:  {input_path}\nOutput: {output_path}\nLog:    {log_path}\n\n")

        _section(f, "VIDEO INFO")
        dur = total_frames / fps if fps > 0 else 0
        f.write(f"Duration:     {format_timestamp(dur)} ({dur:.2f}s)\n")
        f.write(f"Total Frames: {total_frames}\nOriginal FPS: {fps:.2f}\n")
        if resizing.get("enabled"):
            f.write(
                f"Resolution:   {resizing['input_width']}x{resizing['input_height']}"
                f" -> {resizing['output_width']}x{resizing['output_height']}\n"
            )
        if frame_skipping.get("enabled"):
            f.write(
                f"Frame Skip:   {frame_skipping['input_fps']:.1f} fps -> "
                f"{frame_skipping['output_fps']:.1f} fps\n"
                f"              {frame_skipping['frames_processed']} processed, "
                f"{frame_skipping['frames_skipped']} skipped\n"
            )
        f.write("\n")

        if context_summary and context_summary.get("task_description"):
            _section(f, "TASK CONTEXT")
            f.write(f"{context_summary['task_description']}\n\n")

        _section(f, "FOCUS CONFIGURATION")
        f.write(f"Mode:            {config.mode}\n")
        f.write(f"Threshold:       {config.threshold}\n")
        f.write(f"Burst Enabled:   {config.burst_enabled}\n")
        f.write(f"Burst Size:      {config.burst_size}\n")
        f.write(f"Max Gate Frames: {config.max_gate_frames}\n")
        if config.mode == "frame":
            f.write(f"Ref Strategy:    {config.reference_update_strategy}\n")
            if config.reference_update_strategy == "ema":
                f.write(f"EMA Alpha:       {config.ema_alpha}\n")
        f.write("\n")

        _section(f, "VLM CONFIGURATION")
        f.write(f'Base Prompt: "{vlm_prompt}"\n')
        if context_summary and context_summary.get("enabled"):
            f.write(
                f"Context Tracking: Enabled (window={context_summary.get('context_window', 5)})\n"
            )
        else:
            f.write("Context Tracking: Disabled\n")
        if spatial_summary and spatial_summary.get("enabled"):
            f.write(
                f"Spatial Attention: Enabled "
                f"(updates={spatial_summary['total_updates']}, "
                f"focus_priors={spatial_summary.get('total_focus_priors', 0)})\n"
            )
            if spatial_summary.get("has_attention_prior"):
                f.write(
                    f"  Attention Prior: active "
                    f"(fusion={spatial_summary.get('fusion_mode', 'multiplicative')}, "
                    f"uses={spatial_summary.get('attention_prior_uses', 0)})\n"
                )
        else:
            f.write("Spatial Attention: Disabled\n")
        if attention_summary and attention_summary.get("enabled"):
            f.write(
                f"Attention Feedback: Enabled "
                f"(layers={attention_summary['num_layers']}, "
                f"extractions={attention_summary['total_extractions']}, "
                f"avg={attention_summary['avg_extraction_time_ms']:.0f}ms)\n"
            )
        else:
            f.write("Attention Feedback: Disabled\n")
        f.write("\n")

        _section(f, "FOCUS RESULTS")
        f.write(f"Frames Processed: {focus_stats['total_frames']}\n")
        f.write(
            f"Captured (USE):   {focus_stats['captured_frames']} ({100 * focus_stats['capture_ratio']:.1f}%)\n"
        )
        f.write(
            f"Gated (GATE):     {focus_stats['gated_frames']} ({100 * focus_stats['gate_ratio']:.1f}%)\n"
        )
        f.write(f"Total Bursts:     {focus_stats['burst_count']}\n\n")

        _section(f, "VLM INFERENCE SUMMARY")
        f.write(f"Total Inferences:   {vlm_stats['total_inferences']}\n")
        f.write(f"Dropped Requests:   {vlm_stats['dropped_requests']}\n")
        f.write(f"Avg Inference Time: {vlm_stats['avg_inference_time_ms']:.0f}ms\n\n")

        _section(f, "PERFORMANCE")
        f.write(f"Total Processing Time: {perf_stats['total_time_s']:.1f}s\n")
        f.write(f"Average FPS:           {perf_stats['avg_fps']:.2f}\n")
        f.write(f"Average Latency:       {perf_stats['avg_latency_ms']:.1f}ms/frame\n")
        f.write(f"RAM Used:              {perf_stats.get('ram_used_gb', 0):.2f} GB\n")
        if "gpu_mem_used_gb" in perf_stats:
            f.write(f"GPU Memory Used:       {perf_stats['gpu_mem_used_gb']:.2f} GB\n")
        f.write("\n")

        if context_summary and context_summary.get("enabled") and context_summary.get("history"):
            f.write("=" * 80 + "\n  NARRATIVE SUMMARY\n" + "=" * 80 + "\n\n")
            for entry in context_summary["history"]:
                f.write(f"  Burst {entry['burst_id']}: {entry['summary']}\n")
            f.write("\n")

        f.write("=" * 80 + "\n  VLM RESPONSES BY BURST\n" + "=" * 80 + "\n\n")
        if not vlm_responses:
            f.write("No VLM responses recorded.\n")
        else:
            eff_fps = (
                frame_skipping.get("output_fps", fps) if frame_skipping.get("enabled") else fps
            )
            for resp in vlm_responses:
                st = resp.frame_start / eff_fps if eff_fps > 0 else 0
                et = resp.frame_end / eff_fps if eff_fps > 0 else 0
                f.write(f"  BURST #{resp.burst_id}\n")
                f.write(f"  Frames:    {resp.frame_start} - {resp.frame_end}\n")
                f.write(f"  Timestamp: {format_timestamp(st)} - {format_timestamp(et)}\n")
                f.write(f"  Inference: {resp.inference_time_ms:.0f}ms\n")
                f.write("  Response:\n")
                for line in textwrap.wrap(resp.text, width=74):
                    f.write(f"    {line}\n")
                f.write("\n")

        f.write("=" * 80 + "\n  END OF LOG\n" + "=" * 80 + "\n")


def _section(f, title: str) -> None:
    f.write("-" * 40 + "\n")
    f.write(f"  {title}\n")
    f.write("-" * 40 + "\n")
