# SPDX-License-Identifier: Apache-2.0
"""End-to-end video processing pipelines and CLI entry points."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

from ..config import FocusConfig
from ..encoders.base import BaseEncoder
from ..gating import create_gating
from ..monitor import PerformanceMonitor
from ..vlm.base import BaseVLMPipeline
from ..vlm.context import VLMContextManager
from ..vlm.inference import BurstState, VLMInferenceQueue
from ..vlm.spatial import SpatialAttentionAnalyzer
from .io import compress_video
from .visualization import (
    draw_overlay,
    draw_vlm_overlay,
    save_convergence_figure,
    save_heatmap_overlay,
    save_vlm_log,
)


def process_video(
    input_path: str,
    output_path: str,
    config: FocusConfig,
    encoder: BaseEncoder,
    show_preview: bool = False,
    compress: bool = True,
    crf: int = 23,
    show_perf_stats: bool = True,
    target_fps: Optional[float] = 24.0,
    target_height: Optional[int] = 480,
) -> dict:
    """Process a video with FOCUS.

    Each frame is encoded via *encoder*, passed through the gating logic,
    and annotated with a USE/GATE overlay before being written to
    *output_path*.

    Args:
        input_path: Source video file.
        output_path: Destination video file.
        config: Gating configuration.
        encoder: Any :class:`BaseEncoder` implementation.
        show_preview: Show an OpenCV preview window.
        compress: Re-encode with FFmpeg after processing.
        crf: FFmpeg CRF quality value.
        show_perf_stats: Display real-time stats in the progress bar.
        target_fps: Down-sample to this frame rate (``None`` to disable).
        target_height: Resize to this height (``None`` to disable).

    Returns:
        A dict of statistics (gating, performance, frame-skipping, resizing).
    """
    if Path(input_path).resolve() == Path(output_path).resolve():
        raise ValueError("Input and output video paths must differ")
    if target_fps is not None and target_fps <= 0:
        raise ValueError("target_fps must be positive")
    if target_height is not None and target_height < 2:
        raise ValueError("target_height must be at least 2")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    gate = create_gating(config)
    perf = PerformanceMonitor(window_size=30, device=encoder.device)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError(f"Video has invalid frame rate: {input_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    skip, step, out_fps = False, 1.0, fps
    if target_fps is not None and fps > target_fps:
        skip, step, out_fps = True, fps / target_fps, target_fps

    resize = False
    out_w, out_h = width, height
    if target_height is not None and height > target_height:
        resize = True
        scale = target_height / height
        out_h = target_height
        out_w = int(width * scale)
        out_w += out_w % 2  # ensure even

    print(f"\nInput: {input_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps:.1f}, Frames: {total_frames}")
    if resize:
        print(f"Resizing: {width}x{height} -> {out_w}x{out_h}")
    if skip:
        est = int(total_frames / step)
        print(f"Frame skipping: {fps:.1f} -> {out_fps:.1f} fps (step={step:.2f}, ~{est} frames)")
    print(f"Mode: {config.mode}")
    print(
        f"Config: threshold={config.threshold}, burst_enabled={config.burst_enabled}, burst_size={config.burst_size}"
    )
    if config.mode == "frame":
        extra = ""
        if config.reference_update_strategy == "ema":
            extra = f", ema_alpha={config.ema_alpha}"
        elif config.reference_update_strategy == "periodic":
            extra = f", update_interval={config.reference_update_interval}"
        print(f"Frame mode: strategy={config.reference_update_strategy}{extra}")

    temp = output_path + ".temp.mp4" if compress else output_path
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp, fourcc, out_fps, (out_w, out_h))
    if not out.isOpened():
        cap.release()
        raise ValueError(f"Could not create output video: {temp}")

    pbar = tqdm(total=total_frames, desc="Processing", dynamic_ncols=True)
    idx, next_proc, processed, skipped = 0, 0.0, 0, 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if skip and idx < next_proc:
                skipped += 1
                idx += 1
                pbar.update(1)
                continue
            if skip:
                next_proc += step
            if resize:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

            perf.start_frame()
            perf.start_encode()
            emb = encoder(frame)
            perf.end_encode()

            use, dissim, meta = gate.process_frame(emb)
            annotated = draw_overlay(frame, use, dissim, meta, config)
            out.write(annotated)
            processed += 1
            perf.end_frame()

            if show_perf_stats:
                pbar.set_postfix_str(perf.format_stats_line())
            if show_preview:
                cv2.imshow("FOCUS", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            idx += 1
            pbar.update(1)

    finally:
        pbar.close()
        cap.release()
        out.release()
        if show_preview:
            cv2.destroyAllWindows()

    if compress:
        print("Compressing video with FFmpeg...")
        if compress_video(temp, output_path, crf=crf):
            os.remove(temp)
            in_mb = os.path.getsize(input_path) / (1024 * 1024)
            out_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  Input: {in_mb:.1f} MB -> Output: {out_mb:.1f} MB")
        else:
            shutil.move(temp, output_path)

    stats = gate.get_stats()
    perf_summary = perf.get_summary()

    print(f"\n{'=' * 60}")
    print("Results:")
    print(f"  Mode: {stats.get('mode', config.mode)}")
    if resize:
        print(f"  Resolution: {width}x{height} -> {out_w}x{out_h}")
    if skip:
        print(f"  Input frames: {total_frames} ({fps:.1f} fps)")
        print(f"  After skip: {processed} ({out_fps:.1f} fps), skipped {skipped}")
    print(f"  Total frames processed: {stats['total_frames']}")
    print(f"  Captured (USE): {stats['captured_frames']} ({100 * stats['capture_ratio']:.1f}%)")
    print(f"  Gated (GATE): {stats['gated_frames']} ({100 * stats['gate_ratio']:.1f}%)")
    if stats.get("burst_enabled", config.burst_enabled):
        print(f"  Bursts: {stats['burst_count']}")
    if config.mode == "frame":
        print(
            f"  Reference strategy: {stats.get('reference_strategy', config.reference_update_strategy)}"
        )
    print(f"  Output: {output_path}")
    print("\nPerformance:")
    print(f"  Total processing time: {perf_summary['total_time_s']:.1f}s")
    print(f"  Average FPS: {perf_summary['avg_fps']:.2f}")
    print(f"  Average latency: {perf_summary['avg_latency_ms']:.1f}ms/frame")
    print(f"  Average encode time: {perf_summary['avg_encode_ms']:.1f}ms/frame")
    print(f"  RAM used: {perf_summary.get('ram_used_gb', 0):.2f} GB")
    if perf.gpu_available:
        if "gpu_mem_used_gb" in perf_summary:
            print(f"  GPU memory used: {perf_summary['gpu_mem_used_gb']:.2f} GB")
        elif "gpu_mem_allocated_gb" in perf_summary:
            print(f"  GPU memory allocated: {perf_summary['gpu_mem_allocated_gb']:.2f} GB")
    print(f"{'=' * 60}")

    stats["performance"] = perf_summary
    stats["frame_skipping"] = (
        {
            "enabled": True,
            "input_fps": fps,
            "output_fps": out_fps,
            "input_frames": total_frames,
            "frames_processed": processed,
            "frames_skipped": skipped,
        }
        if skip
        else {"enabled": False}
    )
    stats["resizing"] = (
        {
            "enabled": True,
            "input_width": width,
            "input_height": height,
            "output_width": out_w,
            "output_height": out_h,
        }
        if resize
        else {"enabled": False}
    )
    return stats


def process_video_with_vlm(
    input_path: str,
    output_path: str,
    config: FocusConfig,
    vlm: BaseVLMPipeline,
    encoder: BaseEncoder | BaseVLMPipeline | None = None,
    vlm_prompt: str = "Describe what is happening in these video frames briefly.",
    max_new_tokens: int = 100,
    max_queue_size: int = 2,
    do_sample: bool = False,
    show_preview: bool = False,
    compress: bool = True,
    crf: int = 23,
    target_fps: float | None = 24.0,
    target_height: int | None = 480,
    save_log: bool = True,
    log_path: str | None = None,
    task_description: str = "",
    context_window: int = 5,
    enable_context: bool = True,
    enable_spatial: bool = True,
    enable_attention_feedback: bool = True,
    attention_blend_alpha: float = 0.5,
    fusion_mode: str = "multiplicative",
    fusion_gamma: float = 0.5,
    focus_alpha: float = 1.02,
    focus_range: tuple[float, float] = (0.0, 1.0),
    focus_suppress: float = 0.0,
    save_viz: bool = False,
    viz_dir: str | None = None,
) -> dict:
    """Process a video with FOCUS gating **and** VLM text generation.

    Burst generation runs synchronously so no burst is dropped and feedback
    from burst n is available before burst n+1. The response is
    rendered on the output video.  When spatial attention analysis and
    attention feedback are enabled, the combined dissimilarity + attention
    prior is used to calibrate the VLM's value cache during generation,
    focusing the model on the task-relevant region.

    Args:
        input_path: Source video file.
        output_path: Destination video file.
        config: Gating configuration.
        vlm: Any :class:`BaseVLMPipeline` implementation (used for
            inference and, when *encoder* is ``None``, also embedding).
        encoder: Optional separate :class:`BaseEncoder` for frame
            embedding.  When provided, ``encoder(frame)`` is used for
            patch dissimilarity instead of ``vlm.encode_frame(frame)``.
        vlm_prompt: Base prompt for generation.
        max_new_tokens: Token budget per generation call.
        max_queue_size: Retained for API compatibility; synchronous execution
            does not queue or drop bursts.
        show_preview: Show an OpenCV preview window.
        compress: Re-encode with FFmpeg.
        crf: FFmpeg CRF quality.
        target_fps: Down-sample to this rate (``None`` to disable).
        target_height: Resize to this height (``None`` to disable).
        save_log: Write a human-readable log file.
        log_path: Log file path (defaults to output with ``.txt`` suffix).
        task_description: Task description for temporal context.
        context_window: How many previous responses in the prompt.
        enable_context: Enable temporal context tracking.
        enable_spatial: Enable spatial attention analysis and the
            attention feedback loop for value-cache calibration.
        enable_attention_feedback: Extract VLM attention maps after each
            generation and feed them back into the spatial analyser as
            a top-down prior for subsequent bursts.
        attention_blend_alpha: Blending weight for the spatial analyser
            (only used in additive fusion mode).
        fusion_mode: ``"multiplicative"`` (default) or ``"additive"``.
        fusion_gamma: Exponent for attention sharpness in multiplicative
            fusion.
        focus_alpha: Per-step value-cache scale factor for spatial focus
            calibration.  Set to ``1.0`` to disable.
        focus_range: ``(low, high)`` prior-value window.  Only vision
            tokens with prior in ``[low, high]`` are scaled.
        save_viz: Save per-burst attention heatmap overlays and a
            convergence figure showing how the spatial focus evolves.
        viz_dir: Directory for visualization outputs.  Defaults to
            ``{output_stem}_viz/``.

    Returns:
        A dict of FOCUS, VLM, performance, and context statistics.
    """
    if Path(input_path).resolve() == Path(output_path).resolve():
        raise ValueError("Input and output video paths must differ")
    if target_fps is not None and target_fps <= 0:
        raise ValueError("target_fps must be positive")
    if target_height is not None and target_height < 2:
        raise ValueError("target_height must be at least 2")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    gate = create_gating(config)

    ctx = VLMContextManager(
        task_description=task_description,
        context_window=context_window,
        base_prompt=vlm_prompt,
        enabled=enable_context,
    )
    spatial = (
        SpatialAttentionAnalyzer(
            enabled=enable_spatial,
            attention_blend_alpha=attention_blend_alpha,
            fusion_mode=fusion_mode,
            fusion_gamma=fusion_gamma,
        )
        if enable_spatial
        else None
    )

    viz_snapshots: list[dict] = []
    viz_burst_frames_cache: dict[int, np.ndarray] = {}
    if save_viz:
        if viz_dir is None:
            viz_dir = str(Path(output_path).with_suffix("")) + "_viz"
        Path(viz_dir).mkdir(parents=True, exist_ok=True)
        print(f"Visualizations: {viz_dir}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError(f"Video has invalid frame rate: {input_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    skip, step, out_fps = False, 1.0, fps
    if target_fps is not None and fps > target_fps:
        skip, step, out_fps = True, fps / target_fps, target_fps

    for name, value in (
        ("focus_alpha", focus_alpha),
        ("focus_range", focus_range),
        ("focus_suppress", focus_suppress),
    ):
        if hasattr(vlm, name):
            setattr(vlm, name, value)

    use_attn_feedback = enable_attention_feedback and enable_spatial
    vlm_q = VLMInferenceQueue(
        vlm=vlm,
        max_queue_size=max_queue_size,
        prompt=vlm_prompt,
        max_new_tokens=max_new_tokens,
        context_manager=ctx,
        video_fps=out_fps,
        do_sample=do_sample,
        enable_attention_feedback=use_attn_feedback,
        synchronous=True,
    )
    vlm_q.start()

    perf = PerformanceMonitor(
        window_size=30,
        device=encoder.device if encoder is not None else vlm.device,
    )

    resize = False
    out_w, out_h = width, height
    if target_height is not None and height > target_height:
        resize = True
        scale = target_height / height
        out_h = target_height
        out_w = int(width * scale)
        out_w += out_w % 2

    print(f"\nInput: {input_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps:.1f}, Frames: {total_frames}")
    if resize:
        print(f"Resizing: {width}x{height} -> {out_w}x{out_h}")
    if skip:
        print(f"Frame skipping: {fps:.1f} -> {out_fps:.1f} fps (step={step:.2f})")
    print(f"Mode: {config.mode}, threshold={config.threshold}, burst_size={config.burst_size}")
    print(f"VLM: max_tokens={max_new_tokens}, synchronous feedback")
    if enable_context:
        print(f"Context: enabled (window={context_window})")
        if task_description:
            print(f"Task: {task_description}")
    if enable_spatial:
        fb = " + attention feedback" if use_attn_feedback else ""
        fr = (
            f", range={focus_range[0]:.2f}-{focus_range[1]:.2f}"
            if focus_range != (0.0, 1.0)
            else ""
        )
        fs = f", suppress={focus_suppress}" if focus_suppress > 0.0 else ""
        fa = (
            f" + focus calibration (alpha={focus_alpha}{fr}{fs})"
            if focus_alpha > 1.0 or focus_suppress > 0.0
            else ""
        )
        print(f"Spatial attention: enabled{fb}{fa}")

    temp = output_path + ".temp.mp4" if compress else output_path
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp, fourcc, out_fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        vlm_q.stop()
        raise ValueError(f"Could not create output video: {temp}")

    burst_frames: list[np.ndarray] = []
    bstate = BurstState()
    fidx = 0
    prev_state = None

    in_idx, next_proc, processed, skipped_n = 0, 0.0, 0, 0
    pbar = tqdm(total=total_frames, desc="Processing", dynamic_ncols=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if skip and in_idx < next_proc:
                skipped_n += 1
                in_idx += 1
                pbar.update(1)
                continue
            if skip:
                next_proc += step
            if resize:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

            perf.start_frame()
            perf.start_encode()
            _enc = encoder if encoder is not None else vlm
            emb = _enc.encode_frame(frame) if isinstance(_enc, BaseVLMPipeline) else _enc(frame)
            perf.end_encode()

            use, dissim, meta = gate.process_frame(emb)
            cur_state = meta.get("state", "")

            if use:
                if not bstate.is_collecting():
                    bstate.start_burst(bstate.collecting_burst_id, fidx)
                burst_frames.append(frame.copy())
                bstate.add_frame(fidx)

                if spatial is not None and meta.get("dissimilarity_map") is not None:
                    _enc = encoder if encoder is not None else vlm
                    grid_hw = getattr(_enc, "last_grid_hw", None)
                    if grid_hw is not None:
                        spatial.update(meta["dissimilarity_map"], *grid_hw)

                done = (
                    not config.burst_enabled
                    or len(burst_frames) >= config.burst_size
                    or (prev_state == "CAPTURE" and cur_state == "WATCH")
                )
                if done and burst_frames:
                    spatial_prior = None
                    if spatial is not None:
                        spatial_prior = spatial.get_focus_prior()

                        if save_viz:
                            mid = burst_frames[len(burst_frames) // 2]
                            snap = spatial.get_maps_snapshot()
                            snap["burst_id"] = bstate.collecting_burst_id
                            snap["frame"] = mid.copy()
                            viz_snapshots.append(snap)
                            viz_burst_frames_cache[bstate.collecting_burst_id] = mid.copy()

                        spatial.reset()

                    submitted = vlm_q.submit_burst(
                        bstate.collecting_burst_id,
                        list(burst_frames),
                        bstate.collecting_start_frame,
                        bstate.collecting_end_frame,
                        spatial_prior=spatial_prior,
                    )
                    if submitted:
                        bstate.submit_burst()
                    else:
                        bstate.collecting_burst_id += 1
                        bstate.collecting_start_frame = -1
                        bstate.collecting_end_frame = -1
                    burst_frames = []

            new_resp = vlm_q.check_new_response()
            if new_resp:
                bstate.receive_response(new_resp)
                if spatial is not None and new_resp.attention_map is not None:
                    spatial.set_attention_prior(new_resp.attention_map)

                if save_viz and new_resp.attention_map is not None:
                    rep_frame = viz_burst_frames_cache.pop(new_resp.burst_id, None)
                    if rep_frame is not None:
                        save_heatmap_overlay(
                            rep_frame,
                            new_resp.attention_map,
                            Path(viz_dir) / f"burst_{new_resp.burst_id:03d}_attention.png",
                            title=f"Burst {new_resp.burst_id} -- VLM Attention",
                        )
                    for snap in viz_snapshots:
                        if snap["burst_id"] == new_resp.burst_id:
                            snap["vlm_attention"] = new_resp.attention_map
                            break

            annotated = draw_vlm_overlay(
                frame,
                fidx,
                use,
                dissim,
                meta,
                config,
                bstate,
                vlm_q.get_stats(),
            )
            writer.write(annotated)
            perf.end_frame()
            pbar.set_postfix_str(perf.format_stats_line())
            processed += 1

            if show_preview:
                cv2.imshow("FOCUS + VLM", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            prev_state = cur_state
            in_idx += 1
            pbar.update(1)
            fidx += 1

    finally:
        pbar.close()
        cap.release()
        writer.release()
        if show_preview:
            cv2.destroyAllWindows()

    # Keep a partial final burst instead of silently discarding the end of a clip.
    if burst_frames:
        spatial_prior = spatial.get_focus_prior() if spatial is not None else None
        vlm_q.submit_burst(
            bstate.collecting_burst_id,
            list(burst_frames),
            bstate.collecting_start_frame,
            bstate.collecting_end_frame,
            spatial_prior=spatial_prior,
        )
        bstate.submit_burst()
        response = vlm_q.check_new_response()
        if response is not None:
            bstate.receive_response(response)
            if spatial is not None and response.attention_map is not None:
                spatial.set_attention_prior(response.attention_map)

    vlm_q.stop()

    if save_viz and viz_snapshots:
        convergence_path = Path(viz_dir) / "convergence.png"
        save_convergence_figure(viz_snapshots, convergence_path)
        print(f"Convergence figure saved to: {convergence_path}")

    if compress:
        print("Compressing video with FFmpeg...")
        if compress_video(temp, output_path, crf=crf):
            os.remove(temp)
            in_mb = os.path.getsize(input_path) / (1024 * 1024)
            out_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  Input: {in_mb:.1f} MB -> Output: {out_mb:.1f} MB")
        else:
            shutil.move(temp, output_path)

    focus_stats = gate.get_stats()
    perf_summary = perf.get_summary()
    vlm_final = vlm_q.get_stats()
    vlm_responses = vlm_q.get_all_responses()

    print(f"\n{'=' * 60}")
    print("Results:")
    print(f"  Mode: {focus_stats.get('mode', config.mode)}")
    print(
        f"  Captured (USE): {focus_stats['captured_frames']} ({100 * focus_stats['capture_ratio']:.1f}%)"
    )
    print(f"  Gated (GATE): {focus_stats['gated_frames']} ({100 * focus_stats['gate_ratio']:.1f}%)")
    print(f"  Bursts: {focus_stats['burst_count']}")
    print(
        f"\nVLM: {vlm_final['total_inferences']} inferences, avg {vlm_final['avg_inference_time_ms']:.0f}ms"
    )
    print(
        f"Performance: {perf_summary['avg_fps']:.2f} fps, {perf_summary['total_time_s']:.1f}s total"
    )
    print(f"{'=' * 60}")

    fs = (
        {
            "enabled": True,
            "input_fps": fps,
            "output_fps": out_fps,
            "input_frames": total_frames,
            "frames_processed": processed,
            "frames_skipped": skipped_n,
        }
        if skip
        else {"enabled": False}
    )
    rs = (
        {
            "enabled": True,
            "input_width": width,
            "input_height": height,
            "output_width": out_w,
            "output_height": out_h,
        }
        if resize
        else {"enabled": False}
    )

    spatial_summary = spatial.get_summary() if spatial else None
    attention_summary = (
        vlm.attention_extractor.get_stats()
        if use_attn_feedback and hasattr(vlm, "attention_extractor")
        else None
    )

    if save_log:
        if log_path is None:
            log_path = str(Path(output_path).with_suffix(".txt"))
        save_vlm_log(
            log_path=log_path,
            input_path=input_path,
            output_path=output_path,
            vlm_responses=vlm_responses,
            focus_stats=focus_stats,
            vlm_stats=vlm_final,
            perf_stats=perf_summary,
            frame_skipping=fs,
            resizing=rs,
            config=config,
            fps=fps,
            total_frames=total_frames,
            vlm_prompt=vlm_prompt,
            context_summary=ctx.get_summary(),
            spatial_summary=spatial_summary,
            attention_summary=attention_summary,
        )
        print(f"\nLog saved to: {log_path}")

    return {
        "focus": focus_stats,
        "vlm": vlm_final,
        "vlm_responses": [r.to_dict() for r in vlm_responses],
        "performance": perf_summary,
        "frame_skipping": fs,
        "resizing": rs,
        "context": ctx.get_summary(),
        "spatial": spatial_summary,
        "attention_feedback": attention_summary,
        "log_path": log_path if save_log else None,
        "viz_dir": viz_dir if save_viz else None,
    }


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Arguments shared by both CLIs."""
    parser.add_argument(
        "-m",
        "--mode",
        choices=["patch", "frame"],
        default="patch",
        help="Gating mode (default: patch)",
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.3, help="Dissimilarity threshold (default: 0.3)"
    )
    parser.add_argument(
        "--max-gate", type=int, default=300, help="Max frames to gate before forced capture"
    )
    parser.add_argument("-b", "--burst-size", type=int, default=8, help="Frames per burst")
    parser.add_argument("--no-burst", action="store_true", help="Disable burst capture mode")
    parser.add_argument(
        "-s",
        "--strategy",
        choices=["replace", "ema", "periodic"],
        default="replace",
        help="Reference update strategy (frame mode)",
    )
    parser.add_argument("--ema-alpha", type=float, default=0.7, help="EMA alpha")
    parser.add_argument("--update-interval", type=int, default=10, help="Periodic update interval")
    parser.add_argument(
        "--dissimilarity-method",
        choices=["mean", "max", "ratio"],
        default="mean",
        help="Patch dissimilarity aggregation",
    )
    parser.add_argument(
        "--embed-model",
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="Model for frame embedding / patch dissimilarity",
    )
    parser.add_argument(
        "--attention-model",
        default=None,
        help="Model for attention extraction (default: same as --inference-model)",
    )
    parser.add_argument(
        "--inference-model",
        default=None,
        help="Model for VLM text generation (default: same as --embed-model)",
    )
    parser.add_argument("--device", default=None, help="Device (default: auto)")
    parser.add_argument("--preview", action="store_true", help="Show preview window")
    parser.add_argument("--no-compress", action="store_true", help="Skip FFmpeg compression")
    parser.add_argument("--crf", type=int, default=23, help="FFmpeg CRF quality")
    parser.add_argument("--target-fps", type=float, default=24.0, help="Target FPS")
    parser.add_argument("--no-fps-limit", action="store_true", help="Process all frames")
    parser.add_argument("--target-height", type=int, default=480, help="Target resolution height")
    parser.add_argument("--no-resize", action="store_true", help="Keep original resolution")


def _make_config(args) -> FocusConfig:
    return FocusConfig(
        mode=args.mode,
        threshold=args.threshold,
        max_gate_frames=args.max_gate,
        burst_enabled=not args.no_burst,
        burst_size=args.burst_size,
        reference_update_strategy=args.strategy,
        reference_update_interval=args.update_interval,
        ema_alpha=args.ema_alpha,
        dissimilarity_method=args.dissimilarity_method,
    )


def focus_gate_cli() -> None:
    """Console-script entry point: ``focus-gate``."""
    parser = argparse.ArgumentParser(
        description="FOCUS -- temporal redundancy detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Input video file")
    parser.add_argument("-o", "--output", default=None, help="Output video file")
    parser.add_argument(
        "--no-perf-stats", action="store_true", help="Hide real-time performance stats"
    )
    _add_common_args(parser)
    args = parser.parse_args()

    if args.output is None:
        p = Path(args.input)
        args.output = str(p.parent / f"{p.stem}_focus_{args.mode}.mp4")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from ..encoders import load_encoder

    encoder = load_encoder(args.embed_model, device=device)

    process_video(
        input_path=args.input,
        output_path=args.output,
        config=_make_config(args),
        encoder=encoder,
        show_preview=args.preview,
        compress=not args.no_compress,
        crf=args.crf,
        show_perf_stats=not args.no_perf_stats,
        target_fps=None if args.no_fps_limit else args.target_fps,
        target_height=None if args.no_resize else args.target_height,
    )


def focus_video_cli() -> None:
    """Console-script entry point: ``focus-video``."""
    parser = argparse.ArgumentParser(
        description="FOCUS + VLM integration pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Input video file")
    parser.add_argument("-o", "--output", default=None, help="Output video file")
    _add_common_args(parser)
    parser.set_defaults(burst_size=32)

    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe what is happening in these video frames briefly.",
        help="VLM prompt",
    )
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument(
        "--do-sample", action="store_true", help="Use sampling instead of greedy decoding"
    )
    parser.add_argument("--log", type=str, default=None, help="Log file path")
    parser.add_argument("--no-log", action="store_true", help="Disable log file")

    parser.add_argument("--task", type=str, default="", help="Task description for context")
    parser.add_argument("--context-window", type=int, default=5, help="Context window size")
    parser.add_argument(
        "--no-context", action="store_true", help="Disable temporal context tracking"
    )
    parser.add_argument(
        "--no-spatial", action="store_true", help="Disable spatial attention analysis"
    )
    parser.add_argument(
        "--no-attention-feedback", action="store_true", help="Disable VLM attention feedback loop"
    )
    parser.add_argument(
        "--focus-alpha",
        type=float,
        default=1.02,
        help="Value-cache focus calibration strength (1.0=disabled, default: 1.02)",
    )
    parser.add_argument(
        "--focus-range",
        type=str,
        default="0.0-1.0",
        help="Prior-value range for token selection, e.g. '0.2-0.8' (default: 0.0-1.0)",
    )
    parser.add_argument(
        "--focus-suppress",
        type=float,
        default=0.0,
        help="Per-step suppression for tokens outside focus-range (0.0=off, try 0.02)",
    )
    parser.add_argument(
        "--save-viz",
        action="store_true",
        help="Save per-burst attention heatmaps and convergence figure",
    )
    parser.add_argument(
        "--viz-dir",
        type=str,
        default=None,
        help="Directory for visualization outputs (default: {output}_viz/)",
    )

    args = parser.parse_args()

    if args.output is None:
        p = Path(args.input)
        args.output = str(p.parent / f"{p.stem}_focus_video.mp4")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    lo, hi = (float(x) for x in args.focus_range.split("-", 1))
    focus_range = (lo, hi)

    embed_model = args.embed_model
    inference_model = args.inference_model or embed_model
    attention_model = args.attention_model or inference_model

    from ..vlm.model_pool import load_vlm_pipeline

    inference_vlm = load_vlm_pipeline(
        inference_model,
        device=device,
        focus_alpha=args.focus_alpha,
        focus_range=focus_range,
        focus_suppress=args.focus_suppress,
    )

    encoder = None
    if embed_model != inference_model:
        embed_vlm = load_vlm_pipeline(embed_model, device=device)
        encoder = embed_vlm

    if attention_model != inference_model:
        attn_vlm = load_vlm_pipeline(attention_model, device=device)
        inference_vlm.attention_model_override = attn_vlm.model
        inference_vlm.attention_processor_override = attn_vlm.processor
        inference_vlm.attention_spatial_merge_override = attn_vlm.spatial_merge_size

    print(f"Embed model: {embed_model}")
    if inference_model != embed_model:
        print(f"Inference model: {inference_model}")
    if attention_model != embed_model:
        print(f"Attention model: {attention_model}")

    process_video_with_vlm(
        input_path=args.input,
        output_path=args.output,
        config=_make_config(args),
        vlm=inference_vlm,
        encoder=encoder,
        vlm_prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        do_sample=args.do_sample,
        show_preview=args.preview,
        compress=not args.no_compress,
        crf=args.crf,
        target_fps=None if args.no_fps_limit else args.target_fps,
        target_height=None if args.no_resize else args.target_height,
        save_log=not args.no_log,
        log_path=args.log,
        task_description=args.task,
        context_window=args.context_window,
        enable_context=not args.no_context,
        enable_spatial=not args.no_spatial,
        enable_attention_feedback=not args.no_attention_feedback,
        focus_alpha=args.focus_alpha,
        focus_range=focus_range,
        focus_suppress=args.focus_suppress,
        save_viz=args.save_viz,
        viz_dir=args.viz_dir,
    )
