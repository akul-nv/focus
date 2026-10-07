"""One streaming captioning runner for FOCUS and the compared baseline adaptations."""

from __future__ import annotations

import argparse
import json
import platform
import time
from contextlib import closing
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import cv2
import numpy as np
import torch

from .config import FocusConfig
from .gating import PatchLevelGating
from .vlm.spatial import SpatialAttentionAnalyzer

METHODS = ("uniform", "bolt", "dycoke", "gating", "focus")


@dataclass
class RunConfig:
    method: str = "focus"
    target_fps: float = 24.0
    burst_size: int = 32
    interval: int = 100
    threshold: float = 0.4
    max_gate_frames: int = 300
    max_tokens: int = 100
    do_sample: bool = False
    prompt: str = "Describe what is happening in these video frames briefly."
    feedback_gamma: float = 0.5

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if not np.isfinite(self.target_fps) or self.target_fps <= 0:
            raise ValueError("target_fps must be finite and positive")
        if self.burst_size < 1 or self.interval < 1 or self.max_tokens < 1:
            raise ValueError("burst_size, interval and max_tokens must be positive")
        if self.method in ("uniform", "bolt", "dycoke") and self.interval < self.burst_size:
            raise ValueError("interval must be >= burst_size; overlapping bursts are not supported")
        if not np.isfinite(self.feedback_gamma) or self.feedback_gamma <= 0:
            raise ValueError("feedback_gamma must be positive")
        FocusConfig(
            threshold=self.threshold,
            max_gate_frames=self.max_gate_frames,
            burst_size=self.burst_size,
        )


def iter_video_frames(path: str | Path, target_fps: float):
    """Yield ``(source_frame_index, timestamp_seconds, BGR_frame)`` in time order.

    Fractional scheduling avoids drift for nonintegral source/target rate ratios.
    The source FPS must be valid; rates above it simply retain every source frame.
    """
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(source_fps) or source_fps <= 0:
            raise ValueError(f"Video has no valid frame rate: {path}")
        step = max(1.0, source_fps / target_fps)
        next_sample = 0.0
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index + 1e-9 >= next_sample:
                yield index, index / source_fps, frame
                next_sample += step
            index += 1
    finally:
        cap.release()


def run_stream(frames, vlm, config: RunConfig, *, encoder=None, collect_representatives=False):
    """Consume an indexed frame iterator synchronously with one fresh state per video.

    Fixed methods capture the first ``burst_size`` sampled frames every ``interval``
    samples. FOCUS/gating trigger bursts through patch dissimilarity. A final partial
    burst is always submitted. Feedback from a burst is installed before the next
    frame is processed. The same previous-caption prompt is used by every method.
    """
    adaptive = config.method in ("focus", "gating")
    if adaptive and encoder is None:
        raise ValueError("FOCUS and gating need an encoder")
    gater = (
        PatchLevelGating(
            FocusConfig(
                threshold=config.threshold,
                burst_size=config.burst_size,
                max_gate_frames=config.max_gate_frames,
            )
        )
        if adaptive
        else None
    )
    spatial = SpatialAttentionAnalyzer(fusion_gamma=config.feedback_gamma)
    responses, representatives, burst = [], [], []
    total_frames = captured_frames = sent_frames = 0
    pipeline_ms = 0.0
    attention_owner = getattr(vlm, "pipeline", vlm)
    old_extract = getattr(attention_owner, "extract_attention", False)
    attention_owner.extract_attention = config.method == "focus"
    attention_owner.last_attention_map = None
    started = time.perf_counter()

    def flush():
        nonlocal pipeline_ms, sent_frames
        prompt = config.prompt
        if responses:
            prompt = (
                f"Previously: {responses[-1]['text'][:80]}\n"
                f"{config.prompt}\nDescribe what changed. Do not repeat the previous description."
            )
        kwargs = {"max_new_tokens": config.max_tokens, "do_sample": config.do_sample}
        if config.method == "focus":
            kwargs["spatial_prior"] = spatial.get_focus_prior()
        attention_owner.last_attention_map = None
        text, ms = vlm.generate_from_frames([item[2] for item in burst], prompt=prompt, **kwargs)
        selection = dict(getattr(vlm, "last_stats", {}))
        kept = selection.get("selected_indices", list(range(len(burst))))
        sent_frames += len(kept)
        responses.append(
            {
                "text": text,
                "frame_indices": [item[0] for item in burst],
                "selected_frame_indices": [burst[i][0] for i in kept],
                "start_s": burst[0][1],
                "end_s": burst[-1][1],
                "pipeline_reported_time_ms": ms,
                "selection": selection,
            }
        )
        if collect_representatives:
            representatives.append(burst[kept[len(kept) // 2]][2].copy())
        pipeline_ms += ms
        if config.method == "focus" and vlm.last_attention_map is not None:
            spatial.set_attention_prior(vlm.last_attention_map)
        spatial.reset()
        burst.clear()

    try:
        for sample_index, item in enumerate(frames):
            total_frames += 1
            keep = sample_index % config.interval < config.burst_size
            if gater is not None:
                with torch.inference_mode():
                    encode = getattr(encoder, "encode_frame", encoder)
                    embedding = encode(item[2])
                keep, _, metadata = gater.process_frame(embedding)
                dmap = metadata.get("dissimilarity_map")
                if keep and config.method == "focus" and dmap is not None:
                    grid = getattr(encoder, "last_grid_hw", None)
                    if grid is None:
                        raise ValueError("FOCUS encoder must expose last_grid_hw")
                    spatial.update(dmap, *grid)
            if keep:
                captured_frames += 1
                burst.append(item)
                if len(burst) == config.burst_size:
                    flush()
        if burst:
            flush()
    finally:
        attention_owner.extract_attention = old_extract
    if total_frames == 0:
        raise ValueError("Video contains no decodable frames")
    return {
        "config": asdict(config),
        "stats": {
            "sampled_frames": total_frames,
            "captured_frames": captured_frames,
            "frames_sent_to_vlm": sent_frames,
            "caption_calls": len(responses),
            "capture_ratio": captured_frames / total_frames,
            "pipeline_reported_time_ms": pipeline_ms,
            "wall_time_s": time.perf_counter() - started,
        },
        "responses": responses,
    }, representatives


def run_video(path, vlm, config, *, encoder=None, collect_representatives=False):
    with closing(iter_video_frames(path, config.target_fps)) as frames:
        return run_stream(
            frames, vlm, config, encoder=encoder, collect_representatives=collect_representatives
        )


def add_run_arguments(parser, *, target_fps=24.0):
    parser.add_argument("--method", choices=METHODS, default="focus")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--encoder-model", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target-fps", type=float, default=target_fps)
    parser.add_argument("--burst-size", type=int, default=32)
    parser.add_argument(
        "--interval",
        type=int,
        default=100,
        help="Fixed-baseline burst interval in sampled frames (>= burst size)",
    )
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--max-gate-frames", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default=RunConfig.prompt)
    parser.add_argument("--focus-alpha", type=float, default=1.05)
    parser.add_argument("--focus-suppress", type=float, default=0.04)
    parser.add_argument(
        "--focus-range", type=float, nargs=2, default=(0.75, 1.0), metavar=("LOW", "HIGH")
    )
    parser.add_argument("--feedback-gamma", type=float, default=0.5)
    parser.add_argument("--bolt-frames", type=int, default=16)
    parser.add_argument("--bolt-power", type=float, default=3.0)
    parser.add_argument("--bolt-query", help="Optional CLIP query; otherwise the caption prompt")
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--dycoke-drop-fraction", type=float, default=0.3)
    parser.add_argument("--dycoke-keep-fraction", type=float, default=0.8)
    parser.add_argument("--dycoke-start-layer", type=int, default=3)
    parser.add_argument("--dycoke-similarity-threshold", type=float, default=0.9)
    parser.add_argument(
        "--dycoke-no-cache",
        action="store_true",
        help="Temporal-selection-only ablation; disables KV zeroing",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")


def config_from_args(args):
    if not 0 <= args.seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if not (0 <= args.focus_range[0] <= args.focus_range[1] <= 1):
        raise ValueError("focus-range must satisfy 0 <= LOW <= HIGH <= 1")
    if any(
        not np.isfinite(value) or value < 0 for value in (args.focus_alpha, args.focus_suppress)
    ):
        raise ValueError("Focus alpha/suppress must be finite and nonnegative")
    if args.bolt_frames < 1 or (
        args.bolt_power != -1 and (not np.isfinite(args.bolt_power) or args.bolt_power <= 0)
    ):
        raise ValueError("bolt-frames must be positive; bolt-power must be -1 or positive")
    if not (0 <= args.dycoke_drop_fraction < 1 and 0 <= args.dycoke_keep_fraction <= 1):
        raise ValueError("DyCoke drop fraction must be in [0,1), keep fraction in [0,1]")
    if args.dycoke_start_layer < 0 or not -1 <= args.dycoke_similarity_threshold <= 1:
        raise ValueError("Invalid DyCoke start layer or similarity threshold")
    return RunConfig(**{name: getattr(args, name) for name in RunConfig.__dataclass_fields__})


def load_pipelines(args):
    """Return (caption pipeline, encoder, plain generator for final QA)."""
    from .vlm.model_pool import load_vlm_pipeline

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    options = {}
    if args.method == "focus":
        options = dict(
            focus_alpha=args.focus_alpha,
            focus_suppress=args.focus_suppress,
            focus_range=tuple(args.focus_range),
        )
    base = load_vlm_pipeline(args.model, device=args.device, **options)
    encoder = None
    if args.method in ("focus", "gating"):
        encoder = load_vlm_pipeline(args.encoder_model, device=args.device)
    pipeline = base
    if args.method == "bolt":
        from .baselines import BoltPipeline

        pipeline = BoltPipeline(
            base,
            num_frames=args.bolt_frames,
            power=args.bolt_power,
            clip_model=args.clip_model,
            query=args.bolt_query,
        )
    elif args.method == "dycoke":
        from .baselines import DyCokePipeline

        pipeline = DyCokePipeline(
            base,
            drop_fraction=args.dycoke_drop_fraction,
            keep_fraction=args.dycoke_keep_fraction,
            start_layer=args.dycoke_start_layer,
            similarity_threshold=args.dycoke_similarity_threshold,
            cache_pruning=not args.dycoke_no_cache,
        )
    return pipeline, encoder, base


def write_json(path, data):
    """Atomically replace a checkpoint; no automatic resume across different configs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def runtime_metadata(pipeline):
    """Record installed runtime versions and actual model dtype when available."""
    packages = {}
    for name in ("focus-vlm", "torch", "transformers", "numpy", "opencv-python"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "packages": packages,
        "model_dtype": str(getattr(getattr(pipeline, "model", None), "dtype", "unavailable")),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", type=Path, help="One or more local video files")
    add_run_arguments(parser)
    args = parser.parse_args(argv)
    try:
        config = config_from_args(args)
        if args.output.exists() and not args.overwrite:
            raise ValueError("Output already exists; choose a new path or pass --overwrite")
        for path in args.videos:
            if not path.is_file():
                raise ValueError(f"Video does not exist: {path}")
            if path.resolve() == args.output.resolve():
                raise ValueError("Output must not overwrite an input video")
    except ValueError as exc:
        parser.error(str(exc))
    pipeline, encoder, base = load_pipelines(args)
    output = {
        "schema_version": 1,
        "arguments": vars(args),
        "results": [],
        "runtime": runtime_metadata(base),
        "videos_requested": len(args.videos),
        "complete": False,
    }
    for path in args.videos:
        result, _ = run_video(path, pipeline, config, encoder=encoder)
        result["video"] = str(path)
        output["results"].append(result)
        output["complete"] = len(output["results"]) == len(args.videos)
        write_json(args.output, output)
        print(f"{path.name}: {result['stats']['caption_calls']} caption calls -> {args.output}")


if __name__ == "__main__":
    main()
