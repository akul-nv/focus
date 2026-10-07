#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Example: FOCUS + VLM pipeline with value-cache focus calibration.

Usage::

    python -m pip install -e '.[qwen]'
    python examples/vlm_integration.py video.mp4 \\
        --task "Robot picks up apple from bowl and places it in box"
"""

from focus import FocusConfig
from focus.video.processing import process_video_with_vlm
from focus.vlm import load_vlm_pipeline


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Input video")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--task", default="", help="Task description")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--burst-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--focus-alpha", type=float, default=1.05)
    args = parser.parse_args()

    from pathlib import Path

    output = args.output or str(Path(args.video).with_stem(Path(args.video).stem + "_focus"))

    vlm = load_vlm_pipeline(
        args.model,
        device=args.device,
        focus_alpha=args.focus_alpha,
        focus_range=(0.75, 1.0),
        focus_suppress=0.04,
    )

    config = FocusConfig(
        mode="patch",
        threshold=args.threshold,
        burst_enabled=True,
        burst_size=args.burst_size,
    )

    results = process_video_with_vlm(
        input_path=args.video,
        output_path=output,
        config=config,
        vlm=vlm,
        task_description=args.task,
        enable_context=True,
        focus_alpha=args.focus_alpha,
        focus_range=(0.75, 1.0),
        focus_suppress=0.04,
    )

    print(f"\nDone. Output: {output}")
    if results.get("log_path"):
        print(f"Log: {results['log_path']}")


if __name__ == "__main__":
    main()
