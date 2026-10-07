#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate a synthetic test video with varying motion patterns.

Useful for testing FOCUS without requiring a real video file::

    python scripts/generate_test_video.py
    python scripts/generate_test_video.py --duration 30 --output test.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def generate_test_video(
    output_path: str,
    duration_seconds: float = 15.0,
    fps: int = 30,
    resolution: tuple[int, int] = (640, 480),
) -> str:
    """Generate a synthetic video with five motion segments.

    Segments: STATIC -> MOTION -> FLASH -> MULTI -> GRADUAL

    Args:
        output_path: Destination file path.
        duration_seconds: Total duration.
        fps: Frame rate.
        resolution: ``(width, height)``.

    Returns:
        The *output_path*.
    """
    width, height = resolution
    if duration_seconds <= 0 or fps <= 0 or width < 2 or height < 2:
        raise ValueError("Duration, FPS, and image dimensions must be positive")
    if width % 2 or height % 2:
        raise ValueError("Video width and height must be even")
    total = int(duration_seconds * fps)
    if total < 1:
        raise ValueError("Duration must include at least one frame")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, resolution)
    if not out.isOpened():
        raise ValueError(f"Could not create video: {output_path}")

    print(f"Generating test video: {output_path}")
    print(f"  Duration: {duration_seconds}s, FPS: {fps}, Frames: {total}")

    for i in range(total):
        t = i / fps
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (60, 40, 40)

        if t < 3.0:
            cv2.rectangle(frame, (100, 100), (200, 200), (0, 100, 200), -1)
            seg = "STATIC"
        elif t < 7.0:
            lt = t - 3.0
            cx = int(100 + lt * 120)
            cy = int(height // 2 + 60 * np.sin(lt * 2.5))
            cv2.circle(frame, (cx, cy), 45, (0, 255, 0), -1)
            seg = "MOTION"
        elif t < 8.0:
            frame[:] = (220, 220, 220)
            cv2.putText(
                frame,
                "SCENE CHANGE",
                (width // 2 - 100, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (50, 50, 50),
                2,
            )
            seg = "FLASH"
        elif t < 11.0:
            lt = t - 8.0
            for j in range(3):
                cx = int((width // 4) * (j + 1))
                cy = int(height // 2 + 80 * np.sin(lt * 4 + j * 1.5))
                colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]
                cv2.circle(frame, (cx, cy), 30, colors[j], -1)
            seg = "MULTI"
        else:
            lt = t - 11.0
            shift = int(40 * lt)
            frame[:, :, 0] = min(255, 80 + shift)
            frame[:, :, 1] = 60
            frame[:, :, 2] = 100
            cv2.rectangle(frame, (200, 150), (450, 350), (150, 180, 200), -1)
            seg = "GRADUAL"

        cv2.putText(
            frame,
            f"Frame: {i}",
            (10, height - 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (150, 150, 150),
            1,
        )
        cv2.putText(
            frame,
            f"Segment: {seg}",
            (10, height - 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (150, 150, 150),
            1,
        )
        out.write(frame)

    out.release()
    print(f"  Generated {total} frames")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic test video for FOCUS")
    parser.add_argument("-o", "--output", default="test_video.mp4", help="Output path")
    parser.add_argument("-d", "--duration", type=float, default=15.0, help="Duration in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate")
    parser.add_argument("--width", type=int, default=640, help="Width")
    parser.add_argument("--height", type=int, default=480, help="Height")
    args = parser.parse_args()

    generate_test_video(
        output_path=args.output,
        duration_seconds=args.duration,
        fps=args.fps,
        resolution=(args.width, args.height),
    )


if __name__ == "__main__":
    main()
