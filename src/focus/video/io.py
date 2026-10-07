# SPDX-License-Identifier: Apache-2.0
"""Video I/O helpers (FFmpeg compression, etc.)."""

from __future__ import annotations


def compress_video(input_path: str, output_path: str, crf: int = 23) -> bool:
    """Re-encode a video with H.264 via FFmpeg.

    Args:
        input_path: Path to the uncompressed source video.
        output_path: Destination path for the compressed file.
        crf: Constant Rate Factor (0-51).  Lower = better quality;
            18-23 is visually lossless.

    Returns:
        *True* on success, *False* if FFmpeg is missing or fails.
    """
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        print("  Warning: ffmpeg not found, skipping compression")
        return False

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                input_path,
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                str(crf),
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                "-loglevel",
                "error",
                output_path,
            ],
            check=True,
        )
        return True
    except subprocess.CalledProcessError as exc:
        print(f"  Warning: ffmpeg compression failed: {exc}")
        return False
