# SPDX-License-Identifier: Apache-2.0
"""Real-time performance monitoring for video processing pipelines."""

from __future__ import annotations

import time
from collections import deque

import psutil
import torch


class PerformanceMonitor:
    """Collects latency, memory, and GPU utilisation metrics.

    Intended to be used around a frame-processing loop::

        mon = PerformanceMonitor(device="cuda")
        for frame in video:
            mon.start_frame()
            mon.start_encode()
            embedding = encoder(frame)
            mon.end_encode()
            # ... gating / writing ...
            mon.end_frame()
        summary = mon.get_summary()

    Attributes:
        window_size: Number of recent samples kept for rolling statistics.
    """

    def __init__(self, window_size: int = 30, device: str = "cuda") -> None:
        self.window_size = window_size
        self.device = device
        self.process = psutil.Process()

        self.frame_times: deque[float] = deque(maxlen=window_size)
        self.encode_times: deque[float] = deque(maxlen=window_size)
        self.total_times: deque[float] = deque(maxlen=window_size)

        self._frame_start: float | None = None
        self._encode_start: float | None = None
        self._total_start: float | None = None

        self.gpu_available = torch.cuda.is_available() and "cuda" in device
        self.pynvml_available = False
        self._nvml_handle = None

        if self.gpu_available:
            try:
                import pynvml

                pynvml.nvmlInit()
                idx = int(device.split(":")[1]) if ":" in device else 0
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                self.pynvml_available = True
            except Exception:
                pass

        self.total_frames: int = 0
        self.total_encode_time: float = 0
        self.total_frame_time: float = 0

    def start_frame(self) -> None:
        """Call at the start of processing a frame."""
        self._total_start = time.perf_counter()

    def start_encode(self) -> None:
        """Call before encoder inference."""
        self._encode_start = time.perf_counter()

    def end_encode(self) -> None:
        """Call after encoder inference."""
        if self._encode_start is not None:
            ms = (time.perf_counter() - self._encode_start) * 1000
            self.encode_times.append(ms)
            self.total_encode_time += ms

    def end_frame(self) -> None:
        """Call at the end of processing a frame."""
        if self._total_start is not None:
            ms = (time.perf_counter() - self._total_start) * 1000
            self.total_times.append(ms)
            self.total_frame_time += ms
            self.total_frames += 1

    def get_memory_stats(self) -> dict:
        """Current RAM usage."""
        mem = self.process.memory_info()
        sys_mem = psutil.virtual_memory()
        return {
            "ram_used_gb": mem.rss / (1024**3),
            "ram_percent": self.process.memory_percent(),
            "ram_total_gb": sys_mem.total / (1024**3),
            "ram_available_gb": sys_mem.available / (1024**3),
        }

    def get_gpu_stats(self) -> dict:
        """Current GPU memory and utilisation (empty dict if unavailable)."""
        if not self.gpu_available:
            return {}
        stats: dict = {
            "gpu_mem_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
            "gpu_mem_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
        }
        if self.pynvml_available and self._nvml_handle:
            try:
                import pynvml

                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                stats.update(
                    {
                        "gpu_util_percent": util.gpu,
                        "gpu_mem_util_percent": util.memory,
                        "gpu_mem_total_gb": mem_info.total / (1024**3),
                        "gpu_mem_used_gb": mem_info.used / (1024**3),
                        "gpu_mem_free_gb": mem_info.free / (1024**3),
                    }
                )
            except Exception:
                pass
        return stats

    def get_latency_stats(self) -> dict:
        """Rolling-window latency and FPS."""
        stats: dict = {}
        if self.total_times:
            times = list(self.total_times)
            avg = sum(times) / len(times)
            stats.update(
                {
                    "latency_ms": times[-1],
                    "latency_avg_ms": avg,
                    "latency_min_ms": min(times),
                    "latency_max_ms": max(times),
                    "fps_current": 1000 / times[-1] if times[-1] > 0 else 0,
                    "fps_avg": 1000 / avg if avg > 0 else 0,
                }
            )
        if self.encode_times:
            times = list(self.encode_times)
            stats.update(
                {
                    "encode_ms": times[-1],
                    "encode_avg_ms": sum(times) / len(times),
                }
            )
        return stats

    def get_all_stats(self) -> dict:
        """Merge memory, GPU, and latency stats into a single dict."""
        stats: dict = {}
        stats.update(self.get_memory_stats())
        stats.update(self.get_gpu_stats())
        stats.update(self.get_latency_stats())
        return stats

    def format_stats_line(self) -> str:
        """One-line summary suitable for a progress-bar postfix."""
        s = self.get_all_stats()
        parts: list[str] = []
        if "fps_current" in s:
            parts.append(f"FPS:{s['fps_current']:.1f}")
        if "latency_ms" in s:
            parts.append(f"Lat:{s['latency_ms']:.0f}ms")
        if "encode_ms" in s:
            parts.append(f"Enc:{s['encode_ms']:.0f}ms")
        parts.append(f"RAM:{s.get('ram_used_gb', 0):.1f}GB")
        if self.gpu_available:
            if "gpu_util_percent" in s:
                parts.append(f"GPU:{s['gpu_util_percent']}%")
            if "gpu_mem_used_gb" in s:
                parts.append(f"VRAM:{s['gpu_mem_used_gb']:.1f}GB")
            elif "gpu_mem_allocated_gb" in s:
                parts.append(f"VRAM:{s['gpu_mem_allocated_gb']:.1f}GB")
        return " | ".join(parts)

    def get_summary(self) -> dict:
        """Final report after processing is complete."""
        summary = {
            "total_frames_processed": self.total_frames,
            "total_time_s": self.total_frame_time / 1000 if self.total_frame_time else 0,
            "avg_latency_ms": self.total_frame_time / self.total_frames if self.total_frames else 0,
            "avg_encode_ms": self.total_encode_time / self.total_frames if self.total_frames else 0,
            "avg_fps": (
                self.total_frames / (self.total_frame_time / 1000) if self.total_frame_time else 0
            ),
        }
        summary.update(self.get_memory_stats())
        summary.update(self.get_gpu_stats())
        return summary
