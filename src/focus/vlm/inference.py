# SPDX-License-Identifier: Apache-2.0
"""Asynchronous VLM inference queue and supporting data structures.

:class:`VLMInferenceQueue` runs VLM generation in a background thread
so frame processing can continue without blocking on inference.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .base import BaseVLMPipeline
from .context import VLMContextManager


@dataclass
class VLMResponse:
    """Container for a single VLM inference result."""

    text: str
    burst_id: int
    frame_start: int
    frame_end: int
    inference_time_ms: float
    timestamp: float
    attention_map: Optional[torch.Tensor] = None

    def to_dict(self) -> dict:
        return {
            "burst_id": self.burst_id,
            "text": self.text,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "inference_time_ms": self.inference_time_ms,
            "timestamp": self.timestamp,
        }


@dataclass
class BurstState:
    """Tracks burst collection and VLM response lifecycle.

    A burst goes through: **collecting** -> **pending** (submitted to
    VLM queue) -> **responded** (result available).
    """

    collecting_burst_id: int = 0
    collecting_start_frame: int = -1
    collecting_end_frame: int = -1

    pending_burst_id: int = -1
    pending_start_frame: int = -1
    pending_end_frame: int = -1

    last_response: Optional[VLMResponse] = None

    def start_burst(self, burst_id: int, frame_idx: int) -> None:
        self.collecting_burst_id = burst_id
        self.collecting_start_frame = frame_idx
        self.collecting_end_frame = frame_idx

    def add_frame(self, frame_idx: int) -> None:
        self.collecting_end_frame = frame_idx

    def submit_burst(self) -> None:
        self.pending_burst_id = self.collecting_burst_id
        self.pending_start_frame = self.collecting_start_frame
        self.pending_end_frame = self.collecting_end_frame
        self.collecting_burst_id += 1
        self.collecting_start_frame = -1
        self.collecting_end_frame = -1

    def receive_response(self, response: VLMResponse) -> None:
        self.last_response = response
        if response.burst_id == self.pending_burst_id:
            self.pending_burst_id = -1

    def is_collecting(self) -> bool:
        return self.collecting_start_frame >= 0

    def is_pending(self) -> bool:
        return self.pending_burst_id >= 0

    def current_frame_in_response_range(self, frame_idx: int) -> bool:
        if self.last_response is None:
            return False
        return self.last_response.frame_start <= frame_idx <= self.last_response.frame_end


class VLMInferenceQueue:
    """Background-threaded VLM inference with bounded queue.

    Args:
        vlm: A :class:`BaseVLMPipeline` implementation.
        max_queue_size: Maximum queued burst requests before dropping.
        prompt: Default generation prompt.
        max_new_tokens: Token budget per generation call.
        context_manager: Optional temporal context tracker.
        video_fps: Video frame rate (used for timestamp display in prompts).
        enable_attention_feedback: When *True*, request the VLM to extract
            attention maps after each generation for the spatial feedback
            loop.
        synchronous: Run generation in the calling thread. This guarantees
            each response is available before the next burst is collected and
            never drops requests. The video demo uses this mode by default.
    """

    def __init__(
        self,
        vlm: BaseVLMPipeline,
        max_queue_size: int = 2,
        prompt: str = "Describe what is happening in these video frames briefly.",
        max_new_tokens: int = 100,
        context_manager: Optional[VLMContextManager] = None,
        video_fps: float = 24.0,
        enable_attention_feedback: bool = False,
        do_sample: bool = False,
        synchronous: bool = False,
    ) -> None:
        if max_queue_size < 1:
            raise ValueError("max_queue_size must be >= 1")
        self.synchronous = synchronous
        self._errors: list[Exception] = []
        self.vlm = vlm
        self.max_queue_size = max_queue_size
        self.base_prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.context_manager = context_manager
        self.video_fps = video_fps
        self.enable_attention_feedback = enable_attention_feedback
        self.do_sample = do_sample

        self.request_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._latest_response: Optional[VLMResponse] = None
        self._response_lock = threading.Lock()
        self._new_response_event = threading.Event()
        self._all_responses: list[VLMResponse] = []

        self.total_inferences: int = 0
        self.dropped_requests: int = 0
        self.total_inference_time: float = 0

        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Spawn the background inference thread."""
        if self._running:
            return
        self._running = True
        if self.synchronous:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the thread to exit and wait for it."""
        if self._thread is not None:
            self.request_queue.join()
        self._running = False
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._errors:
            raise RuntimeError("VLM inference failed") from self._errors[0]

    def submit_burst(
        self,
        burst_id: int,
        frames: list[np.ndarray],
        frame_start: int,
        frame_end: int,
        spatial_prior: torch.Tensor | None = None,
    ) -> bool:
        """Enqueue a burst for inference.

        Args:
            burst_id: Sequential burst number.
            frames: List of BGR frames.
            frame_start: First frame index in the burst.
            frame_end: Last frame index in the burst.
            spatial_prior: Optional normalised ``(grid_h, grid_w)`` heatmap
                for value-cache calibration during generation.

        Returns:
            *True* if accepted, *False* if the queue was full (dropped).
        """
        if not self._running:
            raise RuntimeError("Call start() before submitting bursts")
        if not frames:
            raise ValueError("A burst must contain at least one frame")
        if self.synchronous:
            self._infer(burst_id, frames, frame_start, frame_end, spatial_prior)
            return True
        try:
            self.request_queue.put_nowait((burst_id, frames, frame_start, frame_end, spatial_prior))
            return True
        except queue.Full:
            self.dropped_requests += 1
            return False

    def get_latest_response(self) -> Optional[VLMResponse]:
        """Thread-safe access to the most recent response."""
        with self._response_lock:
            return self._latest_response

    def check_new_response(self) -> Optional[VLMResponse]:
        """Return the latest response if one arrived since the last check."""
        with self._response_lock:
            if self._new_response_event.is_set():
                self._new_response_event.clear()
                return self._latest_response
            return None

    def get_all_responses(self) -> list[VLMResponse]:
        """Thread-safe copy of all responses received so far."""
        with self._response_lock:
            return list(self._all_responses)

    def get_stats(self) -> dict:
        return {
            "total_inferences": self.total_inferences,
            "dropped_requests": self.dropped_requests,
            "avg_inference_time_ms": (
                self.total_inference_time / self.total_inferences
                if self.total_inferences > 0
                else 0
            ),
            "queue_size": self.request_queue.qsize(),
        }

    def _build_prompt(
        self,
        burst_id: int,
        frame_start: int,
        frame_end: int,
    ) -> str:
        if self.context_manager and self.context_manager.enabled:
            return self.context_manager.build_prompt(
                burst_id=burst_id,
                frame_start=frame_start,
                frame_end=frame_end,
                fps=self.video_fps,
            )
        return self.base_prompt

    def _infer(self, burst_id, frames, frame_start, frame_end, spatial_prior):
        prompt = self._build_prompt(burst_id, frame_start, frame_end)
        self.vlm.extract_attention = self.enable_attention_feedback
        self.vlm.last_attention_map = None
        text, ms = self.vlm.generate_from_frames(
            frames,
            prompt=prompt,
            max_new_tokens=self.max_new_tokens,
            spatial_prior=spatial_prior,
            do_sample=self.do_sample,
        )
        attention_map = self.vlm.last_attention_map
        if self.context_manager:
            self.context_manager.update_from_response(text, burst_id, frame_start, frame_end)
        response = VLMResponse(
            text=text,
            burst_id=burst_id,
            frame_start=frame_start,
            frame_end=frame_end,
            inference_time_ms=ms,
            timestamp=time.time(),
            attention_map=attention_map,
        )
        with self._response_lock:
            self._latest_response = response
            self._all_responses.append(response)
            self._new_response_event.set()
        self.total_inferences += 1
        self.total_inference_time += ms

    def _loop(self) -> None:
        while self._running:
            try:
                request = self.request_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._infer(*request)
            except Exception as exc:
                self._errors.append(exc)
            finally:
                self.request_queue.task_done()
