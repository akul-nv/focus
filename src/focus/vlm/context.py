# SPDX-License-Identifier: Apache-2.0
"""Rolling-history context manager for VLM inference.

:class:`VLMContextManager` maintains a simple rolling window of previous
VLM responses so that each new prompt includes a narrative of what
happened so far.  This lets the model reason about temporal progression
and focus on what *changed* rather than repeating earlier observations.

No domain-specific keywords, stage definitions, or entity lists are
required -- the context is built purely from the model's own past output.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VLMContextManager:
    """Rolling-history context across VLM inference bursts.

    Stores each VLM response and injects the most recent
    ``context_window`` responses into subsequent prompts as a
    "story so far".

    Attributes:
        task_description: Optional free-text task description prepended
            to every prompt (e.g. ``"Robot picking up a red block"``).
        context_window: How many previous responses to include.
        base_prompt: Fallback prompt when context is disabled.
        enabled: Master switch for context tracking.
        max_summary_length: Per-response character cap stored in history.
        initial_prompt: Prompt text for the very first burst.
        continuation_prompt: Prompt text for subsequent bursts.
    """

    task_description: str = ""
    context_window: int = 5
    base_prompt: str = "Describe what is happening in these video frames briefly."
    enabled: bool = True
    max_summary_length: int = 80

    initial_prompt: str = "Describe the scene in these frames briefly."
    continuation_prompt: str = (
        "Look at the frames and describe what you see NOW. "
        "What is different from before? "
        "Do NOT repeat the previous description."
    )

    _history: list[tuple[int, str]] = field(default_factory=list)

    def build_prompt(
        self,
        burst_id: int,
        frame_start: int,
        frame_end: int,
        fps: float = 24.0,
    ) -> str:
        """Build a context-aware prompt for the current burst.

        Args:
            burst_id: Sequential burst number.
            frame_start: First frame index in the burst.
            frame_end: Last frame index in the burst.
            fps: Video frame rate (for timestamp display).

        Returns:
            A multi-section prompt string.
        """
        if not self.enabled:
            return self.base_prompt

        parts: list[str] = [self.base_prompt] if self.base_prompt else []

        if self.task_description:
            parts.append(f"TASK: {self.task_description}")

        start_time = frame_start / fps if fps > 0 else 0
        end_time = frame_end / fps if fps > 0 else 0

        recent = self._history[-self.context_window :]
        if recent:
            _, latest = recent[-1]
            parts.append(f"Previously: {latest}")
            if len(recent) > 1:
                parts.append(f"({len(recent)} of {len(self._history)} observations)")

        parts.append(f"NOW ({start_time:.1f}s - {end_time:.1f}s):")

        if not self._history:
            parts.append(self.initial_prompt)
        else:
            parts.append(self.continuation_prompt)

        return "\n".join(parts)

    def update_from_response(
        self,
        response_text: str,
        burst_id: int,
        frame_start: int,
        frame_end: int,
    ) -> None:
        """Store a VLM response in the rolling history.

        Args:
            response_text: The VLM's output for this burst.
            burst_id: Burst number.
            frame_start: First frame index (retained for compatibility).
            frame_end: Last frame index (retained for compatibility).
        """
        if not self.enabled:
            return

        summary = response_text.strip()
        if len(summary) > self.max_summary_length:
            summary = summary[: self.max_summary_length] + "..."
        self._history.append((burst_id, summary))

    def get_summary(self) -> dict:
        """Snapshot of context state for logging / serialisation."""
        return {
            "enabled": self.enabled,
            "task_description": self.task_description,
            "context_window": self.context_window,
            "total_responses": len(self._history),
            "history": [{"burst_id": bid, "summary": text} for bid, text in self._history],
        }

    def get_narrative_for_log(self) -> str:
        """Human-readable narrative for a log file."""
        if not self.enabled or not self._history:
            return "Context tracking was disabled or no observations recorded."

        lines: list[str] = []
        if self.task_description:
            lines += [f"Task: {self.task_description}", ""]
        lines.append("Sequence of Events:")
        for bid, text in self._history:
            lines.append(f"  Burst {bid}: {text}")
        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all accumulated history (keeps configuration)."""
        self._history = []
