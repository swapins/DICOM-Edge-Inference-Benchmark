"""Low-overhead phase and queue profiling for EXP-002.

Profiling is opt-in. The default EXP-001 runner path does not create a
timeline or CUDA events, preserving its timing behavior.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import time
from typing import Any, Iterator

from .timeline import Timeline


@dataclass
class ProfileTrace:
    """One trial's host timeline and phase measurements."""

    enabled: bool = True
    detailed_torch: bool = False
    timeline: Timeline = field(default_factory=Timeline)
    phase_ms: dict[str, list[float]] = field(default_factory=dict)
    cuda_event_ms: dict[str, list[float]] = field(default_factory=dict)
    queue: dict[str, Any] = field(default_factory=dict)
    telemetry: dict[str, Any] = field(default_factory=dict)

    @contextmanager
    def phase(self, name: str, **fields: Any) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        start = time.perf_counter_ns()
        self.timeline.record(f"{name}_start", **fields)
        try:
            yield
        finally:
            end = time.perf_counter_ns()
            self.phase_ms.setdefault(name, []).append((end - start) / 1_000_000)
            self.timeline.record(
                f"{name}_finish", elapsed_ns=end - start, **fields
            )

    def add_cuda_event(self, name: str, milliseconds: float) -> None:
        if self.enabled:
            self.cuda_event_ms.setdefault(name, []).append(milliseconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "detailed_torch": self.detailed_torch,
            "phase_ms": self.phase_ms,
            "cuda_event_ms": self.cuda_event_ms,
            "queue": self.queue,
            "telemetry": self.telemetry,
            "timeline": self.timeline.to_dict(),
        }
