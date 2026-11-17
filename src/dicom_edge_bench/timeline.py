"""Thread-safe monotonic event timeline used by optional profiling."""
from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import Any


@dataclass
class Timeline:
    """Collect ordered events using one monotonic host clock."""

    _events: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, event: str, **fields: Any) -> dict[str, Any]:
        row = {
            "timestamp_ns": time.perf_counter_ns(),
            "event": event,
            "thread": threading.current_thread().name,
            **fields,
        }
        with self._lock:
            self._events.append(row)
        return row

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events)

    def to_dict(self) -> dict[str, Any]:
        events = self.snapshot()
        return {
            "clock": "time.perf_counter_ns",
            "event_count": len(events),
            "events": events,
        }
