"""Optional hardware and process telemetry collectors.

Unavailable counters are represented as ``None`` with a capability note;
collectors never manufacture zero-valued measurements.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
import platform
import subprocess
from typing import Any


@dataclass
class TelemetryCollector:
    enabled: bool = False
    capabilities: dict[str, bool] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.capabilities = {
            "process_rss": False,
            "cpu_percent": False,
            "nvidia_smi": False,
            "cuda": False,
            "tegrastats": False,
        }
        if not self.enabled:
            return
        try:
            import psutil  # noqa: F401
            self.capabilities["process_rss"] = True
            self.capabilities["cpu_percent"] = True
        except ImportError:
            self.notes["process"] = "psutil not installed"
        self.capabilities["nvidia_smi"] = _command_available("nvidia-smi")
        self.capabilities["tegrastats"] = _command_available("tegrastats")
        try:
            import torch
            self.capabilities["cuda"] = bool(torch.cuda.is_available())
        except ImportError:
            self.notes["cuda"] = "torch not installed"

    def snapshot(self, label: str) -> dict[str, Any]:
        result: dict[str, Any] = {
            "label": label,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "process_rss_bytes": None,
            "cpu_percent": None,
            "gpu": None,
            "tegrastats": None,
        }
        if not self.enabled:
            result["not_measured"] = "telemetry disabled"
            return result
        try:
            import psutil
            process = psutil.Process(os.getpid())
            result["process_rss_bytes"] = process.memory_info().rss
            result["cpu_percent"] = process.cpu_percent(interval=None)
        except (ImportError, OSError):
            result["not_measured"] = "psutil unavailable"
        if self.capabilities.get("nvidia_smi"):
            result["gpu"] = _run_command(
                ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw", "--format=csv,noheader,nounits"]
            )
        if self.capabilities.get("tegrastats"):
            result["tegrastats"] = _run_command(["tegrastats", "--interval", "100", "--count", "1"])
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "capabilities": self.capabilities,
            "notes": self.notes,
        }


def _command_available(command: str) -> bool:
    try:
        subprocess.run([command, "--help"], capture_output=True, timeout=2, check=False)
        return True
    except (OSError, subprocess.TimeoutExpired):
        return False


def _run_command(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None
