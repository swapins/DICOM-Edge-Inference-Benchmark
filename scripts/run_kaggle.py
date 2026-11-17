#!/usr/bin/env python3
"""Kaggle single-GPU benchmark orchestration (synthetic data only).

All GPU inference occurs on CUDA device 0; T4 x2 is NOT used as a multi-GPU
benchmark. Kaggle is a cloud reference platform, not a Jetson/edge device.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dicom_edge_bench.runner import benchmark, benchmark_prefetch_sweep, save_result  # noqa: E402
from dicom_edge_bench.synthetic import create_dicom  # noqa: E402

HIGH_DICOM_COMMIT = "d42f664006a9cd2194eda0bf5b1cf0fa2fb0188e"


def experiment_plan(profile: str) -> dict:
    """Return a deterministic workload plan; no hardware required."""
    if profile == "pilot":
        return {
            "encodings": ["native", "jpeg"],
            "selections": ["uniform:16"],
            "devices": ["cpu", "cuda"],
            "warmups": 1,
            "repeats": 2,
        }
    if profile == "full":
        return {
            "encodings": ["native", "jpeg"],
            "selections": ["first:1", "uniform:4", "uniform:16", "uniform:64", "all"],
            "devices": ["cpu", "cuda"],
            "warmups": 3,
            "repeats": 10,
        }
    if profile == "validation2":
        return {
            "encodings": ["native", "jpeg"],
            "selections": ["uniform:32"],
            "devices": ["cpu", "cuda"],
            "batch_sizes": [4, 8, 16, 32],
            "warmups": 3,
            "repeats": 10,
            "mode_order_policy": "randomized per measured repeat by runner seed 347 + repeat index",
        }
    if profile == "exp002":
        return {
            "encodings": ["native", "jpeg"],
            "selections": ["uniform:32"],
            "devices": ["cpu", "cuda"],
            "batch_sizes": [8, 16, 32],
            "baseline_prefetch": 2,
            "prefetch_depths": [1, 2, 4],
            "warmups": 3,
            "repeats": 10,
            "shared_configurations": 12,
            "method_series": 60,
            "mode_order_policy": "randomized per measured repeat by runner seed 347 + repeat index",
            "baseline_policy": "A and B run once per shared configuration; C runs at each prefetch depth",
        }
    if profile == "exp002_smoke":
        return {
            "encodings": ["native"],
            "selections": ["uniform:32"],
            "devices": ["cuda"],
            "batch_sizes": [16],
            "baseline_prefetch": 2,
            "prefetch_depths": [1, 2, 4],
            "warmups": 1,
            "repeats": 1,
            "shared_configurations": 1,
            "method_series": 5,
            "mode_order_policy": "randomized per measured repeat by runner seed 347 + repeat index",
            "baseline_policy": "A and B run once per shared configuration; C runs at each prefetch depth",
        }
    raise ValueError(f"Unknown profile: {profile}")


def _gpu_details(torch):
    details = []
    for i in range(torch.cuda.device_count()):
        details.append({
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "memory_bytes": torch.cuda.get_device_properties(i).total_memory,
            "capability": list(torch.cuda.get_device_capability(i)),
        })
    return details


def _nvidia_smi() -> str | None:
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"],
            check=False, capture_output=True, text=True, timeout=10,
        )
        return p.stdout.strip() if p.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run(*, profile: str, output: Path, rows: int = 128, columns: int = 128,
        frames: int = 96, batch_size: int = 16, prefetch: int = 2,
        max_experiments: int = 64, max_runtime_seconds: int = 900) -> dict:
    # Intentional hard failure: never silently fall back to CPU.
    import torch
    import highdicom
    from highdicom.io import ImageFileReader

    if not torch.cuda.is_available():
        raise RuntimeError(
            "A working Kaggle CUDA accelerator is required. In Kaggle Settings, "
            "select GPU T4 x2, restart the session, and confirm torch.cuda.is_available()."
        )
    if not callable(getattr(ImageFileReader, "read_frames", None)):
        raise RuntimeError("Installed highdicom lacks issue #347 read_frames()")
    if frames < 96:
        raise ValueError("Use at least 96 frames to preserve the published workload")
    if batch_size < 1 or prefetch < 1:
        raise ValueError("batch_size and prefetch must be >= 1")

    plan = experiment_plan(profile)
    series_count = plan.get("method_series", len(plan["encodings"]) * len(plan["devices"]) * len(plan["selections"]) * len(plan.get("batch_sizes", [batch_size])) * 3)
    if series_count > max_experiments:
        raise ValueError(f"Experiment series count {series_count} exceeds max_experiments={max_experiments}")
    output.mkdir(parents=True, exist_ok=True)
    inputs_dir = output / "data"
    results_dir = output / "trials"
    inputs_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    machine = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": "Kaggle cloud GPU reference platform (not edge hardware)",
        "single_gpu_experiment": True,
        "cuda_device_used": 0,
        "available_gpus": _gpu_details(torch),
        "gpu_query": _nvidia_smi(),
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": str(torch.version.cuda),
        "highdicom_version": str(highdicom.__version__),
        "highdicom_expected_commit": HIGH_DICOM_COMMIT,
        "highdicom_commit_verification": "Install from pinned SHA in notebook; verify pip direct_url.json separately",
        "pydicom_version": _package_version("pydicom"),
        "pillow_version": _package_version("pillow"),
        "psutil_version": _package_version("psutil"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch_num_threads": 1,
        "plan": plan,
        "dataset": {"synthetic_only": True, "frames": frames, "rows": rows, "columns": columns},
        "batch_size": batch_size,
        "prefetch_queue_capacity_frames": prefetch,
        "note": (
            "GPU is used for PyTorch inference only; DICOM decoding is CPU. "
            "Second T4, if present, is not used. CPU and GPU timings on the same "
            "Kaggle machine must not be called Jetson results. No clinical claims."
        ),
    }
    (output / "machine.json").write_text(json.dumps(machine, indent=2) + "\n")

    summary_rows = []
    plan_batch_sizes = plan.get("batch_sizes", [batch_size])
    started = time.monotonic()
    if profile in ("exp002", "exp002_smoke"):
        for encoding in plan["encodings"]:
            dicom_path = inputs_dir / f"synthetic_{encoding}.dcm"
            create_dicom(dicom_path, frames=frames, rows=rows, columns=columns,
                         encoding=encoding, seed=347)
            for device in plan["devices"]:
                for selection in plan["selections"]:
                    for condition_batch_size in plan_batch_sizes:
                        if time.monotonic() - started > max_runtime_seconds:
                            raise TimeoutError(
                                f"EXP-002 runtime budget exceeded before {encoding}/{device}/batch={condition_batch_size}"
                            )
                        print(
                            f"[START] {profile}: {encoding} / {device} / {selection} / "
                            f"batch={condition_batch_size} / prefetch={plan['prefetch_depths']}",
                            flush=True,
                        )
                        result = benchmark_prefetch_sweep(
                            dicom_path, backend="highdicom", inference_type="torch",
                            device=device, selection=selection,
                            batch_size=condition_batch_size,
                            baseline_prefetch=plan["baseline_prefetch"],
                            prefetch_depths=plan["prefetch_depths"],
                            repeats=plan["repeats"], warmups=plan["warmups"],
                            torch_threads=1, profile=True,
                            detailed_torch=True, telemetry=True,
                        )
                        if result["metadata"]["device"] != device:
                            raise RuntimeError(f"Device mismatch; requested {device}")
                        if not result["metadata"]["source_is_synthetic"]:
                            raise RuntimeError("Non-synthetic input detected; aborting")
                        if not result["metadata"]["identical_decoded_frames_across_all_trials"]:
                            raise RuntimeError("EXP-002 decoded-frame hashes mismatch; aborting")
                        label = f"{encoding}_{device}_uniform-32_batch-{condition_batch_size}.json"
                        save_result(result, results_dir / label)
                        for key, values in result["summary"].items():
                            mode, depth = key.split("@")
                            summary_rows.append({
                                "profile": profile, "encoding": encoding,
                                "device": device, "selection": selection,
                                "batch_size": condition_batch_size, "prefetch": int(depth),
                                "frames_selected": result["metadata"]["frames_selected"],
                                "mode": mode, "trial_count": values["trial_count"],
                                "median_total_ms": values["median_total_ms"],
                                "median_fps": values["median_fps"],
                                "hashes_match": True,
                            })
                        print(
                            f"[DONE] {encoding}/{device}/batch={condition_batch_size}; "
                            f"series={len(result['summary'])}", flush=True,
                        )
    else:
      for encoding in plan["encodings"]:
        dicom_path = inputs_dir / f"synthetic_{encoding}.dcm"
        create_dicom(dicom_path, frames=frames, rows=rows, columns=columns,
                     encoding=encoding, seed=347)
        for device in plan["devices"]:
            for selection in plan["selections"]:
                for condition_batch_size in plan_batch_sizes:
                    print(
                        f"[START] {profile}: {encoding} / {device} / "
                        f"{selection} / batch={condition_batch_size}",
                        flush=True,
                    )
                    result = benchmark(
                        dicom_path,
                        backend="highdicom",
                        inference_type="torch",
                        device=device,
                        selection=selection,
                        batch_size=condition_batch_size,
                        prefetch=prefetch,
                        repeats=plan["repeats"],
                        warmups=plan["warmups"],
                        torch_threads=1,
                    )
                    if result["metadata"]["device"] != device:
                        raise RuntimeError(f"Device mismatch; requested {device}")
                    if not result["metadata"]["source_is_synthetic"]:
                        raise RuntimeError("Non-synthetic input detected; aborting")
                    if not result["metadata"]["identical_decoded_frames_across_all_trials"]:
                        raise RuntimeError("A/B/C decoded-frame hashes mismatch; aborting")
                    label = (
                        f"{encoding}_{device}_{selection.replace(':', '-')}_"
                        f"batch-{condition_batch_size}.json"
                    )
                    save_result(result, results_dir / label)
                    baseline = result["summary"]["A"]["median_total_ms"]
                    for mode, values in result["summary"].items():
                        summary_rows.append({
                            "profile": profile,
                            "encoding": encoding,
                            "device": device,
                            "selection": selection,
                            "batch_size": condition_batch_size,
                            "prefetch": prefetch,
                            "frames_selected": result["metadata"]["frames_selected"],
                            "mode": mode,
                            "median_total_ms": values["median_total_ms"],
                            "median_fps": values["median_fps"],
                            "speedup_vs_A": round(baseline / values["median_total_ms"], 4),
                            "hashes_match": True,
                        })
                    print(
                        f"[DONE] batch={condition_batch_size}; "
                        f"A={baseline:.2f} ms; "
                        f"B={result['summary']['B']['median_total_ms']:.2f} ms; "
                        f"C={result['summary']['C']['median_total_ms']:.2f} ms",
                        flush=True,
                    )

    result_csv = output / "summary.csv"
    with result_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    machine["experiment_count"] = plan.get("shared_configurations", len(summary_rows) // 3)
    machine["result_rows"] = len(summary_rows)
    machine["method_series"] = len(summary_rows)
    machine["batch_sizes"] = plan_batch_sizes
    machine["mode_order_policy"] = plan.get(
        "mode_order_policy",
        "randomized per measured repeat by runner seed 347 + repeat index",
    )
    (output / "machine.json").write_text(json.dumps(machine, indent=2) + "\n")
    return {"output": str(output), "conditions": machine["experiment_count"],
            "summary_csv": str(result_csv), "gpu_device": machine["available_gpus"][0]["name"]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=["pilot", "full", "validation2", "exp002_smoke", "exp002"], default="pilot")
    p.add_argument("--out", type=Path, default=Path("/kaggle/working/dicom_edge_results"))
    p.add_argument("--frames", type=int, default=96)
    p.add_argument("--rows", type=int, default=128)
    p.add_argument("--columns", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--prefetch", type=int, default=2)
    p.add_argument("--max-experiments", type=int, default=64)
    p.add_argument("--max-runtime-seconds", type=int, default=900)
    args = p.parse_args(argv)
    result = run(profile=args.profile, output=args.out, frames=args.frames,
                 rows=args.rows, columns=args.columns,
                 batch_size=args.batch_size, prefetch=args.prefetch,
                 max_experiments=args.max_experiments,
                 max_runtime_seconds=args.max_runtime_seconds)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
