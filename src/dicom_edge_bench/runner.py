"""Three-mode benchmark with identical dataset, selection, model and chunks."""
from __future__ import annotations

import hashlib
import json
import platform
import queue
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .decoding import decoded_frames
from .inference import Inference
from .metrics import Measurements
from .selection import parse_selection

_MODES = ("A", "B", "C")
_STOP = object()


def _digest_update(sha: Any, frame: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(frame)
    sha.update(str(contiguous.dtype).encode())
    sha.update(str(contiguous.shape).encode())
    sha.update(contiguous.tobytes(order="C"))


def _timed_frames(path: Path, indices: list[int], backend: str,
                  chunk_size: int, mode: str, inference: Inference):
    """Sequential A or B, tracking decoding time independently from inference."""
    decode_seconds = infer_seconds = 0.0
    digest = hashlib.sha256()
    with decoded_frames(path, indices, mode, backend, chunk_size) as frames:
        it = iter(frames)
        for _ in indices:
            before_decode = time.perf_counter()
            frame = next(it)
            decode_seconds += time.perf_counter() - before_decode
            _digest_update(digest, frame)
            before_infer = time.perf_counter()
            inference.process(frame)
            infer_seconds += time.perf_counter() - before_infer
    return decode_seconds, infer_seconds, digest.hexdigest()


def _prefetched_frames(path: Path, indices: list[int], backend: str,
                       chunk_size: int, inference: Inference, prefetch: int):
    """Single decoding producer; bounded queue; inference on the consumer.

    The reader is constructed and closed on the producer's thread.
    The queue never retains more than `prefetch` decoded frames at once.
    The workload is identical to B, only scheduling differs.
    """
    slots = queue.Queue(maxsize=prefetch)
    cancelled = threading.Event()
    producer_decode = [0.0]
    infer_seconds = 0.0
    digest = hashlib.sha256()

    def put(value):
        while not cancelled.is_set():
            try:
                slots.put(value, timeout=0.1)
                return
            except queue.Full:
                continue

    def producer():
        try:
            with decoded_frames(path, indices, "B", backend, chunk_size) as frames:
                it = iter(frames)
                for _ in indices:
                    before = time.perf_counter()
                    item = next(it)
                    producer_decode[0] += time.perf_counter() - before
                    if cancelled.is_set():
                        return
                    put(item)
        except Exception as exc:
            put(exc)
        finally:
            put(_STOP)

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="dicom-decode") as pool:
        future = pool.submit(producer)
        try:
            count = 0
            while True:
                item = slots.get(timeout=60.0)
                if item is _STOP:
                    break
                if isinstance(item, Exception):
                    raise item
                _digest_update(digest, item)
                before = time.perf_counter()
                inference.process(item)
                infer_seconds += time.perf_counter() - before
                count += 1
            future.result()
            if count != len(indices):
                raise RuntimeError(f"Prefetched {count}, expected {len(indices)} frames")
        finally:
            cancelled.set()
            # Future is awaited by ThreadPoolExecutor; producer's puts respect
            # cancellation so a consumer exception cannot deadlock the worker.
    return producer_decode[0], infer_seconds, digest.hexdigest()


def _trial(path: Path, indices: list[int], backend: str, inference: Inference,
           mode: str, chunk_size: int, prefetch: int) -> dict[str, Any]:
    inference.synchronize()
    inference.reset_gpu_peak()
    with Measurements() as m:
        if mode == "C":
            decoded, inferred, signature = _prefetched_frames(
                path, indices, backend, chunk_size, inference, prefetch
            )
        else:
            decoded, inferred, signature = _timed_frames(
                path, indices, backend, chunk_size, mode, inference
            )
        inference.synchronize()
    return {
        "mode": mode,
        "frame_count": len(indices),
        "total_ms": round(m.wall_seconds * 1000, 3),
        "decode_work_ms": round(decoded * 1000, 3),
        "inference_work_ms": round(inferred * 1000, 3),
        "cpu_time_ms": round(m.cpu_seconds * 1000, 3),
        "cpu_core_equivalent_percent": round(
            m.cpu_seconds / max(m.wall_seconds, 1e-10) * 100, 2
        ),
        "python_alloc_peak_mib": round(m.python_peak_mib, 4),
        "sampled_process_rss_peak_mib": (
            round(m.rss_sampled_mib, 3) if m.rss_sampled_mib is not None else None
        ),
        "torch_gpu_tensor_peak_mib": (
            round(inference.gpu_peak_mib(), 3)
            if inference.gpu_peak_mib() is not None else None
        ),
        "decoded_sha256": signature,
        "frames_per_second": round(len(indices) / m.wall_seconds, 3),
    }


def benchmark(path: str | Path, *, backend: str, inference_type: str,
              device: str, selection: str, batch_size: int,
              prefetch: int, repeats: int, warmups: int,
              torch_threads: int = 1) -> dict[str, Any]:
    """Benchmark three strategies with identical data and a stable dummy model."""
    from pydicom import dcmread, __version__ as pydicom_version
    if repeats < 1 or warmups < 0 or batch_size < 1 or prefetch < 1:
        raise ValueError("repeats/batch-size/prefetch >= 1, warmups >= 0")
    path = Path(path).resolve(strict=True)
    metadata = dcmread(path, stop_before_pixels=True)
    n_frames = int(getattr(metadata, "NumberOfFrames", 1))
    channels = int(metadata.SamplesPerPixel)
    indices = parse_selection(selection, n_frames)
    backend_version = pydicom_version
    if backend == "highdicom":
        import highdicom
        backend_version = str(highdicom.__version__)
    context = Inference.create(inference_type, device, channels, torch_threads)
    results = []
    for _ in range(warmups):
        for mode in _MODES:
            _trial(path, indices, backend, context, mode, batch_size, prefetch)

    # Rotate order to limit systematic thermal/cache effects from always
    # measuring modes in the same sequence. Warmups aren't in measurements.
    for trial_num in range(repeats):
        modes = list(_MODES)
        random.Random(347 + trial_num).shuffle(modes)
        for mode in modes:
            row = _trial(path, indices, backend, context, mode, batch_size, prefetch)
            row["repeat"] = trial_num + 1
            results.append(row)

    signatures = {row["decoded_sha256"] for row in results}
    summary = {
        mode: {
            "median_total_ms": round(statistics.median(
                r["total_ms"] for r in results if r["mode"] == mode
            ), 3),
            "median_fps": round(statistics.median(
                r["frames_per_second"] for r in results if r["mode"] == mode
            ), 3),
        } for mode in _MODES
    }
    metadata_summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "project_version": "0.1.0",
        "dataset_basename": path.name,
        "dataset_bytes": path.stat().st_size,
        "source_is_synthetic": str(metadata.get("PatientID", "")) == "SYNTHETIC-ONLY",
        "transfer_syntax_uid": str(metadata.file_meta.TransferSyntaxUID),
        "available_frames": n_frames,
        "frames_selected": len(indices),
        "selection": selection,
        "backend": backend,
        "backend_version": backend_version,
        "pydicom_version": pydicom_version,
        "inference_type": inference_type,
        "device": context.device,
        "torch_version": (
            str(context.torch.__version__) if context.torch is not None else None
        ),
        "torch_cuda_runtime_version": (
            str(context.torch.version.cuda)
            if context.torch is not None and context.device == "cuda" else None
        ),
        "torch_cuda_device_name": (
            context.torch.cuda.get_device_name()
            if context.torch is not None and context.device == "cuda" else None
        ),
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "architecture": platform.machine(),
        "batch_size": batch_size,
        "prefetch_queue_capacity_frames": prefetch,
        "repeats": repeats,
        "warmups": warmups,
        "torch_threads": torch_threads,
        "identical_decoded_frames_across_all_trials": len(signatures) == 1,
        "metrics_caveat": (
            "tracemalloc measures Python allocations only; RSS is sampled "
            "and not guaranteed true peak; GPU tensor peak excludes driver "
            "and non-PyTorch allocations. Decode/inference work times can "
            "overlap in C and must not be summed to estimate wall time."
        ),
        "model_caveat": (
            "Untrained synthetic convolution model; no clinical or model "
            "accuracy assertions. GPU idle time, power and energy not measured."
        ),
    }
    return {"metadata": metadata_summary, "summary": summary, "trials": results}


def save_result(result: dict, destination: str | Path) -> Path:
    import csv
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    csv_path = destination.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["trials"][0]))
        writer.writeheader()
        writer.writerows(result["trials"])
    return destination
