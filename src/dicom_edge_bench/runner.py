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
from .profiling import ProfileTrace
from .selection import parse_selection
from .telemetry import TelemetryCollector

_MODES = ("A", "B", "C")
_STOP = object()


def _digest_update(sha: Any, frame: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(frame)
    sha.update(str(contiguous.dtype).encode())
    sha.update(str(contiguous.shape).encode())
    sha.update(contiguous.tobytes(order="C"))


def _timed_frames(path: Path, indices: list[int], backend: str,
                  chunk_size: int, mode: str, inference: Inference,
                  trace: ProfileTrace | None = None):
    """Sequential A or B, tracking decoding time independently from inference."""
    decode_seconds = infer_seconds = 0.0
    digest = hashlib.sha256()
    with decoded_frames(path, indices, mode, backend, chunk_size) as frames:
        it = iter(frames)
        for frame_id, _ in enumerate(indices):
            phase = trace.phase("decode_frame", frame_id=frame_id) if trace else _null_context()
            with phase:
                before_decode = time.perf_counter()
                frame = next(it)
                decode_seconds += time.perf_counter() - before_decode
            _digest_update(digest, frame)
            before_infer = time.perf_counter()
            inference.process(frame, trace=trace, frame_id=frame_id)
            infer_seconds += time.perf_counter() - before_infer
    return decode_seconds, infer_seconds, digest.hexdigest()


def _prefetched_frames(path: Path, indices: list[int], backend: str,
                       chunk_size: int, inference: Inference, prefetch: int,
                       trace: ProfileTrace | None = None):
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
    queue_stats = {
        "prefetch_capacity": prefetch,
        "enqueue_wait_ms": 0.0,
        "queue_block_ms": 0.0,
        "dequeue_wait_ms": 0.0,
        "max_occupancy": 0,
        "producer_exception": None,
    }

    def record(event: str, **fields):
        if trace is not None:
            trace.timeline.record(event, **fields)

    def put(value, *, frame_id=None):
        start = time.perf_counter_ns()
        blocked_ns = 0
        record("producer_enqueue_start", frame_id=frame_id)
        while not cancelled.is_set():
            try:
                slots.put(value, timeout=0.1)
                elapsed_ns = time.perf_counter_ns() - start
                queue_stats["enqueue_wait_ms"] += elapsed_ns / 1_000_000
                queue_stats["queue_block_ms"] += blocked_ns / 1_000_000
                queue_stats["max_occupancy"] = max(queue_stats["max_occupancy"], slots.qsize())
                record("producer_enqueue_finish", frame_id=frame_id,
                       elapsed_ns=elapsed_ns, queue_size=slots.qsize(),
                       blocked_ns=blocked_ns)
                return
            except queue.Full:
                full_start = time.perf_counter_ns()
                record("producer_queue_full", frame_id=frame_id,
                       queue_size=slots.qsize())
                while slots.full() and not cancelled.is_set():
                    time.sleep(0.001)
                blocked_ns += time.perf_counter_ns() - full_start
                continue

    def producer():
        record("producer_thread_start")
        try:
            with decoded_frames(path, indices, "B", backend, chunk_size) as frames:
                it = iter(frames)
                for frame_id, _ in enumerate(indices):
                    phase = trace.phase("decode_frame", frame_id=frame_id) if trace else _null_context()
                    with phase:
                        before = time.perf_counter()
                        item = next(it)
                        producer_decode[0] += time.perf_counter() - before
                    if cancelled.is_set():
                        return
                    put(item, frame_id=frame_id)
        except Exception as exc:
            queue_stats["producer_exception"] = repr(exc)
            record("producer_exception", error=repr(exc))
            put(exc)
        finally:
            put(_STOP, frame_id=None)
            record("producer_thread_finish")

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="dicom-decode") as pool:
        future = pool.submit(producer)
        try:
            count = 0
            while True:
                dequeue_start = time.perf_counter_ns()
                record("consumer_dequeue_start", queue_size=slots.qsize())
                item = slots.get(timeout=60.0)
                dequeue_elapsed_ns = time.perf_counter_ns() - dequeue_start
                queue_stats["dequeue_wait_ms"] += dequeue_elapsed_ns / 1_000_000
                record("consumer_dequeue_finish", elapsed_ns=dequeue_elapsed_ns,
                       queue_size=slots.qsize())
                if item is _STOP:
                    break
                if isinstance(item, Exception):
                    raise item
                _digest_update(digest, item)
                before = time.perf_counter()
                inference.process(item, trace=trace, frame_id=count)
                infer_seconds += time.perf_counter() - before
                count += 1
            future.result()
            if count != len(indices):
                raise RuntimeError(f"Prefetched {count}, expected {len(indices)} frames")
        finally:
            cancelled.set()
            # Future is awaited by ThreadPoolExecutor; producer's puts respect
            # cancellation so a consumer exception cannot deadlock the worker.
    if trace is not None:
        trace.queue = {
            **queue_stats,
            "producer_decode_ms": producer_decode[0] * 1000,
            "consumer_inference_ms": infer_seconds * 1000,
        }
    return producer_decode[0], infer_seconds, digest.hexdigest()


def _trial(path: Path, indices: list[int], backend: str, inference: Inference,
           mode: str, chunk_size: int, prefetch: int,
           trace: ProfileTrace | None = None,
           telemetry: TelemetryCollector | None = None) -> dict[str, Any]:
    inference.synchronize()
    inference.reset_gpu_peak()
    if trace is not None:
        trace.timeline.record("pipeline_start", mode=mode)
    if telemetry is not None:
        trace.telemetry["start"] = telemetry.snapshot("start")
    with Measurements() as m:
        if mode == "C":
            decoded, inferred, signature = _prefetched_frames(
                path, indices, backend, chunk_size, inference, prefetch, trace
            )
        else:
            decoded, inferred, signature = _timed_frames(
                path, indices, backend, chunk_size, mode, inference, trace
            )
        inference.synchronize()
    if telemetry is not None:
        trace.telemetry["end"] = telemetry.snapshot("end")
    if trace is not None:
        trace.timeline.record("pipeline_finish", mode=mode)
    result = {
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
    if trace is not None:
        result["profiling"] = trace.to_dict()
    return result


def _null_context():
    class Context:
        def __enter__(self):
            return None

        def __exit__(self, *exc):
            return False
    return Context()


def benchmark(path: str | Path, *, backend: str, inference_type: str,
              device: str, selection: str, batch_size: int,
              prefetch: int, repeats: int, warmups: int,
              torch_threads: int = 1, profile: bool = False,
              detailed_torch: bool = False, telemetry: bool = False,
              modes: tuple[str, ...] = _MODES) -> dict[str, Any]:
    """Benchmark three strategies with identical data and a stable dummy model."""
    from pydicom import dcmread, __version__ as pydicom_version
    if repeats < 1 or warmups < 0 or batch_size < 1 or prefetch < 1:
        raise ValueError("repeats/batch-size/prefetch >= 1, warmups >= 0")
    if not modes or any(mode not in _MODES for mode in modes):
        raise ValueError(f"modes must be a non-empty subset of {_MODES}")
    path = Path(path).resolve(strict=True)
    setup_trace = ProfileTrace(enabled=profile, detailed_torch=detailed_torch)
    with setup_trace.phase("metadata_preparation"):
        metadata = dcmread(path, stop_before_pixels=True)
    n_frames = int(getattr(metadata, "NumberOfFrames", 1))
    channels = int(metadata.SamplesPerPixel)
    with setup_trace.phase("frame_selection"):
        indices = parse_selection(selection, n_frames)
    backend_version = pydicom_version
    if backend == "highdicom":
        import highdicom
        backend_version = str(highdicom.__version__)
    context = Inference.create(inference_type, device, channels, torch_threads)
    telemetry_collector = TelemetryCollector(enabled=telemetry) if profile else None
    results = []
    for _ in range(warmups):
        for mode in modes:
            _trial(path, indices, backend, context, mode, batch_size, prefetch)

    # Rotate order to limit systematic thermal/cache effects from always
    # measuring modes in the same sequence. Warmups aren't in measurements.
    for trial_num in range(repeats):
        ordered_modes = list(modes)
        random.Random(347 + trial_num).shuffle(ordered_modes)
        for mode in ordered_modes:
            trace = ProfileTrace(enabled=True, detailed_torch=detailed_torch) if profile else None
            row = _trial(path, indices, backend, context, mode, batch_size, prefetch,
                         trace=trace, telemetry=telemetry_collector)
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
        } for mode in modes
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
        "profiling": {
            "enabled": profile,
            "detailed_torch": detailed_torch,
            "telemetry": telemetry,
            "setup": setup_trace.to_dict() if profile else None,
            "phase_clock": "time.perf_counter_ns",
            "cuda_clock": "torch.cuda.Event" if detailed_torch else "not_measured",
        },
        "modes": list(modes),
        "mode_order_policy": "randomized per measured repeat by runner seed 347 + repeat index",
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


def benchmark_prefetch_sweep(
    path: str | Path, *, backend: str, inference_type: str, device: str,
    selection: str, batch_size: int, baseline_prefetch: int,
    prefetch_depths: list[int], repeats: int, warmups: int,
    torch_threads: int = 1, profile: bool = True,
    detailed_torch: bool = False, telemetry: bool = False,
) -> dict[str, Any]:
    """Run A/B once and C at each depth for every measured repeat.

    This keeps A and B as one shared baseline while randomizing the execution
    order of A, B and each C depth within each repeat.
    """
    if not prefetch_depths or any(depth < 1 for depth in prefetch_depths):
        raise ValueError("prefetch_depths must contain positive integers")
    if baseline_prefetch < 1:
        raise ValueError("baseline_prefetch must be positive")
    path = Path(path).resolve(strict=True)
    from pydicom import dcmread, __version__ as pydicom_version

    setup_trace = ProfileTrace(enabled=profile, detailed_torch=detailed_torch)
    with setup_trace.phase("metadata_preparation"):
        metadata = dcmread(path, stop_before_pixels=True)
    n_frames = int(getattr(metadata, "NumberOfFrames", 1))
    channels = int(metadata.SamplesPerPixel)
    with setup_trace.phase("frame_selection"):
        indices = parse_selection(selection, n_frames)
    backend_version = pydicom_version
    if backend == "highdicom":
        import highdicom
        backend_version = str(highdicom.__version__)
    context = Inference.create(inference_type, device, channels, torch_threads)
    telemetry_collector = TelemetryCollector(enabled=telemetry) if profile else None
    planned = [("A", baseline_prefetch), ("B", baseline_prefetch)]
    planned.extend(("C", depth) for depth in prefetch_depths)
    results: list[dict[str, Any]] = []

    for _ in range(warmups):
        for mode, depth in planned:
            _trial(path, indices, backend, context, mode, batch_size, depth)

    for repeat in range(repeats):
        ordered = list(planned)
        random.Random(347 + repeat).shuffle(ordered)
        for mode, depth in ordered:
            trace = ProfileTrace(enabled=True, detailed_torch=detailed_torch) if profile else None
            row = _trial(
                path, indices, backend, context, mode, batch_size, depth,
                trace=trace, telemetry=telemetry_collector,
            )
            row["repeat"] = repeat + 1
            row["prefetch"] = depth
            results.append(row)

    signatures = {row["decoded_sha256"] for row in results}
    summary: dict[str, dict[str, Any]] = {}
    for mode, depth in planned:
        key = f"{mode}@{depth}"
        selected = [row for row in results if row["mode"] == mode and row["prefetch"] == depth]
        summary[key] = {
            "mode": mode,
            "prefetch": depth,
            "trial_count": len(selected),
            "median_total_ms": round(statistics.median(row["total_ms"] for row in selected), 3),
            "median_fps": round(statistics.median(row["frames_per_second"] for row in selected), 3),
        }
    return {
        "metadata": {
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
            "torch_version": str(context.torch.__version__) if context.torch is not None else None,
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
            "baseline_prefetch": baseline_prefetch,
            "prefetch_depths": prefetch_depths,
            "repeats": repeats,
            "warmups": warmups,
            "torch_threads": torch_threads,
            "modes": [f"A@{baseline_prefetch}", f"B@{baseline_prefetch}"] + [f"C@{d}" for d in prefetch_depths],
            "mode_order_policy": "randomized per measured repeat by runner seed 347 + repeat index",
            "identical_decoded_frames_across_all_trials": len(signatures) == 1,
            "profiling": {
                "enabled": profile,
                "detailed_torch": detailed_torch,
                "telemetry": telemetry,
                "setup": setup_trace.to_dict() if profile else None,
                "phase_clock": "time.perf_counter_ns",
                "cuda_clock": "torch.cuda.Event" if detailed_torch else "not_measured",
            },
        },
        "summary": summary,
        "trials": results,
    }


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
