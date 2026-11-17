# Experimental Protocol — v0.1

## Aim / hypothesis

Assess whether selective frame retrieval changes end-to-end ingestion + inference latency and process resource usage on resource-constrained edge hardware. The critical hypothesis is that fewer unnecessary DICOM decode operations can lower CPU demand and the ingestion critical path **when frame selection is a real workload requirement**.

Not every speedup is attributable to decoding: file-system caching, codec choice, pydicom metadata overhead, interpreter scheduling, and thermal throttling can all dominate small workloads.

## Factorial design

- Hardware: development CPU, Jetson Nano (only if a compatible torch/CUDA environment exists), Jetson Orin, Raspberry Pi.
- Input encoding: native Explicit VR Little Endian; JPEG Baseline (Pillow-generated monochrome fixture).
- Input size: 96 × 128 × 128 for smoke runs, then 714 × 128 × 128 and a larger frame size only if device memory allows.
- Selection: 1, 4, 16, 64, 96 selected frames. Also test noncontiguous indices (e.g. `uniform:16`), contiguous (`first:16`), and varying selection ratios.
- Mode A: repeated highdicom `read_frame()` access.
- Mode B: selective `read_frames()` on fixed-size chunks.
- Mode C: same selective chunks with a bounded prefetch queue and concurrent inference.
- Inference: same seeded untrained CNN with CPU or CUDA (`torch`). This is a performance workload, **not an AI accuracy study**.

For each platform and condition, use ≥3 warmups and ≥10 repeats after pilot runs. Record Python, pydicom, highdicom feature branch SHA, PyTorch, JetPack, CUDA runtime, DICOM codec packages, frame size, power mode and clock state. Do not compare across different decode libraries without noting that distinction.

## Primary outcome

Median and dispersion of **end-to-end wall-clock milliseconds** for completing a selected-frame inference pass. Calculate effective frames/sec using total wall time. Only claim an improvement if the decoded-frame SHA256 signatures match across A/B/C.

## Secondary outcomes

- `decode_work_ms` / `inference_work_ms`, with overlap caution in C.
- Sampled peak resident memory, Python traced allocations, PyTorch GPU tensor peak.
- Process CPU core-equivalent usage and CPU time.
- External Jetson tegrastats raw logs, separately archived and versioned (not parsed into reliable per-trial GPU-usage or energy values in v0.1).

**Not implemented in v0.1:** GPU idle-time attribution, energy/joule measurement, CPU thread core affinity, sensor timestamp synchronization, TensorRT benchmarking, CUDA pinned-memory transfers, double-buffering with CUDA streams. These are future controlled experiments, not claims.

## Validity controls

1. Use the same fixed DICOM file and selected frame indices for all modes in a stratum.
2. Run one mode at a time for sequential trials; randomized mode order; no concurrent test process.
3. Control input frame size, codec and JPEG quality.
4. Avoid CPU/GPU thermal-throttle comparisons across different power modes.
5. Align the PyTorch / CUDA setup and device selection across A/B/C.
6. Record unsupported conditions or missing codec plugins as not run, not as success.
7. Run the upstream highdicom regression suite where fixtures are present; fixture-related skips are not passing checks.
8. Benchmark on *locally generated synthetic files only* unless explicit permissions and privacy controls exist for real DICOM inputs. Never publish PHI or study UIDs.

## Analysis

Use paired per-repeat comparisons. Report median A, B, C runtime, percentage difference versus A, decoded-frame equivalence, confidence/uncertainty where repeats are sufficient, and failure rates. Do not report power savings without calibrated power telemetry. The benchmark's simulated CNN provides no clinical accuracy, sensitivity or safety evidence.

## Stop / failure criteria

- Any decoded-frame digest mismatch → block performance conclusions.
- CUDA unavailable when requested → exit rather than silently falling back to CPU.
- Memory capacity exceeded or throttling → reduce frame count and document the deviation.
- Highdicom feature branch missing `read_frames` → do not substitute unrelated package silently.
