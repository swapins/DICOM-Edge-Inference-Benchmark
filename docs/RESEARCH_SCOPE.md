# Scope and upstream isolation

Project: DICOM-Edge-Inference-Benchmark

Research directions:
1. Quantify resource cost of repeated vs selective DICOM frame decoding.
2. Characterize when bounded CPU prefetch overlaps with downstream inference on heterogeneous edge systems.
3. Produce comparable CPU / CUDA / power-mode datasets on supported hardware.
4. Explore future edge-aware policies for buffer size, frame selection density and computational resource allocation.

This is **not** a claim of novel GPU decoding, a CUDA extension, proprietary edge OS execution control, hardware-aware orchestration, or medical validation. Those would require separate engineering mechanisms, tests and evidence. Do not modify, amend or represent the highdicom upstream PR #520 as containing this benchmark.

Milestones:
- M1 (this scaffold): synthetic DICOM generation; A/B/C runner; frame-output equivalence checking; CSV/JSON; documentation; basic tests.
- M2: reproduce highdicom feature branch results locally and add full CI with fixture availability.
- M3: validate CPU / Jetson Orin / Raspberry Pi runs; add hardware metadata.
- M4: energy measurement with synchronized calibrated telemetry; GPU queue and idle-time tracking.
- M5: bounded adaptive prefetch policy and full statistical analysis.

Do not publish client or patient data, proprietary medical datasets, or machine-identifying secrets.
