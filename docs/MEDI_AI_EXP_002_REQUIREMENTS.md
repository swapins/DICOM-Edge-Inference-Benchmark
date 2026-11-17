# MEDI-AI-EXP-002 Requirements

**Title:** Hardware-Aware Medical Imaging Inference Profiling
**Programme:** MEDI AI+
**Research organization:** Provenant AI Labs Pte. Ltd., Singapore
**Research lead:** Swapin Vidya
**Status:** In progress until validated results exist

## Research question

Which CPU decoding, memory-transfer, inference, queueing, and synchronization costs determine end-to-end DICOM inference performance?

## Hypotheses to test

- **H1:** CPU decoding and preprocessing can dominate the critical path even when GPU inference is available.
- **H2:** Bounded prefetching may underperform because of producer/consumer synchronization, queue waits, thread overhead, or insufficient useful overlap.
- **H3:** The best batch size and prefetch depth may depend on encoding, workload size, inference device, and available resources.

These are hypotheses under test, not established findings.

## Baseline and controls

EXP-002 uses the verified EXP-001 implementation as its reference:

- A: repeated individual-frame decoding;
- B: selective multi-frame decoding;
- C: selective decoding with bounded prefetching.

The reference highdicom revision is commit `d42f664006a9cd2194eda0bf5b1cf0fa2fb0188e`. Existing EXP-001 profiles and measured results remain unchanged. All methods use the same synthetic DICOM input, selected indices, backend, model, preprocessing, batch size, and requested device within a shared configuration.

## Instrumentation requirements

The optional EXP-002 instrumentation records monotonic host-clock phase boundaries, optional CUDA event timings, queue enqueue/dequeue events, queue blocking intervals, producer/consumer lifecycle, exceptions, occupancy, process RSS, CPU process telemetry, CUDA memory, and supported NVIDIA/Jetson telemetry. Unsupported values are recorded as unavailable rather than zero.

The instrumentation must preserve decoded-frame hashes and must not silently fall back from a requested CUDA run to CPU. Detailed CUDA events and external telemetry are opt-in because they can add measurement overhead.

## Exclusions

The experiment does not claim Jetson execution without Jetson hardware, energy efficiency without calibrated power measurement, clinical validity, diagnostic accuracy, proprietary MEDI AI+ scheduling, or Bio-JERE arbitration.
