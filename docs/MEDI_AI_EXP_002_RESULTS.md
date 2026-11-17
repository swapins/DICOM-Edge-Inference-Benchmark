# MEDI-AI-EXP-002 Results

**Status:** instrumentation and controlled pilot workflow prepared; remote execution currently blocked by Kaggle scheduling.

## Current execution state

Kaggle notebook version 6 was submitted to the existing private notebook `stellaraisg/dicom-edge-inference-benchmark` as the EXP-002 instrumentation smoke test. Its status remains **QUEUED** with no execution logs. The full EXP-002 matrix has not been submitted, and no GPU benchmark result is claimed. A duplicate Kaggle job will not be submitted while version 6 is queued.

The prepared implementation is on branch `research/medi-ai-exp-002` in draft PR [#2](https://github.com/swapins/DICOM-Edge-Inference-Benchmark/pull/2). GitHub CI passes for Python 3.10 and 3.12, plus the security check.

This document will be updated only from downloaded and independently checked EXP-002 outputs. It will record the exact Kaggle notebook version, archive SHA-256, highdicom provenance, condition counts, trial counts, phase timings, queue events, telemetry coverage, uncertainty, and limitations.

No Jetson hardware is connected to the current Windows development environment. Jetson execution status is therefore **NOT EXECUTED — HARDWARE REQUIRED** until a supported device and verified JetPack/PyTorch stack are available.

The EXP-001 baseline remains the reference implementation. No EXP-001 measured result is replaced by this document.
