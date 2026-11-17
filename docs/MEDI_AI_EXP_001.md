# MEDI-AI-EXP-001: Selective DICOM Frame Decoding for Edge AI Inference

| Field | Record |
| --- | --- |
| Programme | MEDI AI+ |
| Experiment identifier | MEDI-AI-EXP-001 |
| Research component | Medical imaging ingestion and inference optimization |
| Research organization | Provenant AI Labs Pte. Ltd., Singapore |
| Author | Swapin Vidya |
| Date | 17 November 2026 |
| Status | Initial Kaggle GPU pilot completed |
| Repository | [DICOM-Edge-Inference-Benchmark](https://github.com/swapins/DICOM-Edge-Inference-Benchmark) |

## Purpose

This is a component-level experiment within the MEDI AI+ research programme. It evaluates whether selective retrieval and decoding of required DICOM frames can reduce CPU-side processing overhead and improve end-to-end inference latency.

The benchmark is maintained in a separate GitHub repository so that the experiment can be reproduced independently. It remains separate from the upstream [highdicom PR #520](https://github.com/ImagingDataCommons/highdicom/pull/520), which supplies the tested selective multi-frame API.

## Hypotheses

### Primary hypothesis

Selective retrieval and decoding of required DICOM frames can reduce CPU-side processing overhead and improve end-to-end inference latency.

### Secondary hypothesis

Bounded prefetching may allow CPU decoding to overlap downstream inference, but its queueing and scheduling overhead must be evaluated experimentally.

## Experimental methods

- **A - Repeated individual-frame decoding:** retrieve each selected frame independently with `ImageFileReader.read_frame(i)`.
- **B - Selective decoding:** retrieve selected frames in bounded chunks with `ImageFileReader.read_frames(indices=...)`.
- **C - Selective decoding with bounded prefetching:** use the same selective chunks with a bounded producer queue while inference processes earlier chunks.

The pilot used synthetic DICOM inputs only: 96 frames at 128 x 128 pixels, `uniform:16` selection, one warmup, and two measured repeats. It tested native and JPEG inputs on CPU and CUDA using GPU 0 of a Kaggle Tesla T4 instance.

## Verified pilot record

- GPU: NVIDIA Tesla T4, GPU 0
- CUDA: 13.0
- PyTorch: 2.11.0+cu130
- Remote tests: 19 passed
- Local tests: 17 passed, 2 skipped
- highdicom revision: `d42f664006a9cd2194eda0bf5b1cf0fa2fb0188e`
- Decoded-frame hashes: matched across A/B/C within each condition

| Condition | A (ms) | B (ms) | C (ms) | B vs A |
| --- | ---: | ---: | ---: | ---: |
| Native CPU | 34.335 | 29.645 | 30.517 | 13.66% faster |
| Native CUDA | 31.407 | 26.859 | 28.519 | 14.48% faster |
| JPEG CPU | 36.941 | 32.229 | 36.806 | 12.76% faster |
| JPEG CUDA | 34.356 | 31.296 | 33.150 | 8.91% faster |

Selective decoding (B) reduced measured median end-to-end latency by 8.91-14.48% relative to repeated decoding (A) across the four tested Kaggle conditions. Bounded prefetching (C) did not outperform selective decoding alone (B) in this pilot. These observations come from a small synthetic pilot and do not establish universal speedup or clinical value.

## Relationship to MEDI AI+

The tested processing path is:

```text
DICOM ingestion
    -> selected-frame CPU decoding
    -> CPU/CUDA inference
    -> latency and memory measurements
```

The experiment is potentially relevant to resource-efficient medical image ingestion, edge inference preprocessing, CPU/GPU pipeline performance, future bounded-memory execution, and NVIDIA Jetson deployment research. These are research connections, not evidence about the broader MEDI AI+ platform.

This experiment does not establish:

- MEDI AI+ OS-level scheduling or proprietary runtime behavior
- Bio-JERE resource arbitration
- Clinical diagnostic performance
- Validated medical-device functionality
- GPU-accelerated DICOM decompression
- NVIDIA Jetson performance measurements

Those capabilities require separate implementations, controls, and evidence.

## Reproducibility

See [README.md](../README.md), [the local workflow](LOCAL_VSCODE_KAGGLE_WORKFLOW.md), [the Kaggle protocol](KAGGLE_PROTOCOL.md), and [the experiment protocol](EXPERIMENT_PROTOCOL.md). Raw DICOM inputs, benchmark results, and credentials remain excluded from Git by design.

## Licensing and attribution

This repository is released under the MIT License. Author and citation metadata are recorded in [AUTHORS.md](../AUTHORS.md) and [CITATION.cff](../CITATION.cff). highdicom remains an independent upstream open-source project under its own license and contribution history.
