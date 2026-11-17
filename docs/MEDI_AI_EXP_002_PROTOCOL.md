# MEDI-AI-EXP-002 Protocol

## Constrained GPU pilot

The first controlled pilot uses the existing Kaggle notebook and one Tesla T4 GPU:

| Factor | Levels |
| --- | --- |
| Encoding | native, JPEG |
| Device | CPU, CUDA |
| Selection | `uniform:32` from 96 frames |
| Batch size | 8, 16, 32 |
| Method A/B baseline prefetch | 2 |
| Method C prefetch | 1, 2, 4 |
| Warmups | 3, excluded from statistics |
| Measured repeats | 10 per method/prefetch series |

There are 12 shared configurations: 2 encodings × 2 devices × 3 batch sizes. A and B are each executed once per shared configuration. C is executed at each of the three prefetch depths. The effective output contains 60 method series and 600 measured method trials. The execution plan records this count before measurement and refuses to exceed the configured experiment-series or runtime budget.

Within each measured repeat, the runner randomizes A, B, and all C-depth executions using seed `347 + repeat index`. Warmups are not included in summaries.

## Measurements

Host phases use `time.perf_counter_ns()` and are kept separate from CUDA event timings. The profiled phases are metadata preparation, frame selection, frame decoding, preprocessing, host-to-device transfer, inference, and GPU synchronization. Method C records producer enqueue start/finish, queue-full intervals, consumer dequeue start/finish, occupancy, producer lifecycle, exceptions, and pipeline boundaries.

Memory and telemetry fields distinguish sampled RSS, Python allocation peak, PyTorch allocated/reserved/peak values where available, and external hardware counters. Missing GPU or Jetson counters are represented as null/unavailable with an explanation.

## Analysis

For each method series report sample count, median total/decode/preprocessing/transfer/inference timing, median absolute deviation, observed range, and percentile bootstrap 95% intervals. Use 20,000 bootstrap resamples with replacement, Python `random.Random(20261117)`, and the empirical 2.5th/97.5th percentiles. These are exploratory intervals from one execution environment and are not population-level guarantees.

Compare C across prefetch depths and distinguish directly measured queue waits from inferred overlap proxies. Do not attribute causality to scheduling without direct evidence.

## Provenance and safety

Install highdicom from the pinned commit above and record `direct_url.json` provenance. Use synthetic DICOM files only. Keep Kaggle credentials, raw DICOM files, generated results, and telemetry logs outside Git. Preserve the EXP-001 pilot and Validation Run 2 archives unchanged.
