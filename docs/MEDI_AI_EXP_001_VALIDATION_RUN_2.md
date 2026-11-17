# MEDI-AI-EXP-001 — Validation Run 2

**Status:** completed and locally retrieved
**Execution date:** 17 November 2026
**Remote platform:** Kaggle GPU notebook, single NVIDIA Tesla T4 (GPU 0)
**Kaggle notebook:** <https://www.kaggle.com/code/stellaraisg/dicom-edge-inference-benchmark> (version 5)

**Validated results archive SHA-256:** `825edaf29934c3b0b4efb15400ed595f45b26c22c2ac1c5fdfc57d3e615e8c39`

This validation run preserves the original pilot results. It does not change the experiment identifier and does not modify highdicom PR #520.

## Protocol

Validation Run 2 used the existing A/B/C benchmark implementation with a validation-specific orchestration profile:

- synthetic native and JPEG DICOM inputs;
- `uniform:32` frame selection from 96-frame, 128 × 128 inputs;
- CPU and CUDA inference on the same Kaggle machine;
- batch sizes 4, 8, 16, and 32;
- 3 warmups and 10 measured repeats per method and condition;
- randomized A/B/C order for each measured repeat using runner seed 347 plus repeat index;
- one bounded producer queue for Method C with capacity 2 frames;
- identical model and selected frames across methods within each condition.

The remote bundle recorded Python 3.13.16, PyTorch 2.11.0+cu130, CUDA runtime 13.0, pydicom 3.0.2, Pillow 13.0.0, and highdicom 0.29.0. Kaggle logs explicitly recorded the highdicom installation provenance as commit `d42f664006a9cd2194eda0bf5b1cf0fa2fb0188e`, the pinned PR #520 revision.

## Remote execution checks

The Kaggle logs recorded CUDA availability as `True`, GPU 0 and GPU 1 as NVIDIA Tesla T4 devices, and GPU 0 as the device used. The second T4 was not used. The remote test run reported **20 passed in 2.39 s**.

The downloaded `DICOM_Edge_Kaggle_Results.zip` passed an archive integrity check. It contains:

- 16 experimental conditions: 2 encodings × 2 devices × 4 batch sizes;
- 48 summary rows: 16 conditions × 3 methods;
- 16 JSON/CSV trial pairs;
- 30 measured trial records in every JSON file: 10 repeats × 3 methods;
- native and JPEG synthetic DICOM inputs;
- `machine.json`, `summary.csv`, and the result archive.

Every condition reported identical decoded-frame hashes across all methods and trials. All numeric measurement fields were finite. The archive contains synthetic data only; no patient-identifying values, Kaggle credentials, GitHub tokens, or private-key material were found during the output audit.

## Per-condition statistical summary

The sanitized [statistical summary CSV](MEDI_AI_EXP_001_VALIDATION_RUN_2_stats.csv) is derived only from the retained trial JSON records. It contains no DICOM payloads, patient data, credentials, or generated raw measurements beyond aggregate timing statistics.

For each method and condition, `median_ms` is the median of the 10 measured `total_ms` values, `mad_ms` is the median absolute deviation around that median, and `min_ms`/`max_ms` are the observed range. The `bootstrap_ci95_*` columns are percentile intervals from 20,000 bootstrap resamples of 10 observations with replacement. Each series uses Python's `random.Random` initialized with seed `20261117`; the interval is the empirical 2.5th through 97.5th percentile of the bootstrap medians. These are exploratory intervals from one Kaggle execution environment and ten repeats, not population-level confidence statements or evidence of cross-machine generalization.

The underlying archive and CSV verify 10 trials each for A, B, and C in all 16 conditions (480 measured method records total). The only condition where B's median was higher than A's was JPEG/CUDA/batch 8: 64.841 ms versus 64.159 ms.

## Measured end-to-end latency

Values are medians across the 10 measured repeats. Latency is in milliseconds. Positive percentages mean lower latency than Method A; negative percentages mean higher latency than Method A.

| Encoding | Device | Batch | A | B | C | B vs A | C vs A |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native | CPU | 4 | 56.93 | 53.32 | 59.37 | 6.33% | -4.30% |
| native | CPU | 8 | 56.57 | 51.02 | 56.96 | 9.80% | -0.69% |
| native | CPU | 16 | 56.44 | 44.22 | 48.42 | 21.65% | 14.20% |
| native | CPU | 32 | 56.12 | 40.80 | 45.35 | 27.30% | 19.19% |
| native | CUDA | 4 | 52.08 | 50.92 | 55.78 | 2.23% | -7.10% |
| native | CUDA | 8 | 53.38 | 49.82 | 52.75 | 6.67% | 1.19% |
| native | CUDA | 16 | 53.16 | 42.96 | 48.77 | 19.18% | 8.25% |
| native | CUDA | 32 | 53.42 | 39.58 | 42.62 | 25.91% | 20.23% |
| JPEG | CPU | 4 | 68.65 | 65.88 | 65.65 | 4.04% | 4.37% |
| JPEG | CPU | 8 | 68.47 | 66.35 | 71.03 | 3.10% | -3.74% |
| JPEG | CPU | 16 | 68.68 | 61.74 | 63.98 | 10.09% | 6.84% |
| JPEG | CPU | 32 | 69.27 | 61.46 | 63.65 | 11.27% | 8.11% |
| JPEG | CUDA | 4 | 65.19 | 62.07 | 64.17 | 4.79% | 1.57% |
| JPEG | CUDA | 8 | 64.16 | 64.84 | 66.27 | -1.06% | -3.29% |
| JPEG | CUDA | 16 | 63.27 | 59.14 | 62.10 | 6.53% | 1.86% |
| JPEG | CUDA | 32 | 63.27 | 56.09 | 58.93 | 11.35% | 6.86% |

Throughput is the reciprocal frame-rate measure recorded by the runner. Representative medians at batch 32 were:

- native CPU: A 570.2 FPS, B 784.3 FPS, C 705.7 FPS;
- native CUDA: A 599.0 FPS, B 808.5 FPS, C 750.9 FPS;
- JPEG CPU: A 462.0 FPS, B 520.7 FPS, C 502.7 FPS;
- JPEG CUDA: A 505.8 FPS, B 570.5 FPS, C 543.0 FPS.

These are measured Kaggle T4-platform results, not Jetson, Raspberry Pi, or other edge-hardware results.

## Decode, inference, and resource measurements

The JSON trial records include `decode_work_ms`, `inference_work_ms`, `total_ms`, frames per second, CPU core-equivalent percentage, sampled process RSS, and PyTorch tensor GPU peak memory. For example, at native CUDA batch 32, the medians were:

| Method | Decode work (ms) | Inference work (ms) | End-to-end (ms) | CPU core-equivalent | GPU tensor peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 29.92 | 18.58 | 53.42 | 101.55% | 10.189 MiB |
| B | 17.90 | 16.76 | 39.58 | 101.98% | 10.189 MiB |
| C | 17.82 | 17.42 | 42.62 | 107.06% | 10.189 MiB |

At that condition, sampled process RSS medians were 991.463 MiB for A and B and 991.471 MiB for C. Across the measured conditions, CPU core-equivalent utilization was approximately one core for A/B and higher for C, especially on JPEG inputs. RSS is sampled rather than a guaranteed process maximum; GPU tensor peak excludes driver and other non-PyTorch allocations. GPU utilization, power, thermals, energy, and transfer timing were not measured.

## Method C profiling interpretation

Method C uses one producer thread and a bounded queue. The recorded phase timings show whether end-to-end time is lower than the sum of its decode and inference work timings, but they do not directly measure queue wait time, producer/consumer scheduling latency, or overlap intervals. Therefore, any overlap value derived as `decode_work_ms + inference_work_ms - total_ms` is only a timing proxy and is not reported as direct queueing evidence.

In this run, Method C did not consistently improve on Method B. At batch 32 it remained slower than B for every encoding/device combination: native CPU 45.35 vs 40.80 ms, native CUDA 42.62 vs 39.58 ms, JPEG CPU 63.65 vs 61.46 ms, and JPEG CUDA 58.93 vs 56.09 ms. This supports measuring queueing overhead rather than assuming that bounded prefetching improves throughput.

## Limitations

- The dataset is synthetic and the convolution model is untrained; there are no clinical or accuracy claims.
- Results are from one Kaggle machine and one T4 execution context, with four virtual CPUs reported by the runtime.
- Only `uniform:32` selection was used in this validation run; selection strategies and larger real-world studies remain open.
- The batch-size parameter is explored within the existing benchmark’s selected-frame pipeline; it is not a full production batching study across independent studies.
- CPU utilization is a core-equivalent timing measure, not a complete system utilization trace. GPU utilization and power/energy were not captured.
- Kaggle output is retained locally under the ignored `results/kaggle_validation2` path for audit and is intentionally not committed to Git.

## Reproduction record

The source changes for this run are prepared on the local branch `validation/medi-ai-exp-001-run-2`, which has been pushed to GitHub. Draft PR [#1](https://github.com/swapins/DICOM-Edge-Inference-Benchmark/pull/1) targets `main` and remains unmerged. The original pilot output directory remains separate. The generated validation archive remains under the ignored `results/kaggle_validation2` path.
