# Kaggle GPU experimental protocol — local-first deployment

The **canonical research code lives in the Windows VS Code folder**. Kaggle GPU execution is triggered via its CLI from the local VS Code terminal; the GitHub research repository can be created afterward. See [`LOCAL_VSCODE_KAGGLE_WORKFLOW.md`](LOCAL_VSCODE_KAGGLE_WORKFLOW.md).

1. Use `kaggle auth login` locally; never give the token to Codex or add it to Git.
2. Run `python scripts/prepare_kaggle_kernel.py --owner YOUR_KAGGLE_USERNAME --profile pilot` locally. This builds a self-contained Kaggle notebook from approved local code (no Dataset upload needed).
3. Run `kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4`, then `kaggle kernels status` and `kaggle kernels logs`. Kaggle will execute with GPU 0.
4. Download outputs using `kaggle kernels output YOUR_KAGGLE_USERNAME/dicom-edge-inference-benchmark -p results/kaggle`.
5. Validate pilot first; regenerate with `--profile full` for the full experiment. Changes in VS Code require regeneration and redeployment.
6. A/B/C results are cloud-based CPU decoding + CUDA inference only. Do not call these Jetson or GPU codec benchmarks.

## Experiment conditions

**A:** repeated `ImageFileReader.read_frame(index)` calls; **B:** selective `read_frames(indices=...)` in equal-sized chunks; **C:** same selective path with bounded CPU producer/consumer prefetch. All modes use the same seeded, *untrained* PyTorch CNN, selected frames, device, encoder and batch size. All datasets are 100% synthetic.

| Profile | Encodings | Devices | Selections | Warmups | Repeats |
| --- | --- | --- | --- | ---: | ---: |
| Pilot | Native + JPEG Baseline | CPU + CUDA | uniform:16 | 1 | 2 |
| Full | Native + JPEG Baseline | CPU + CUDA | first:1, uniform:4, uniform:16, uniform:64, all | 3 | 10 |

Input is 96 frames at 128 × 128 pixels by default. Matching SHA-256 decoded-frame hashes across modes are required. The three strategies are timed separately under the same runtime for each condition. Note that JPEG is lossy; only compare equivalent decoded frames **within the same compressed dataset**.

## Recorded measures and limitations

- End-to-end wall-clock milliseconds and frames/second.
- CPU process time and core-equivalent percent.
- `tracemalloc` Python allocation peak (not total RAM).
- `psutil` sampled process RSS peak (sampling may miss brief true peaks).
- PyTorch CUDA tensor memory peak (not total GPU allocation).
- Actual GPU names, count, CUDA runtime, PyTorch version and `nvidia-smi` inventory.

**Not measured:** GPU utilization vs time, CPU–GPU transfer breakdown, GPU idle time, power/energy, integrated device thermals or performance on an actual Jetson. This is a **Kaggle cloud-GPU baseline** for a later hardware-specific experiment, not proof of device-level edge optimization. An untrained model has no diagnostic accuracy. Do not publish clinical or general CUDA-decompression claims from these results.

## Baseline research provenance

- Independent research project, not part of [highdicom upstream PR #520](https://github.com/ImagingDataCommons/highdicom/pull/520).
- highdicom source fork: `swapins/highdicom`, pinned to SHA `d42f664006a9cd2194eda0bf5b1cf0fa2fb0188e`.
- Save the `machine.json`, per-trial results and environmental package versions with the experiment; use real Jetson/Raspberry Pi later for hardware conclusions.

## Verified pilot record

Kaggle kernel `StellarAISG/dicom-edge-inference-benchmark` version 4 completed successfully on 2026-11-17. The run used GPU 0 (Tesla T4, 15,360 MiB, compute capability 7.5), CUDA 13.0, PyTorch 2.11.0+cu130, and Python 3.13.16. The second available T4 was not used. Kaggle executed 19 tests successfully, then produced a valid 14-entry results ZIP containing machine metadata, 12 summary rows, and four per-condition trial pairs.

Median end-to-end latency and throughput were:

| Encoding / device | A | B | C |
| --- | --- | --- | --- |
| Native / CPU | 34.335 ms / 466.071 fps | 29.645 ms / 539.925 fps | 30.517 ms / 524.302 fps |
| Native / CUDA | 31.407 ms / 509.431 fps | 26.859 ms / 595.802 fps | 28.519 ms / 561.264 fps |
| JPEG / CPU | 36.941 ms / 433.250 fps | 32.229 ms / 496.517 fps | 36.806 ms / 436.490 fps |
| JPEG / CUDA | 34.356 ms / 465.872 fps | 31.296 ms / 511.599 fps | 33.150 ms / 482.680 fps |

The corresponding B/C end-to-end changes versus A were 13.66%/11.12% for native CPU, 14.48%/9.20% for native CUDA, 12.76%/0.37% for JPEG CPU, and 8.91%/3.51% for JPEG CUDA. Trial CSVs recorded decode and inference work separately; these phase times can overlap and must not be added to derive wall time. CPU core-equivalent usage was approximately 102% for A/B and 104% for C; sampled RSS was approximately 546 MiB for native CPU and 990–993 MiB for CUDA/JPEG strata; PyTorch CUDA tensor peak was 10.189 MiB. GPU utilization, power, energy, transfers, and thermals were not measured.
