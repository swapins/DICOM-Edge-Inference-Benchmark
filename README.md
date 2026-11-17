# DICOM Edge Inference Benchmark

This repository is an independent research prototype for measuring selective DICOM frame decoding in CPU and CUDA inference pipelines. It is not part of [highdicom PR #520](https://github.com/ImagingDataCommons/highdicom/pull/520), and it does not modify the upstream highdicom repository.

> Research-only software. The benchmark uses an untrained synthetic CNN and synthetic DICOM inputs. It is not a medical diagnostic system. Do not place patient data, credentials, or other sensitive material in this repository.

## What is measured

The benchmark compares three strategies using the same selected frames, decoder, batch size, and model:

- **A - Repeated access:** retrieve each selected frame individually with `read_frame(i)`.
- **B - Selective decoding:** retrieve selected frames in bounded `read_frames(indices=batch)` chunks.
- **C - Selective decoding with prefetch:** use the same chunks with a bounded producer queue while inference processes earlier chunks.

The `highdicom` backend evaluates the selective multi-frame API. The `pydicom` backend is provided for a decoding-only harness smoke test and is not evidence about highdicom performance.

The benchmark reports end-to-end wall time, phase timings, throughput, CPU time, sampled process RSS, Python allocation peaks, CUDA tensor memory, and decoded-frame hashes. Decode and inference phase timings can overlap; do not add them to estimate wall time.

## Evidence and scope

The first verified pilot ran on Kaggle using GPU 0, a Tesla T4, CUDA 13.0, and PyTorch 2.11.0. It used 96 synthetic 128x128 frames, `uniform:16` selection, one warmup, and two measured repeats for native and JPEG inputs on CPU and CUDA.

| Encoding / device | A total ms | B total ms | C total ms | B vs A | C vs A |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native / CPU | 34.335 | 29.645 | 30.517 | 13.66% faster | 11.12% faster |
| Native / CUDA | 31.407 | 26.859 | 28.519 | 14.48% faster | 9.20% faster |
| JPEG / CPU | 36.941 | 32.229 | 36.806 | 12.76% faster | 0.37% faster |
| JPEG / CUDA | 34.356 | 31.296 | 33.150 | 8.91% faster | 3.51% faster |

These are measured Kaggle T4 reference results, not Jetson or Raspberry Pi results. DICOM decoding remains CPU-side. GPU utilization, power, energy, thermals, and CPU-to-GPU transfer time were not measured. The highdicom source used remotely was pinned to `d42f664006a9cd2194eda0bf5b1cf0fa2fb0188e`.

## Local setup on Windows

Run the commands below from the repository root in PowerShell. Use a real Python 3.10+ interpreter. If `python` resolves to a Microsoft Store alias, replace it with the absolute path to an installed interpreter.

```powershell
python --version
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
& .\.venv\Scripts\python.exe -m pip install -e ".[monitoring,test]"
& .\.venv\Scripts\python.exe -m pip install -e "D:\OPEN_SOURCE\HIGHDICOM\UPSTREAM"
```

The highdicom checkout should be on `feat/issue-347-selective-multiframe-decoding` at the validated commit above, or at a later validated revision. Do not reset, modify, or push that checkout from this project.

Run the local test suite:

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
```

Expected result for the current project snapshot: `17 passed, 2 skipped`.

## CPU smoke benchmark

This smoke test uses synthetic data and does not require CUDA or PyTorch:

```powershell
New-Item -ItemType Directory -Force data,results | Out-Null
& .\.venv\Scripts\python.exe -m dicom_edge_bench generate `
  --out data\smoke.dcm --frames 12 --rows 32 --columns 32
& .\.venv\Scripts\python.exe -m dicom_edge_bench run `
  --dicom data\smoke.dcm --backend highdicom --inference none `
  --device cpu --selection uniform:8 --batch-size 5 --prefetch 2 `
  --repeats 1 --warmups 0 --out results\local_smoke.json
```

Generated DICOM files and benchmark results are ignored by Git. Never use real patient data for this smoke test.

## Kaggle GPU workflow

Kaggle is used only for remote GPU execution. The local Windows checkout remains the source of truth.

1. Authenticate locally without sharing credentials:

   ```powershell
   & .\.venv\Scripts\python.exe -m pip install --upgrade kaggle six
   & .\.venv\Scripts\kaggle.exe auth login
   ```

2. Generate a private pilot notebook from the local source. Replace the placeholder with your public Kaggle username only:

   ```powershell
   & .\.venv\Scripts\python.exe scripts\prepare_kaggle_kernel.py `
     --owner YOUR_KAGGLE_USERNAME --profile pilot
   ```

3. Upload and monitor the run:

   ```powershell
   & .\.venv\Scripts\kaggle.exe kernels push -p kaggle --accelerator NvidiaTeslaT4
   & .\.venv\Scripts\kaggle.exe kernels status `
     "YOUR_KAGGLE_USERNAME/dicom-edge-inference-benchmark"
   & .\.venv\Scripts\kaggle.exe kernels logs `
     "YOUR_KAGGLE_USERNAME/dicom-edge-inference-benchmark"
   ```

4. Download outputs only after Kaggle reports completion:

   ```powershell
   New-Item -ItemType Directory -Force results\kaggle | Out-Null
   & .\.venv\Scripts\kaggle.exe kernels output `
     "YOUR_KAGGLE_USERNAME/dicom-edge-inference-benchmark" -p results\kaggle
   ```

Never add `kaggle.json`, API keys, OAuth tokens, or access tokens to the repository. Never claim a GPU experiment succeeded based only on upload success.

See [the local workflow](docs/LOCAL_VSCODE_KAGGLE_WORKFLOW.md), [the Kaggle protocol](docs/KAGGLE_PROTOCOL.md), and [the experiment protocol](docs/EXPERIMENT_PROTOCOL.md) for the full reproducibility checklist.

## Jetson and Raspberry Pi follow-up

The Kaggle pilot is a cloud reference experiment. It does not measure Jetson or Raspberry Pi performance. For a hardware claim, repeat the same matrix on the target device with a supported PyTorch build and record device-specific utilization, power, thermals, and timing data. The provided scripts are starting points:

- Jetson: `scripts/run_jetson.sh`
- Raspberry Pi: `scripts/run_raspberry_pi.sh`
- Windows CPU: `scripts/run_windows_cpu.ps1`

Do not install CUDA Toolkit or GPU PyTorch on Windows solely for the Kaggle workflow.

## Repository hygiene

The following remain outside version control by design:

- `.venv/`, caches, and build metadata
- generated DICOM files and benchmark results
- Kaggle credential files and tokens
- generated Kaggle notebook metadata and notebook outputs

This project is maintained separately from highdicom PR #520. The repository is published privately at [swapins/DICOM-Edge-Inference-Benchmark](https://github.com/swapins/DICOM-Edge-Inference-Benchmark).

## License

MIT. See [LICENSE](LICENSE). No clinical, safety, or device-certification claims are made.
