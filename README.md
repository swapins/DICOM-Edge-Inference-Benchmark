# DICOM Edge Inference Benchmark

**Separate research prototype** for measuring the effect of selective DICOM frame decoding on CPU- and CUDA-based inference pipelines. This is **not** part of the upstream [highdicom PR #520](https://github.com/ImagingDataCommons/highdicom/pull/520), and introduces **no upstream code changes**.

> **Research-only.** The model is an **untrained synthetic CNN**, not a medical diagnostic model. Never put real patient DICOM files in this repository. The generated DICOM inputs are synthetic and do not represent clinical imagery.

## Research question

Does selective CPU-side decoding, with and without bounded prefetch, improve end-to-end inference throughput and resource usage on edge hardware relative to repeated single-frame retrieval?

The benchmark compares three implementations using the **same frame indices, decoding backend, batch size, and model**:

- **A — Repeated access**: `ImageFileReader.read_frame(i)` for every selected frame (or repeated pydicom `iter_pixels(...indices=[i])` for the pydicom backend).
- **B — Selective decoding**: `ImageFileReader.read_frames(indices=batch)` over bounded chunks.
- **C — Selective + bounded prefetch**: same chunks as B, decoded in one worker thread into a bounded queue while inference processes preceding frames.

B and C are configured with `--batch-size`; C adds `--prefetch` frames. The API's `read_frames()` returns a list per batch, so there can be up to a batch of decoded frames in the producer **plus** the queue of prefetched frames. That is **not** a fixed constant memory bound independent of frame dimensions.

## Local-first VS Code → Kaggle GPU → GitHub (recommended workflow)

**Keep `D:\OPEN_SOURCE\DICOM_EDGE_INFERENCE_BENCHMARK` as the canonical local working directory.** Open it in VS Code and use Codex for local engineering and tests. Then authenticate Kaggle through the local terminal, deploy a generated notebook remotely, retrieve GPU results into `results/`, and create/push a **separate GitHub repository from the local working tree**. Do not make edits in Kaggle the canonical source or modify highdicom PR #520.

The complete instructions are in [`docs/LOCAL_VSCODE_KAGGLE_WORKFLOW.md`](docs/LOCAL_VSCODE_KAGGLE_WORKFLOW.md). Paste [`CODEX_LOCAL_PROJECT_PROMPT.md`](CODEX_LOCAL_PROJECT_PROMPT.md) into Codex after extracting this project ZIP to Windows.

```powershell
# From VS Code PowerShell with the project's .venv activated:
kaggle auth login
python scripts/prepare_kaggle_kernel.py --owner YOUR_KAGGLE_USERNAME --profile pilot
kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4
kaggle kernels status YOUR_KAGGLE_USERNAME/dicom-edge-inference-benchmark
kaggle kernels output YOUR_KAGGLE_USERNAME/dicom-edge-inference-benchmark -p results/kaggle
```

The locally generated Kaggle notebook embeds a small, explicit allowlist of research source files and requires **no separate Kaggle Dataset upload or pre-existing GitHub repo**. Kaggle runs a remote GPU job, not CUDA on your Windows computer. The notebook deliberately uses one GPU, and all DICOM decoding remains CPU-side.

Do not put Kaggle credentials in prompts or the project. Start Git history and create the remote GitHub repository **from VS Code only after validation**, private by default. See the workflow document for verified commands and checks.

## Quick start — Windows / local CPU

Use Python 3.10+ and a **fresh virtual environment**. From this repository root:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[inference,monitoring,test]"
# Install the tested highdicom feature branch into the SAME environment:
python -m pip install -e "D:\OPEN_SOURCE\HIGHDICOM\UPSTREAM"
python -m dicom_edge_bench generate --out data/synthetic_native.dcm --frames 96 --rows 128 --columns 128
python -m dicom_edge_bench run --dicom data/synthetic_native.dcm --backend highdicom --inference torch --device cpu --selection uniform:16 --batch-size 16 --prefetch 2 --repeats 3 --warmups 1 --out results/windows_cpu.json
python -m pytest -q
```

The highdicom source checkout must be on branch `feat/issue-347-selective-multiframe-decoding`, with commit `d42f664006a9cd2194eda0bf5b1cf0fa2fb0188e` or a later validated commit. The benchmark checks that `ImageFileReader.read_frames()` exists; it does **not** automatically check the branch's exact commit hash.

### Pure pydicom / decoding-only smoke test

No torch installation needed for this mode:

```bash
python -m pip install -e ".[test,monitoring]"
python -m dicom_edge_bench generate --out data/smoke.dcm --frames 12 --rows 32 --columns 32
python -m dicom_edge_bench run --dicom data/smoke.dcm --backend pydicom --inference none --device cpu --selection all --batch-size 5 --prefetch 2 --repeats 1 --warmups 0 --out results/pydicom_smoke.json
```

**The pydicom backend is a harness smoke test, not evidence about highdicom's performance.** To evaluate PR #520, use `--backend highdicom`.

## CUDA / Jetson

On a **supported Jetson + JetPack build**, install an NVIDIA-supported PyTorch build compatible with that JetPack release. Do **not** assume that a generic `pip install torch` provides CUDA on Jetson. See [NVIDIA's Jetson PyTorch compatibility guidance](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html).

Once `python -c "import torch; assert torch.cuda.is_available()"` succeeds:

```bash
python -m pip install -e '.[monitoring,test]'
# Install highdicom issue #347 branch separately, with dependencies.
bash scripts/run_jetson.sh
```

CUDA is used **only for the downstream PyTorch inference stage**. All DICOM decoding remains CPU-side. The prototype does not implement CUDA codecs, zero-copy DICOM-to-CUDA transfer, GPU idle-time measurement, or TensorRT.

On Raspberry Pi, use a supported CPU PyTorch installation and `bash scripts/run_raspberry_pi.sh`, or run `--inference none` if PyTorch is unavailable.

## CLI reference

```text
python -m dicom_edge_bench generate --out data/image.dcm --encoding native|jpeg --frames 96 --rows 128 --columns 128 --seed 347
python -m dicom_edge_bench run --dicom data/image.dcm --backend highdicom|pydicom --inference torch|none --device auto|cpu|cuda --selection uniform:16 --batch-size 16 --prefetch 2 --repeats 3 --warmups 1 --out results/run.json
```

Valid selections: `all`, `first:16`, `uniform:16`, `0:96:4`, `0,4,8,12`. Indices are zero-based; order and duplicates are preserved.

## Output / interpretation

The CLI writes **JSON** with metadata, summary medians, and per-run trials, alongside a flat **CSV** with timing and memory fields. Output stays under `results/`, which is gitignored. `data/` is also gitignored. The report hashes the decoded pixel content to verify that all modes retrieved **identical decoded frames**. It exits with an error when hashes differ.

- `total_ms`: actual elapsed wall time, including decoding, inference, and scheduling.
- `decode_work_ms` and `inference_work_ms`: phase timing; **can overlap in C**, so do not add them to estimate wall time.
- `cpu_core_equivalent_percent`: total process CPU time divided by wall time; multicore workloads may exceed 100%.
- `python_alloc_peak_mib`: `tracemalloc` Python allocation peak, **not** total RAM.
- `sampled_process_rss_peak_mib`: `psutil`-sampled RSS (if installed), **not** a guaranteed true peak.
- `torch_gpu_tensor_peak_mib`: max PyTorch-allocated CUDA tensor memory, **not** total GPU or shared-memory consumption.
- `frames_per_second`: throughput of the **whole pipeline**, not just decoding.

GPU occupancy, end-to-end energy, Jetson power draw, and GPU idle time are **not** reported by this first version. Jetson's `tegrastats` can produce an external raw log, but this prototype does not align it to trial boundaries or compute joules/frame. Do not infer energy efficiency from lower runtime alone.

### Fairness limits

- A/B/C process the same indices and run the same untrained model. B/C use identical chunk size.
- Warmups are excluded; measurement order is randomized in a reproducible manner for each repeat.
- `tracemalloc` can materially affect timings; keep settings consistent when comparing configurations.
- CPU results, CUDA results and different datasets are **separate experimental strata**; do not conflate them.
- JPEG compression varies decoding workload relative to native DICOM. Run both.
- C may be slower if the CPU is saturated, the GIL dominates, or the inference step is too short to overlap meaningfully.
- The highdicom fallback for small selections is preserved; don't claim that every selection is faster.

See [Experiment Protocol](docs/EXPERIMENT_PROTOCOL.md) and [Platform Notes](docs/PLATFORM_NOTES.md).

### Verified Kaggle pilot

The pilot completed on 2026-11-17 in Kaggle's cloud environment using GPU 0, a Tesla T4 (CUDA 13.0, PyTorch 2.11.0+cu130). It used 96 synthetic 128x128-frame inputs, `uniform:16` selection, one warmup, two repeats, and the highdicom revision `d42f664006a9cd2194eda0bf5b1cf0fa2fb0188e`. All 12 A/B/C summaries reported matching decoded-frame hashes and finite measurements.

| Encoding / device | A total ms | B total ms | C total ms | B vs A | C vs A |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native / CPU | 34.335 | 29.645 | 30.517 | 13.66% faster | 11.12% faster |
| Native / CUDA | 31.407 | 26.859 | 28.519 | 14.48% faster | 9.20% faster |
| JPEG / CPU | 36.941 | 32.229 | 36.806 | 12.76% faster | 0.37% faster |
| JPEG / CUDA | 34.356 | 31.296 | 33.150 | 8.91% faster | 3.51% faster |

These are measured Kaggle T4 reference results, not Jetson or Raspberry Pi results. GPU utilization, power, energy, and thermals were not measured; CPU decoding remains the bottleneck under this design. Raw outputs remain in the gitignored `results/` directory.

## Publication / upstream separation

This repository should live at `swapins/DICOM-Edge-Inference-Benchmark`. It is **independent research**, not a PR to highdicom. Keep generated DICOMs, clinical data, `.venv`, and benchmark outputs out of Git. A prepared GitHub CLI publishing script is provided at `scripts/publish_to_github.ps1`. Review it before execution.

## License

MIT. No clinical or device certification claims.
