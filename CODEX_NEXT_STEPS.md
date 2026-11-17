# Local Codex prompt — Phase 2: Validate on real DICOM libraries / edge hardware

Copy the following into Codex after extracting this repository to `D:\OPEN_SOURCE\DICOM_EDGE_INFERENCE_BENCHMARK`.

> Work in `D:\OPEN_SOURCE\DICOM_EDGE_INFERENCE_BENCHMARK` as an **independent repository**. Do not change, commit to, or open PRs in the highdicom project. Existing `D:\OPEN_SOURCE\HIGHDICOM\UPSTREAM` is a dependency to be installed in editable mode at the accepted feature branch. Treat upstream PR #520 as independent.
>
> 1. Inspect scaffold, README, CLI, synthetic generator, three-mode runner, tests, and experiment protocol.
> 2. Create `.venv`; install package with `[inference,monitoring,test]`, then install highdicom issue #347 branch in editable mode. Verify installed highdicom file path and SHA; do not alter its branch.
> 3. Run `pytest -q`, then generate native and JPEG synthetic DICOM datasets. Fix compatibility issues only inside this repository.
> 4. Execute A/B/C on CPU using the highdicom backend for selections `first:1`, `first:4`, `uniform:16`, `uniform:64`, and `all`, with both native and JPEG datasets. Record actual file dimensions and versions. Run >=3 warmups and >=10 repeats once pilot passes.
> 5. Confirm all modes return matching decoded-frame SHA256. Fail any comparison that doesn't match.
> 6. Verify RSS sampling, Python peak allocation, CPU time, end-to-end wall time, and stage time interpretation. Do not invent GPU utilization or joule measurements.
> 7. If compatible CUDA hardware is available, verify PyTorch CUDA explicitly, repeat the three modes on CUDA, and capture separate device/runtime metadata. Do not silently substitute CPU for CUDA.
> 8. For Jetson, follow its exact JetPack + PyTorch compatibility constraints; do not install unsupported wheels. If a Jetson Nano cannot run the required Python/highdicom stack, mark it blocked instead of claiming success.
> 9. Add concise research results in a separate results report **without committing DICOMs or identifiers**; report invalid/missing measurements honestly. Results should be presented as experimental results, not validated clinical performance.
> 10. Do not add CUDA code or dependencies to highdicom PR #520. Publish this work in a separate repository only after the user approves the visibility and final files.


## Kaggle GPU baseline — run first

The GPU measurements will be performed on Kaggle GPU T4 x2, using CUDA device 0 only.
Do not call these Jetson results or GPU-accelerated DICOM decoding.
Use the included `notebooks/Kaggle_GPU_Benchmark.ipynb` and `docs/KAGGLE_PROTOCOL.md`.
Keep the Kaggle results separate from upstream PR #520.
After validated Kaggle results, hardware benchmarks on Jetson/Raspberry Pi remain a distinct phase.
