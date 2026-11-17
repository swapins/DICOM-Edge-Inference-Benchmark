# Codex prompt: local-first project + Kaggle remote GPU + GitHub from Windows

You are acting as a senior Python research engineer in **VS Code on my Windows PC**.

## Ownership and folders

- Work only in the opened repository workspace.
- This project is independent of any highdicom checkout and upstream highdicom PR #520. Treat highdicom solely as a read-only package dependency; do not modify, push or rebase it.
- This folder contains a supplied scaffold with `src/dicom_edge_bench`, `tests`, `notebooks`, `scripts`, and `docs`. Preserve it and develop it in place. Do not clone highdicom again or create duplicate nested projects.
- Main purpose: benchmark CPU DICOM decoding + PyTorch CUDA inference on Kaggle, using the locally authored sources.

## Phase 1: inspect and run locally

1. Inspect git status, paths, README, `docs/LOCAL_VSCODE_KAGGLE_WORKFLOW.md`, experiment protocol, existing code, tests and `.gitignore`. Do not overwrite user work.
2. Create a compatible `.venv` if missing and install the package plus test/monitoring dependencies. Prefer the pinned contribution commit from `https://github.com/swapins/highdicom`; use a local checkout only when its path and commit have been explicitly verified.
3. Run local `pytest -q` and a synthetic-DICOM CPU smoke test with `--inference none`, then optional CPU PyTorch if feasible. Fail on decoded-frame checksum mismatch.
4. Fix only issues inside this research repository. Use seeded synthetic DICOM; do not download patient data.

## Phase 2: connect Kaggle from the local terminal

5. Verify Kaggle CLI availability, install/update into `.venv` as needed, then invoke `kaggle auth login` in the **user's terminal**. Browser-based OAuth only; the user completes sign-in. Never request, print, read, copy, log, upload or commit API tokens. Do not access local credential files in Codex.
6. Validate authentication via `kaggle kernels list -m`; if unavailable, report the specific blocker. Retrieve or ask for the **non-secret Kaggle username** only if not known from an authenticated profile.
7. Run `python scripts/prepare_kaggle_kernel.py --owner <kaggle-username> --profile pilot`. Verify generated metadata, private visibility, code_file reference, GPU `NvidiaTeslaT4`, and embedded project manifest.
8. From the VS Code terminal, execute `kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4` to launch the Kaggle GPU pilot (this remote execution is authorized). Check `kaggle kernels status` and `kaggle kernels logs`; do not imply that the GPU executed until the run completed. Handle Kaggle GPU quota, Internet settings and credentials errors explicitly.
9. Download results via `kaggle kernels output <owner>/dicom-edge-inference-benchmark -p results/kaggle`; verify output artifacts, actual `torch.cuda.is_available()`, GPU name, highdicom revision, selection hashes, timing and separate CPU/CUDA strata.
10. Only after pilot success, run a full profile if quota and system limits permit; retain all raw outputs locally under ignored `results/`. No fake or extrapolated Jetson numbers.

## Phase 3: create GitHub project locally and push

11. Review the `.gitignore` and all staged files for PHI, `.venv`, tokens, `.env`, generated Kaggle notebook, images/DICOMs and raw results. Add an honest research README/report including test status and platform details.
12. Authenticate GitHub CLI with `gh auth login` locally if needed; never ask for tokens in chat.
13. Create a NEW local git repository only if `.git` is absent, using `git init -b main`; commit source, tests, notebook template and docs. **Do not touch any other repository.**
14. From THIS local working tree only, create `swapins/DICOM-Edge-Inference-Benchmark` as a **private** GitHub repo initially and push using `gh repo create ... --private --source . --remote origin --push`. If a repo of this name already exists, stop and inspect it, do not overwrite it or assume ownership. Do not publish publicly without separate user instruction.
15. Verify remote URL, default branch, pushed commit SHA and clean working tree. Do not claim success if the push fails.

## Experimental integrity

- A: repeated per-frame decoding + identical inference model.
- B: selective decoding in bounded batches + same inference model.
- C: bounded prefetch overlapped with downstream inference.
- Use identical synthetic DICOM data, selections, seeds, model and inference device for each within-stratum comparison.
- GPU is for downstream PyTorch inference only. DICOM decode stays CPU-side. Kaggle T4 ≠ Jetson GPU and no CUDA decoder is introduced.
- Distinguish measured RSS vs `tracemalloc` vs GPU tensor allocation.
- Do not claim clinical utility, novel scheduling mechanisms or energy savings from proxy benchmarks.

## Final response expected

Local folder, Python/highdicom SHAs, CPU test results, Kaggle username (non-secret), Kaggle kernel URL and actual run status, GPU model, artifact locations, A/B/C comparison, validation limitations, GitHub repository URL, remote branch and commit SHA. Report all failures clearly.

**Proceed with local work autonomously; pause only for mandatory interactive OAuth/login prompts, missing user-specific username, or a true safety/security blocker.**
