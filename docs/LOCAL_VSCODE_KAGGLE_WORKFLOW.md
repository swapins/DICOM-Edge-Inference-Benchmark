# Local-first VS Code + Codex + Kaggle + GitHub workflow

**The Windows working tree is the only editable source of truth.** Kaggle executes a generated copy of local source; GitHub is created/pushed from VS Code after local and Kaggle validation. Nothing here changes highdicom PR #520.

## 1. Extract and open locally

Extract or clone the repository anywhere convenient. Ensure `pyproject.toml` sits directly inside the opened repository folder, not one nested folder deeper. Open that folder in VS Code, then open **Terminal > New Terminal** and select PowerShell. Ensure Python 3.10+, Git, GitHub CLI (`gh`), and the Codex extension/CLI are available.

> This research prototype is separate from any highdicom checkout. Do not modify highdicom PR #520 in this project.

## 2. Local environment (PowerShell)

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e ".[monitoring,test]" kaggle
& .\.venv\Scripts\python.exe -m pip install "git+https://github.com/swapins/highdicom.git@d42f664006a9cd2194eda0bf5b1cf0fa2fb0188e"
& .\.venv\Scripts\python.exe -m pytest -q
```

Check highdicom's current branch and SHA **without modifying it**:

```powershell
git -C <path-to-highdicom-checkout> status --short
git -C <path-to-highdicom-checkout> rev-parse HEAD
```

Expected API: `ImageFileReader.read_frames` exists. Expected tested SHA when scaffolded: `d42f664006a9cd2194eda0bf5b1cf0fa2fb0188e` (a newer vetted commit may be used and documented). If the highdicom checkout is absent, Codex must use the pinned commit from `swapins/highdicom` rather than invent another implementation.

CPU-only smoke test first (PyTorch is not required):

```powershell
& .\.venv\Scripts\python.exe -m dicom_edge_bench generate --out data\smoke.dcm --frames 12 --rows 32 --columns 32
& .\.venv\Scripts\python.exe -m dicom_edge_bench run --dicom data\smoke.dcm --backend highdicom --inference none --device cpu --selection uniform:8 --batch-size 5 --prefetch 2 --repeats 1 --warmups 0 --out results\local_smoke.json
```

CPU model inference is optional in the local environment. To run it, install a CPU-compatible PyTorch build (check Windows and Python version compatibility) and use `scripts\run_windows_cpu.ps1`. Kaggle handles the main CUDA test.

## 3. Connect Kaggle locally — never paste credentials into Codex

```powershell
& .\.venv\Scripts\python.exe -m pip install -U kaggle
& .\.venv\Scripts\kaggle.exe auth login
& .\.venv\Scripts\kaggle.exe kernels list -m
```

The official Kaggle CLI offers browser-based OAuth login. The browser is opened locally; tokens remain outside your repository. **Never put `kaggle.json`, API keys, OAuth tokens, or notebook secrets into Git, ChatGPT, or Codex prompts.** For remote kernels, your Kaggle account must have notebook GPU access/quota.

## 4. Generate a Kaggle notebook from local source

Set your public **Kaggle username** (not a token) and build the self-contained notebook:

```powershell
$KAGGLE_USER = "REPLACE_WITH_YOUR_KAGGLE_USERNAME"
& .\.venv\Scripts\python.exe scripts\prepare_kaggle_kernel.py --owner $KAGGLE_USER --profile pilot
```

This writes `kaggle\DICOM_Edge_Kaggle.ipynb` and `kaggle\kernel-metadata.json`. The notebook embeds only allowed source, tests and docs from your local checkout. It does **not** need you to upload a project ZIP as a dataset and does **not** need a GitHub repo to already exist. Its `kernel-metadata.json` requests a **private** T4-based kernel and Internet for Python dependencies.

## 5. Run Kaggle GPU remotely from VS Code

```powershell
& .\.venv\Scripts\kaggle.exe kernels push -p kaggle --accelerator NvidiaTeslaT4
& .\.venv\Scripts\kaggle.exe kernels status "$KAGGLE_USER/dicom-edge-inference-benchmark"
& .\.venv\Scripts\kaggle.exe kernels logs "$KAGGLE_USER/dicom-edge-inference-benchmark"
```

The CLI uploads a new notebook version and executes it on Kaggle; it does **not** run a CUDA GPU on Windows. Wait until the **Kaggle** job is complete (do not mistake upload success for benchmark success), then retrieve outputs:

```powershell
New-Item -Type Directory -Force results\kaggle | Out-Null
& .\.venv\Scripts\kaggle.exe kernels output "$KAGGLE_USER/dicom-edge-inference-benchmark" -p results\kaggle
```

Open `results\kaggle` in VS Code to examine the output ZIP and machine/summary files. The notebook creates `/kaggle/working/DICOM_Edge_Kaggle_Results.zip`. If the run errors, use `kaggle kernels logs` and fix the **local** source, regenerate, then push again. Never edit the generated Kaggle notebook as the canonical source.

For the full research run, change `--profile pilot` to `--profile full`, regenerate and push. The full profile uses a larger number of trials and may consume Kaggle GPU quota. Only GPU 0 is used; cloud T4 benchmarks are not Jetson measurements.

**Important:** Kaggle's source is a snapshot of your local code at regeneration time. After editing `src/` or `scripts/run_kaggle.py`, always regenerate before pushing Kaggle.

## 6. Create and push the separate GitHub repository from Windows

Run in the VS Code terminal *after local tests and the Kaggle pilot are checked*. Keep it **private first**; change to public only deliberately.

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
gh auth login
git init -b main
git add .
git status --short
# Inspect the staged diff: no generated DICOM, data, results, secrets, or notebooks with embedded bundles.
git diff --cached --stat
git commit -m "research: initial DICOM edge inference benchmark"
gh repo create swapins/DICOM-Edge-Inference-Benchmark --private --source . --remote origin --push
```

If an existing `.git` directory or remote exists, Codex must inspect it instead of resetting or overwriting it. Check repo creation with `gh repo view swapins/DICOM-Edge-Inference-Benchmark` and `git ls-remote origin HEAD`.

Do not commit raw benchmark files or synthetic DICOM; write a sanitized, reproducible analysis report if useful. Never claim clinical validity, direct GPU DICOM decoding, on-device Jetson measurements, or energy savings without direct evidence.

## 7. Git vs Kaggle responsibilities

| System | Purpose |
|---|---|
| VS Code / Codex on Windows | Canonical editable source, tests, review, Git commits |
| Kaggle | Temporary remote T4 CUDA compute, controlled A/B/C benchmark runs |
| GitHub (your new repo) | Versioned research code and sanitized documentation |
| highdicom PR #520 | Independent upstream contribution; no research benchmark files added |

Official CLI documentation: https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md and https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md
