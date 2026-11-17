# Generated Kaggle kernel

This folder contains generated metadata and a self-contained notebook made from the **local** working tree.

Run from the repository root:

```powershell
python scripts/prepare_kaggle_kernel.py --owner YOUR_KAGGLE_USERNAME --profile pilot
kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4
kaggle kernels status YOUR_KAGGLE_USERNAME/dicom-edge-inference-benchmark
kaggle kernels output YOUR_KAGGLE_USERNAME/dicom-edge-inference-benchmark -p results/kaggle
```

Do not add your token to the project. Log in locally via `kaggle auth login`.
Generated notebook and metadata are ignored by Git; this README is versioned.
