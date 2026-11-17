# Platform setup notes

## Windows development (CPU and optionally NVIDIA discrete GPU)

Install Python 3.10+, `pydicom`, a PyTorch build that matches the installed CUDA driver if CUDA is required, and the pinned highdicom feature commit. If using a local highdicom checkout, install it in editable mode from its actual path and verify `import highdicom; print(highdicom.__file__)` points to that checkout.

Use PowerShell script `scripts/run_windows_cpu.ps1`. If on a CUDA-capable desktop, change `--device cpu` to `--device cuda` after `torch.cuda.is_available()` succeeds.

## Jetson Nano / Orin

Jetson support is contingent on installed JetPack and its matching PyTorch/CUDA release. **Do not overwrite the JetPack-installed CUDA ecosystem with generic x86 wheels.** Follow NVIDIA's official PyTorch for Jetson matrix. The original Nano may require legacy JetPack and consequently legacy Python/PyTorch tooling incompatible with the highdicom branch's Python 3.10+ requirement; in that case, do **not** claim a validated Nano run. Use Orin or another compatible system instead, and separately document any later Nano adaptation.

NVIDIA's `tegrastats` reports GPU, CPU and memory utilization. `scripts/run_jetson.sh` can capture its raw log but does not calculate aligned per-mode power or joules/frame. See NVIDIA's tegrastats documentation.

## Raspberry Pi

Use native CPU inference or `--inference none`; Raspberry Pi has no NVIDIA CUDA device. Record Pi model, memory, CPU frequency, swap usage, temperature and OS. An external accelerator requires its own supported backend; **this project does not implement TPU drivers**.

## Source documentation

- pydicom frame iterator: https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.pixels.iter_pixels.html
- PyTorch CUDA memory APIs: https://docs.pytorch.org/docs/stable/notes/cuda.html
- NVIDIA Jetson PyTorch compatibility: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html
- NVIDIA tegrastats: https://docs.nvidia.com/jetson/archives/r36.2/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html
