"""A fixed, untrained convolution model used only for pipeline timing.

THIS MODEL IS NOT A DIAGNOSTIC MODEL. Outputs have no clinical meaning.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Inference:
    model: object | None
    device: str
    torch: object | None

    @classmethod
    def create(cls, inference: str, requested_device: str, channels: int,
               torch_threads: int = 1) -> "Inference":
        if inference == "none":
            if requested_device == "cuda":
                raise ValueError("CUDA requires --inference torch")
            return cls(None, "none", None)
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Install PyTorch for --inference torch") from exc
        torch.set_num_threads(torch_threads)
        if requested_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = requested_device
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available in this PyTorch build")
        torch.manual_seed(347)
        model = torch.nn.Sequential(
            torch.nn.Conv2d(channels, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(8, 2),
        ).to(device).eval()
        return cls(model, device, torch)

    def synchronize(self) -> None:
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def reset_gpu_peak(self) -> None:
        if self.device == "cuda":
            self.torch.cuda.reset_peak_memory_stats()

    def gpu_peak_mib(self) -> float | None:
        if self.device != "cuda":
            return None
        return self.torch.cuda.max_memory_allocated() / (1024 * 1024)

    def process(self, image: np.ndarray) -> None:
        if self.model is None:
            return
        torch = self.torch
        image = np.ascontiguousarray(image)
        if image.ndim == 2:
            chw = image[np.newaxis, ...]
        elif image.ndim == 3 and image.shape[-1] in (1, 3):
            chw = np.moveaxis(image, -1, 0).copy()
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
        inp = torch.from_numpy(np.ascontiguousarray(chw)).unsqueeze(0)
        inp = inp.to(dtype=torch.float32).mul_(1.0 / 255.0).to(self.device)
        with torch.inference_mode():
            _ = self.model(inp)
        self.synchronize()  # Accurate CUDA wall timing including device execution
