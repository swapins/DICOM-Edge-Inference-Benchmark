"""A fixed, untrained convolution model used only for pipeline timing.

THIS MODEL IS NOT A DIAGNOSTIC MODEL. Outputs have no clinical meaning.
"""
from dataclasses import dataclass
import numpy as np

from .profiling import ProfileTrace


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

    def gpu_memory_mib(self) -> dict[str, float | None]:
        if self.device != "cuda":
            return {"allocated": None, "reserved": None}
        return {
            "allocated": self.torch.cuda.memory_allocated() / (1024 * 1024),
            "reserved": self.torch.cuda.memory_reserved() / (1024 * 1024),
        }

    def process(self, image: np.ndarray, *, trace: ProfileTrace | None = None,
                frame_id: int | None = None) -> None:
        if self.model is None:
            return
        torch = self.torch
        phase_fields = {} if frame_id is None else {"frame_id": frame_id}
        phase = trace.phase if trace is not None else _null_phase
        with phase("preprocess", **phase_fields):
            image = np.ascontiguousarray(image)
            if image.ndim == 2:
                chw = image[np.newaxis, ...]
            elif image.ndim == 3 and image.shape[-1] in (1, 3):
                chw = np.moveaxis(image, -1, 0).copy()
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")
        with phase("batch_preparation", **phase_fields):
            inp = torch.from_numpy(np.ascontiguousarray(chw)).unsqueeze(0)
        with phase("host_to_device", **phase_fields):
            inp = inp.to(dtype=torch.float32).mul_(1.0 / 255.0).to(self.device)
        cuda_start = cuda_end = None
        if trace is not None and trace.detailed_torch and self.device == "cuda":
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            cuda_start.record()
        with phase("inference", **phase_fields), torch.inference_mode():
            _ = self.model(inp)
        with phase("gpu_synchronization", **phase_fields):
            self.synchronize()  # Accurate CUDA wall timing including device execution
        if cuda_start is not None and cuda_end is not None:
            cuda_end.record()
            self.synchronize()
            trace.add_cuda_event("inference", cuda_start.elapsed_time(cuda_end))


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _null_phase(*args, **kwargs):
    return _NullContext()
