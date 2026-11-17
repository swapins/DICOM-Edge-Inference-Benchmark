import numpy as np
import pytest

from dicom_edge_bench.inference import Inference


def test_no_inference_does_not_require_gpu():
    instance = Inference.create("none", "cpu", channels=1)
    assert instance.device == "none"
    assert instance.gpu_peak_mib() is None
    instance.process(np.zeros((8, 8), dtype=np.uint8))


def test_cuda_not_silently_used_for_none():
    with pytest.raises(ValueError, match="CUDA requires"):
        Inference.create("none", "cuda", channels=1)


def test_torch_cpu_pipeline():
    pytest.importorskip("torch")
    instance = Inference.create("torch", "cpu", channels=1, torch_threads=1)
    instance.process(np.zeros((16, 16), dtype=np.uint8))
    assert instance.device == "cpu"
