"""Exercise real A/B/C scheduling without requiring external DICOM dependencies."""
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
import sys
import numpy as np

import dicom_edge_bench.runner as runner


def test_abc_orchestration_and_json_csv(tmp_path, monkeypatch):
    """This proves orchestration, NOT actual pydicom/highdicom correctness."""
    dummy_file = tmp_path / "input.dcm"
    dummy_file.write_bytes(b"synthetic-placeholder")

    class Metadata:
        NumberOfFrames = 12
        SamplesPerPixel = 1
        file_meta = SimpleNamespace(TransferSyntaxUID="1.2.840.10008.1.2.1")

        def get(self, name, default=None):
            return "SYNTHETIC-ONLY" if name == "PatientID" else default

    fake_module = ModuleType("pydicom")
    fake_module.__version__ = "fake-for-unit-test"
    fake_module.dcmread = lambda *_args, **_kwargs: Metadata()
    monkeypatch.setitem(sys.modules, "pydicom", fake_module)

    @contextmanager
    def fake_decoder(_path, indices, mode, backend, chunk_size):
        assert mode in ("A", "B")
        assert backend == "pydicom"
        assert chunk_size == 4
        yield (np.full((8, 8), i, dtype=np.uint8) for i in indices)

    monkeypatch.setattr(runner, "decoded_frames", fake_decoder)
    result = runner.benchmark(
        dummy_file, backend="pydicom", inference_type="none", device="cpu",
        selection="0:12:2", batch_size=4, prefetch=2,
        repeats=2, warmups=0,
    )
    assert len(result["trials"]) == 6
    assert result["metadata"]["identical_decoded_frames_across_all_trials"]
    assert {r["mode"] for r in result["trials"]} == {"A", "B", "C"}
    result_path = runner.save_result(result, tmp_path / "report.json")
    assert result_path.exists()
    assert result_path.with_suffix(".csv").exists()
