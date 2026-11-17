import pytest

pydicom = pytest.importorskip("pydicom")
from dicom_edge_bench.synthetic import create_dicom
from dicom_edge_bench.runner import benchmark, save_result


def test_three_mode_decode_only(tmp_path):
    p = create_dicom(tmp_path / "input.dcm", frames=12, rows=16, columns=16)
    result = benchmark(
        p, backend="pydicom", inference_type="none", device="cpu",
        selection="uniform:8", batch_size=5, prefetch=2,
        repeats=1, warmups=0,
    )
    assert result["metadata"]["identical_decoded_frames_across_all_trials"]
    assert {t["mode"] for t in result["trials"]} == {"A", "B", "C"}
    assert all(t["frame_count"] == 8 for t in result["trials"])
    saved = save_result(result, tmp_path / "report.json")
    assert saved.exists() and saved.with_suffix(".csv").exists()


def test_torch_cpu_dummy_inference(tmp_path):
    pytest.importorskip("torch")
    p = create_dicom(tmp_path / "input.dcm", frames=6, rows=16, columns=16)
    result = benchmark(
        p, backend="pydicom", inference_type="torch", device="cpu",
        selection="all", batch_size=5, prefetch=2,
        repeats=1, warmups=0,
    )
    assert result["metadata"]["device"] == "cpu"
    assert result["metadata"]["identical_decoded_frames_across_all_trials"]
