import numpy as np
import pytest
from dicom_edge_bench.synthetic import create_dicom

pydicom = pytest.importorskip("pydicom")


def test_synthetic_native_deterministic(tmp_path):
    p1 = create_dicom(tmp_path / "a.dcm", frames=8, rows=12, columns=10)
    p2 = create_dicom(tmp_path / "b.dcm", frames=8, rows=12, columns=10)
    a = pydicom.dcmread(p1)
    b = pydicom.dcmread(p2)
    assert a.PatientID == "SYNTHETIC-ONLY"
    assert a.NumberOfFrames == 8
    assert a.PixelData == b.PixelData
    assert a.pixel_array.shape == (8, 12, 10)


def test_synthetic_jpeg(tmp_path):
    pytest.importorskip("PIL")
    p = create_dicom(tmp_path / "compressed.dcm", frames=6,
                     rows=16, columns=16, encoding="jpeg")
    ds = pydicom.dcmread(p)
    assert int(ds.NumberOfFrames) == 6
    assert ds.file_meta.TransferSyntaxUID.is_encapsulated
