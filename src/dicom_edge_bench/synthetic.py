"""Generate deterministic, synthetic, de-identified multi-frame DICOM inputs."""
from pathlib import Path

import numpy as np


def create_dicom(
    path: str | Path,
    *,
    frames: int = 96,
    rows: int = 128,
    columns: int = 128,
    encoding: str = "native",
    seed: int = 347,
) -> Path:
    """Write synthetic 8-bit MONOCHROME2 DICOM (native or JPEG Baseline).

    JPEG mode uses Pillow and encapsulation, not any patient imagery.
    """
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import (
        ExplicitVRLittleEndian,
        JPEGBaseline8Bit,
        SecondaryCaptureImageStorage,
        generate_uid,
    )

    if frames < 1 or rows < 8 or columns < 8:
        raise ValueError("frames >= 1 and rows/columns >= 8 are required")
    if encoding not in ("native", "jpeg"):
        raise ValueError("encoding must be native or jpeg")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    images = rng.integers(0, 256, size=(frames, rows, columns), dtype=np.uint8)

    def uid(scope: str) -> str:
        return generate_uid(entropy_srcs=["dicom-edge-benchmark", scope, str(seed)])

    meta = FileMetaDataset()
    meta.TransferSyntaxUID = (
        ExplicitVRLittleEndian if encoding == "native" else JPEGBaseline8Bit
    )
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = uid("instance")
    meta.ImplementationClassUID = uid("implementation")
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = uid("study")
    ds.SeriesInstanceUID = uid("series")
    ds.Modality = "OT"
    ds.PatientName = "SYNTHETIC^BENCHMARK"
    ds.PatientID = "SYNTHETIC-ONLY"
    ds.Rows, ds.Columns = rows, columns
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.NumberOfFrames = frames
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.ImageType = ["DERIVED", "SECONDARY"]

    if encoding == "native":
        ds.PixelData = images.tobytes(order="C")
    else:
        from io import BytesIO
        from PIL import Image
        from pydicom.encaps import encapsulate

        codestreams = []
        for frame in images:
            buffer = BytesIO()
            Image.fromarray(frame, mode="L").save(buffer, "JPEG", quality=95)
            codestreams.append(buffer.getvalue())
        ds.PixelData = encapsulate(codestreams, has_bot=True)
        ds[0x7FE00010].is_undefined_length = True

    ds.save_as(path, enforce_file_format=True)
    return path
