"""Controlled decoding adapters for highdicom and pydicom backends."""
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
import numpy as np

from .selection import chunks


@contextmanager
def decoded_frames(path: Path, indices: list[int], mode: str,
                   backend: str, chunk_size: int) -> Iterator[Iterator[np.ndarray]]:
    """Yield a frame iterator whose file-reader lifetime is scoped to the context.

    A = repeated frame access; B/C = batched selective access.
    B and C use the SAME decoder and chunk size; C differs in scheduling only.
    """
    if backend == "highdicom":
        from highdicom.io import ImageFileReader
        with ImageFileReader(path) as reader:
            if not hasattr(reader, "read_frames"):
                raise RuntimeError(
                    "Installed highdicom lacks read_frames(). Install the issue #347 "
                    "branch in editable mode before testing the optimization."
                )

            def frames():
                if mode == "A":
                    for index in indices:
                        yield reader.read_frame(index, correct_color=False)
                else:
                    for batch in chunks(indices, chunk_size):
                        yield from reader.read_frames(batch, correct_color=False)
            yield frames()
    elif backend == "pydicom":
        from pydicom.pixels import iter_pixels

        def frames():
            if mode == "A":
                for index in indices:
                    yield next(iter_pixels(path, indices=[index], raw=False))
            else:
                for batch in chunks(indices, chunk_size):
                    yield from iter_pixels(path, indices=batch, raw=False)
        yield frames()
    else:
        raise ValueError(f"Unknown backend: {backend}")
