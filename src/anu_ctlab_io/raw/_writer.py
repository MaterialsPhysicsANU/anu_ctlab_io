import os
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
from dask.delayed import Delayed

from anu_ctlab_io._dataset import Dataset


class _RawFileStore:
    """Store target for ``dask.array.store`` that writes chunks to a raw binary file.

    Each chunk is written at the correct byte offset via ``os.pwrite``, which is
    atomic and safe for concurrent writes to non-overlapping regions.
    """

    def __init__(self, fd: int, shape: tuple[int, ...], le_dtype: np.dtype) -> None:
        self._fd = fd
        self._shape = shape
        self._le_dtype = le_dtype
        self._itemsize = le_dtype.itemsize

    def __setitem__(self, region: tuple[slice, ...], chunk: np.ndarray) -> None:
        chunk = chunk.astype(self._le_dtype, copy=False)
        z_slice, y_slice, x_slice = region
        _, ny, nx = self._shape
        if y_slice != slice(0, ny) or x_slice != slice(0, nx):
            raise ValueError(f"chunks must span full Y and X dimensions, got {region}")
        # Chunks span full Y and X, so the entire chunk is contiguous in the file.
        offset = z_slice.start * ny * nx * self._itemsize
        os.pwrite(self._fd, chunk.tobytes(order="C"), offset)


def dataset_to_raw(
    dataset: Dataset,
    path: Path,
    compute: bool = True,
    **kwargs: Any,
) -> Delayed | None:
    """Write dataset data to a headerless raw binary file.

    Elements are written in C-order (row-major), little-endian byte order,
    with no metadata. The dtype matches the source dataset exactly.

    Data is written via ``dask.array.store`` using a single ``compute()`` call,
    with each chunk sought to its correct byte offset so chunks may be written
    in any order.

    :param dataset: The :any:`Dataset` to write.
    :param path: The path to write the raw binary file to.
    """
    data: da.Array = dataset.data.rechunk({1: -1, 2: -1})  # type: ignore[no-untyped-call]
    le_dtype = data.dtype.newbyteorder("<")

    # Pre-allocate
    total_bytes = int(np.prod(data.shape)) * le_dtype.itemsize
    with path.open("wb") as f:
        f.truncate(total_bytes)

    fd = os.open(path, os.O_WRONLY)
    try:
        store = _RawFileStore(fd, data.shape, le_dtype)
        output: Delayed | None = da.store(data, store, lock=False, compute=compute)  # type: ignore[arg-type]
        return output
    finally:
        os.close(fd)
