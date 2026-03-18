from pathlib import Path

import dask.array as da
import numpy as np
from dask.delayed import Delayed

from anu_ctlab_io._dataset import Dataset


def dataset_to_raw(
    dataset: Dataset, path: Path, compute: bool = True
) -> Delayed | None:
    """Write dataset data to a headerless raw binary file.

    Elements are written in C-order (row-major), little-endian byte order,
    with no metadata. The dtype matches the source dataset exactly.

    Data is written via ``dask.array.store`` into a ``numpy.memmap``, allowing
    chunks to be written concurrently to non-overlapping regions.

    :param dataset: The :any:`Dataset` to write.
    :param path: The path to write the raw binary file to.
    """
    data: da.Array = dataset.data.rechunk({1: -1, 2: -1})  # type: ignore[no-untyped-call]
    le_dtype = data.dtype.newbyteorder("<")
    mmap = np.memmap(path, dtype=le_dtype, mode="w+", shape=data.shape)
    result: Delayed | None = da.store(data, mmap, lock=False, compute=compute)
    return result
