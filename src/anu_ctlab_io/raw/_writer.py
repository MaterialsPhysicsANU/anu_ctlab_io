from pathlib import Path

import dask.array as da
import numpy as np

from anu_ctlab_io._dataset import Dataset


def dataset_to_raw(dataset: Dataset, path: Path) -> None:
    """Write dataset data to a headerless raw binary file.

    Elements are written in C-order (row-major), little-endian byte order,
    with no metadata. The dtype matches the source dataset exactly.

    Data is written incrementally: X/Y dimensions are merged into a single chunk
    and one Z-slab is computed and flushed to disk at a time, keeping memory
    usage proportional to one Z-chunk rather than the full array.

    :param dataset: The :any:`Dataset` to write.
    :param path: The path to write the raw binary file to.
    """
    data = dataset.data
    # Rechunk: preserve existing Z chunks, but merge X and Y into one chunk each.
    rechunked: da.Array = data.rechunk({1: -1, 2: -1})  # type: ignore[no-untyped-call]
    z_delayed = rechunked.to_delayed().squeeze(axis=(1, 2))  # type: ignore[no-untyped-call]

    le_dtype = data.dtype.newbyteorder("<")

    with path.open("wb") as f:
        for z_block in z_delayed:
            slab: np.ndarray = z_block.compute()  # type: ignore[assignment]
            slab = slab.astype(le_dtype, copy=False)
            f.write(slab.tobytes(order="C"))
