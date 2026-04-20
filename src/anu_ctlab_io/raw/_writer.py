from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
from dask.delayed import Delayed, delayed
from dask.graph_manipulation import bind

from anu_ctlab_io._dataset import Dataset


class _RawFileStore:
    """Store target for ``dask.array.store`` that writes chunks to a raw binary file.

    Each chunk is written at the correct byte offset via a ``numpy.memmap``, which
    is safe for concurrent writes to non-overlapping regions.

    The memmap is opened and closed per ``__setitem__`` call so that the store
    object can be serialised and used by distributed workers in separate processes.
    """

    def __init__(self, path: Path, shape: tuple[int, ...], le_dtype: np.dtype) -> None:
        self._path = path
        self._shape = shape
        self._le_dtype = le_dtype

    def __setitem__(self, region: tuple[slice, ...], chunk: np.ndarray) -> None:
        chunk = chunk.astype(self._le_dtype, copy=False)
        mm = np.memmap(self._path, dtype=self._le_dtype, mode="r+", shape=self._shape)
        try:
            mm[region] = chunk
            mm.flush()
        finally:
            del mm


def dataset_to_raw(
    dataset: Dataset,
    path: Path,
    compute: bool = True,
    scheduler: str | None = None,
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
    :param scheduler: Dask scheduler to use for computation (e.g. ``"synchronous"``,
        ``"threads"``, ``"processes"``). ``None`` uses the dask default.
    """
    data: da.Array = dataset.data
    le_dtype = data.dtype.newbyteorder("<")

    total_bytes = int(np.prod(data.shape)) * le_dtype.itemsize

    @delayed  # type: ignore[misc]
    def preallocate() -> None:
        with path.open("wb") as f:
            f.truncate(total_bytes)

    store = _RawFileStore(path, data.shape, le_dtype)
    writes: Delayed = da.store(data, store, lock=False, compute=False)  # type: ignore[arg-type]
    result: Delayed = bind(writes, preallocate(), omit=data)
    if compute:
        result.compute(scheduler=scheduler)  # type: ignore[no-untyped-call]
        return None
    return result
