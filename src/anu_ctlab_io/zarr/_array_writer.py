"""Write plain Zarr arrays."""

import logging
from pathlib import Path
from typing import Any

import dask.array as da
import zarr
from dask.delayed import Delayed

from anu_ctlab_io._dataset import Dataset
from anu_ctlab_io.zarr._layout import ChunkShape

logger = logging.getLogger(__name__)


def _write_zarr_array(
    data_array: da.Array,
    path: Path,
    dataset: Dataset,
    *,
    chunks: ChunkShape,
    subchunks: ChunkShape | None,
    create_array_kwargs: dict[str, Any],
    mango_attrs: dict[str, Any] | None,
    rechunk_before_store: bool,
    compute: bool = True,
) -> Delayed | None:
    """Write data as a simple Zarr V3 array with mango metadata.

    When ``subchunks`` is provided, Zarr v3 sharding is used:
    - chunks = primary write chunks
    - subchunks = subdivisions within each sharded chunk
    """
    if not str(path).endswith(".zarr"):
        path = Path(str(path) + ".zarr")

    # Only use dimension names that match the actual data dimensions
    ndim = data_array.ndim
    dimension_names = (
        dataset.dimension_names[:ndim]
        if len(dataset.dimension_names) >= ndim
        else dataset.dimension_names
    )

    array = zarr.create_array(
        path,
        shape=data_array.shape,
        chunks=subchunks or chunks,
        shards=chunks if subchunks is not None else None,
        dtype=data_array.dtype,
        dimension_names=list(dimension_names),
        overwrite=True,
        **create_array_kwargs,
    )
    logger.debug(
        "Created Zarr array: path=%s, shape=%s, dtype=%s, chunks=%s, shards=%s, "
        "dimension_names=%s",
        path,
        array.shape,
        array.dtype,
        array.chunks,
        array.shards,
        dimension_names,
    )

    if mango_attrs:
        array.attrs["mango"] = mango_attrs

    if rechunk_before_store:
        write_shape = array.shards or array.chunks
        logger.debug(
            "Rechunking input for Zarr write: source_chunks=%s, write_shape=%s",
            data_array.chunks,
            write_shape,
        )
        data_array = data_array.rechunk(write_shape)  # type: ignore[no-untyped-call]
    else:
        logger.debug(
            "Skipping input rechunk for Zarr write: source_chunks=%s",
            data_array.chunks,
        )

    result: Delayed | None = da.store(data_array, array, lock=False, compute=compute)  # type: ignore[arg-type]
    return result
