"""Write data to the ANU CTLab zarr data format."""

import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

from dask.delayed import Delayed

from anu_ctlab_io._dataset import Dataset
from anu_ctlab_io._datatype import DataType
from anu_ctlab_io._parse_history import History
from anu_ctlab_io.zarr._array_writer import _write_zarr_array
from anu_ctlab_io.zarr._layout import (
    ChunkSpec,
    _chunk_key_encoding,
    _resolve_zarr_layout,
)
from anu_ctlab_io.zarr._multiscale import DownsampleMethod
from anu_ctlab_io.zarr._ome_writer import OMEZarrVersion, _write_ome_zarr_group

__all__ = ["OMEZarrVersion", "dataset_to_zarr"]

logger = logging.getLogger(__name__)


def dataset_to_zarr(
    dataset: Dataset,
    path: Path | str,
    datatype: DataType | str | None = None,
    dataset_id: str | None = None,
    ome_zarr_version: OMEZarrVersion | None = OMEZarrVersion.v05,
    max_shard_size_mb: float
    | None = None,  # TODO: 2.0.0 Remove this deprecated parameter
    history: History | None = None,
    chunk_size_mb: float | None = None,  # TODO: 2.0.0 Remove this deprecated parameter
    chunks: ChunkSpec = "auto",
    shards: ChunkSpec | None = "auto",
    create_array_kwargs: dict[str, Any] | None = None,
    dimension_separator_threshold: int | None = 64,
    input_aligned_chunks: bool = False,
    multiscale: bool = True,
    downsample_method: DownsampleMethod = "strided",
    compute: bool = True,
    **extra_attrs: Any,
) -> Delayed | None:
    """Write a :any:`Dataset` to Zarr format.

    By default, sharded writes use power-of-two square or cubic shapes targeting
    ``32**3`` elements for Zarr chunks and ``512**3`` elements for shards. Shapes
    are trimmed to the array dimensions, and shards are rounded up to chunk
    multiples. With ``shards=None``, ``chunks='auto'`` uses the larger unsharded
    write target of ``512**3`` elements.

    :param dataset: The :any:`Dataset` to write.
    :param path: Path to write the Zarr store.
    :param datatype: The data type identifier. If None, attempts to infer from dataset.
    :param dataset_id: Unique identifier for the dataset. Auto-generated if not provided.
    :param ome_zarr_version: OME-Zarr specification version to use.
        Set to :any:`OMEZarrVersion.v05` (default) to write OME-Zarr V0.5 group format.
        Set to ``None`` to write a simple Zarr V3 array with mango metadata.
    :param max_shard_size_mb: Maximum shard size in MB for Zarr v3 sharding.
        Deprecated and ignored.
        Passing this emits a warning and leaves layout selection to ``chunks``/``shards``.
    :param history: Dictionary of history entries to add.
        Keys should be identifiers, values are history strings.
    :param chunk_size_mb: Target chunk size in MB for automatic chunking.
        Deprecated and ignored.
        Passing this emits a warning and leaves layout selection to ``chunks``/``shards``.
    :param chunks: Explicit chunk shape as a tuple shape (e.g., ``(10, 512, 512)``), int (target # of elements), or ``'auto'``.
        Can be provided on its own to write a non-sharded Zarr array, or together with ``shards`` to use the sharding codec.
        An integer specifies the target number of elements for an automatically derived layout.
        With sharding enabled, ``'auto'`` uses a default target corresponding to ``32**3`` or ``256**2`` elements.
        With ``shards=None``, ``'auto'`` uses the larger unsharded target corresponding to ``512**3`` or ``8192**2`` elements.
        To write without sharding, pass ``shards=None`` explicitly.
        A value of ``0`` in a shape tuple means "span this axis" — the full array axis for unsharded writes, or the full shard axis when ``shards`` is also provided.
    :param shards: Explicit shard shape as a tuple shape (e.g., ``(100, 512, 512)``), int (target # of elements), or ``'auto'``.
        May be provided together with an explicit ``chunks`` tuple.
        Providing an explicit shard shape with ``chunks='auto'`` is an error.
        An integer specifies the target number of elements for an automatically derived layout.
        Use ``None`` to disable sharding, or ``'auto'`` to use the default target of ``512**3`` or ``8192**2`` elements.
        A value of ``0`` in a shape tuple means "span the full array axis".
        When provided, the user is responsible for ensuring shard shapes are evenly divisible by chunk shapes.
    :param create_array_kwargs: Additional keyword arguments to pass to zarr.create_array().
        For example, to set compression: ``create_array_kwargs={'compressors': [ZstdCodec(level=5)]}``.
    :param dimension_separator_threshold: Use ``'/'`` as the chunk key dimension
        separator when the number of chunks exceeds this threshold; otherwise use
        ``'.'``. ``None`` uses the Zarr default of ``'/'``.
    :param input_aligned_chunks: Choose Zarr chunks/shards aligned to the input dask
        chunk grid and skip rechunking before storing. With sharding is enabled, shards
        are aligned to the dask chunks and inner chunks evenly divide each shard.
        With ``shards=None``, chunks are aligned to the dask chunks.
    :param multiscale: For OME-Zarr output, write a multiscale pyramid where possible.
        Plain Zarr output remains single-scale.
    :param downsample_method: Downsampling method for multiscale OME-Zarr output.
        ``"strided"`` (default) takes every second voxel without averaging.
        ``"mean"`` averages each 2x block.
        ``"mode"`` chooses the smallest value when a block has tied modes.
    :param compute: If ``True`` (default), compute immediately. If ``False``, return
        a :any:`dask.delayed.Delayed` for deferred execution.
    :param extra_attrs: Additional attributes to include in mango metadata.
    """
    if isinstance(path, str):
        path = Path(path)

    logger.debug(
        "Writing dataset to Zarr: path=%s, datatype=%s, dataset_id=%s, "
        "ome_zarr_version=%s, chunks=%s, shards=%s, input_aligned_chunks=%s, "
        "dimension_separator_threshold=%s, multiscale=%s, downsample_method=%s, "
        "compute=%s, create_array_kwargs=%s, extra_attrs=%s",
        path,
        datatype,
        dataset_id,
        ome_zarr_version,
        chunks,
        shards,
        input_aligned_chunks,
        dimension_separator_threshold,
        multiscale,
        downsample_method,
        compute,
        create_array_kwargs,
        sorted(extra_attrs),
    )

    if datatype is None:
        if dataset._datatype is not None:
            datatype = dataset._datatype
        else:
            datatype = None
    elif isinstance(datatype, str):
        datatype = DataType.from_basename(datatype)

    if downsample_method not in ("strided", "mean", "mode"):
        raise ValueError(
            "downsample_method must be one of 'strided', 'mean', or 'mode'. "
            f"Got {downsample_method!r}."
        )

    # Generate dataset_id if not provided
    if dataset_id is None and datatype is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_id = f"{timestamp}_{datatype}"

    data_array = dataset.data
    logger.debug(
        "Input dask array: shape=%s, dtype=%s, chunks=%s, chunksize=%s, npartitions=%s",
        data_array.shape,
        data_array.dtype,
        data_array.chunks,
        data_array.chunksize,
        data_array.npartitions,
    )

    ignored_size_args = ", ".join(
        name
        for name, value in (
            ("chunk_size_mb", chunk_size_mb),
            ("max_shard_size_mb", max_shard_size_mb),
        )
        if value is not None
    )
    if ignored_size_args:
        verb = "are" if "," in ignored_size_args else "is"
        warnings.warn(
            f"{ignored_size_args} {verb} ignored when writing Zarr. Use chunks/shards or 'auto' instead.",
            UserWarning,
            stacklevel=2,
        )

    if isinstance(shards, tuple) and chunks == "auto" and not input_aligned_chunks:
        raise ValueError("shards cannot be provided without explicit chunks")

    subchunks = chunks if shards is not None else None
    chunks = chunks if shards is None else shards

    mango_attrs: dict[str, Any] | None = None
    if datatype is not None:
        mango_attrs = _build_mango_attrs(
            dataset, datatype, dataset_id, history, extra_attrs
        )

    create_array_kwargs = dict(create_array_kwargs or {})

    if ome_zarr_version is not None:
        logger.debug("Writing OME-Zarr group with version %s", ome_zarr_version)
        return _write_ome_zarr_group(
            data_array,
            path,
            dataset,
            chunks=chunks,
            subchunks=subchunks,
            create_array_kwargs=create_array_kwargs,
            mango_attrs=mango_attrs,
            ome_zarr_version=ome_zarr_version,
            rechunk_before_store=not input_aligned_chunks,
            multiscale=multiscale,
            downsample_method=downsample_method,
            dimension_separator_threshold=dimension_separator_threshold,
            compute=compute,
        )
    else:
        chunks, subchunks = _resolve_zarr_layout(
            shape=data_array.shape,
            chunks=chunks,
            subchunks=subchunks,
            aligned_chunks=data_array.chunks if input_aligned_chunks else None,
        )
        logger.debug(
            "Resolved Zarr layout: chunks=%s, subchunks=%s, rechunk_before_store=%s",
            chunks,
            subchunks,
            not input_aligned_chunks,
        )
        if dimension_separator_threshold is not None:
            create_array_kwargs["chunk_key_encoding"] = _chunk_key_encoding(
                data_array.shape, chunks, dimension_separator_threshold
            )
            logger.debug(
                "Selected chunk key encoding: key_chunks=%s, encoding=%s",
                chunks,
                create_array_kwargs["chunk_key_encoding"],
            )

        logger.debug("Writing plain Zarr array")
        return _write_zarr_array(
            data_array,
            path,
            dataset,
            chunks=chunks,
            subchunks=subchunks,
            create_array_kwargs=create_array_kwargs,
            mango_attrs=mango_attrs,
            rechunk_before_store=not input_aligned_chunks,
            compute=compute,
        )


def _build_mango_attrs(
    dataset: Dataset,
    datatype: DataType,
    dataset_id: str | None,
    history: History | None,
    extra_attrs: dict[str, Any],
) -> dict[str, Any]:
    """Build mango metadata attributes."""
    mango_attrs: dict[str, Any] = {
        "metadata_version_major": 1,
        "metadata_version_minor": 0,
        "basename": str(datatype),
        "voxel_size_xyz": [float(v) for v in dataset.voxel_size],
        "voxel_unit": str(dataset.voxel_unit),
        "coord_transform": "",  # FIXME
        "intensity_f2i_offset_scale": [0.0, 1.0],
        "offset_to_coordinate_origin_xyz": [0.0, 0.0, 0.0],  # FIXME
        "coordinate_origin_xyz": [0, 0, 0],  # FIXME
    }

    if dataset_id is not None:
        mango_attrs["dataset_id"] = dataset_id

    if history is not None:
        mango_attrs["history"] = history
    else:
        mango_attrs["history"] = dataset.history

    mango_attrs.update(extra_attrs)

    return mango_attrs
