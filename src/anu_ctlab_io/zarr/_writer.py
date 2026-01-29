"""Write data to the ANU CTLab zarr data format."""

import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import zarr
from ome_zarr_models.v05.axes import Axis
from ome_zarr_models.v05.coordinate_transformations import VectorScale
from ome_zarr_models.v05.multiscales import Dataset as OMEDataset
from ome_zarr_models.v05.multiscales import Multiscale
from zarr.codecs import ZstdCodec

from anu_ctlab_io._dataset import Dataset
from anu_ctlab_io._datatype import DataType


def dataset_to_zarr(
    dataset: Dataset,
    path: Path | str,
    datatype: DataType | str | None = None,
    dataset_id: str | None = None,
    use_ome_zarr: bool = True,
    max_shard_size_mb: float = 1000.0,
    compression_level: int = 2,
    history: dict[str, str] | None = None,
    chunk_size_mb: float = 10.0,
    **extra_attrs: Any,
) -> None:
    """Write a :any:`Dataset` to Zarr format.

    :param dataset: The :any:`Dataset` to write.
    :param path: Path to write the Zarr store.
    :param datatype: The data type identifier. If None, attempts to infer from dataset.
    :param dataset_id: Unique identifier for the dataset. Auto-generated if not provided.
    :param use_ome_zarr: If True (default), writes OME-Zarr V0.5 group format.
        If False, writes simple Zarr V3 array with mango metadata.
    :param max_shard_size_mb: Maximum shard size in MB for Zarr v3 sharding. Default 1000 MB (1 GB).
        Shards group multiple chunks into indexed files for better filesystem performance.
    :param compression_level: Compression level (0-9) for zstd codec. Default is 2.
    :param history: Dictionary of history entries to add. Keys should be identifiers,
        values are history strings.
    :param chunk_size_mb: Target chunk size in MB for automatic chunking. Default 10.0 MB.
    :param extra_attrs: Additional attributes to include in mango metadata.
    """
    if isinstance(path, str):
        path = Path(path)

    # Infer datatype
    if datatype is None:
        if dataset._datatype is not None:
            datatype = dataset._datatype
        else:
            datatype = None
    elif isinstance(datatype, str):
        datatype = DataType.from_basename(datatype)

    # Generate dataset_id if not provided
    if dataset_id is None and datatype is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_id = f"{timestamp}_{datatype}"

    # Prepare data
    data_array = dataset.data
    storage_dtype = data_array.dtype

    shape = data_array.shape

    # Calculate inner chunks and outer shards for Zarr v3 sharding
    inner_chunks, outer_shards = _calculate_chunks_and_shards(
        shape, storage_dtype, chunk_size_mb, max_shard_size_mb
    )

    # Build mango metadata if datatype exists
    mango_attrs: dict[str, Any] | None = None
    if datatype is not None:
        mango_attrs = _build_mango_attrs(
            dataset, datatype, dataset_id, history, extra_attrs
        )

    # Write store
    if use_ome_zarr:
        _write_ome_zarr_group(
            data_array,
            path,
            dataset,
            inner_chunks,
            outer_shards,
            compression_level,
            mango_attrs,
        )
    else:
        _write_zarr_array(
            data_array,
            path,
            dataset,
            inner_chunks,
            outer_shards,
            compression_level,
            mango_attrs,
        )


def _calculate_chunks_and_shards(
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    chunk_size_mb: float,
    max_shard_size_mb: float,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Calculate inner chunks and outer shards for Zarr v3 sharding.

    In Zarr v3:
    - inner_chunks (chunks param) = subdivisions within each shard file
    - outer_shards (shards param) = how data is split into shard files

    This algorithm:
    - Enforces minimum 2 slices per chunk for compression efficiency
    - Keeps xy planes intact (critical for CT data access patterns)
    - Makes outer shards multiples of inner chunks for alignment

    :param shape: Array shape (z, y, x).
    :param dtype: Array data type.
    :param chunk_size_mb: Target size for inner chunks in MB.
    :param max_shard_size_mb: Maximum shard size in MB.
    :return: Tuple of (inner_chunks, outer_shards).
    """
    zdim, ydim, xdim = shape
    bytes_per_element = np.dtype(dtype).itemsize
    bytes_per_slice = ydim * xdim * bytes_per_element

    # Calculate inner chunk size (subdivisions within shards)
    inner_target_bytes = chunk_size_mb * 1024 * 1024
    z_inner = max(1, int(inner_target_bytes / bytes_per_slice))
    z_inner = min(z_inner, zdim)

    # Enforce minimum 2 slices per chunk for better compression efficiency
    # Single-slice chunks compress poorly (30-50% worse than 2+ slices)
    MIN_Z_SLICES = 2
    z_inner = max(MIN_Z_SLICES, z_inner)
    z_inner = min(z_inner, zdim)

    inner_chunks = (z_inner, ydim, xdim)

    # Calculate outer shard size (how data is split into files)
    shard_target_bytes = max_shard_size_mb * 1024 * 1024
    z_outer = max(1, int(shard_target_bytes / bytes_per_slice))
    z_outer = min(z_outer, zdim)

    # Ensure outer shards are at least as large as inner chunks
    # and are a multiple of inner chunks for proper alignment
    if z_outer < z_inner:
        z_outer = z_inner
    else:
        # Make outer a multiple of inner
        multiple = max(1, z_outer // z_inner)
        z_outer = z_inner * multiple
        z_outer = min(z_outer, zdim)

    outer_shards = (z_outer, ydim, xdim)

    return inner_chunks, outer_shards


def _build_mango_attrs(
    dataset: Dataset,
    datatype: DataType,
    dataset_id: str | None,
    history: dict[str, str] | None,
    extra_attrs: dict[str, Any],
) -> dict[str, Any]:
    """Build mango metadata attributes."""
    mango_attrs: dict[str, Any] = {
        "metadata_version_major": 1,
        "metadata_version_minor": 0,
        "basename": str(datatype),
        "voxel_size_xyz": [float(v) for v in dataset.voxel_size],
        "voxel_unit": str(dataset.voxel_unit),
        "coord_transform": "",
        "intensity_f2i_offset_scale": [0.0, 1.0],
        "offset_to_coordinate_origin_xyz": [0.0, 0.0, 0.0],
        "coordinate_origin_xyz": [0, 0, 0],
    }

    if dataset_id is not None:
        mango_attrs["dataset_id"] = dataset_id

    # Add history
    if history:
        mango_attrs["history"] = history
    elif isinstance(dataset.history, dict):
        mango_attrs["history"] = dataset.history
    else:
        mango_attrs["history"] = {}

    # Add extra attributes
    mango_attrs.update(extra_attrs)

    return mango_attrs


def _write_ome_zarr_group(
    data_array: da.Array,
    path: Path,
    dataset: Dataset,
    inner_chunks: tuple[int, ...],
    outer_shards: tuple[int, ...],
    compression_level: int,
    mango_attrs: dict[str, Any] | None,
) -> None:
    """Write data as an OME-Zarr V0.5 group with Zarr v3 sharding.

    In Zarr v3 sharding:
    - inner_chunks (chunks param) = subdivisions within each shard file
    - outer_shards (shards param) = how data is split into shard files
    """
    # Ensure path has .zarr extension
    if not str(path).endswith(".zarr"):
        path = Path(str(path) + ".zarr")

    # Create group
    root = zarr.create_group(path, overwrite=True)

    # Build OME metadata using ome-zarr-models classes
    ndim = data_array.ndim
    dimension_names = (
        dataset.dimension_names[:ndim]
        if len(dataset.dimension_names) >= ndim
        else dataset.dimension_names
    )

    # Build axes metadata
    axes = [
        Axis(name=name, type="space", unit=str(dataset.voxel_unit))
        for name in dimension_names
    ]

    # Build coordinate transformations with voxel size
    voxel_size_list = [float(v) for v in dataset.voxel_size]
    scale_transform = VectorScale(type="scale", scale=voxel_size_list)
    identity_transform = VectorScale(type="scale", scale=[1.0] * ndim)

    # Build OME multiscale metadata
    multiscale = Multiscale(
        name="",
        axes=axes,
        datasets=(
            OMEDataset(
                path="0",
                coordinateTransformations=(identity_transform,),
            ),
        ),
        coordinateTransformations=(scale_transform,),
    )

    # Set OME attributes on root group (convert to dict for zarr)
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [multiscale.model_dump(mode="json")],
    }

    # Add mango attributes if present
    if mango_attrs:
        root.attrs["mango"] = mango_attrs

    # Create array in subgroup "0" with sharding enabled
    array = root.create_array(
        "0",
        shape=data_array.shape,
        chunks=inner_chunks,  # Inner chunk subdivisions
        shards=outer_shards,  # Outer shard size
        dtype=data_array.dtype,
        dimension_names=list(dimension_names),
        compressors=[ZstdCodec(level=compression_level)],
        overwrite=True,
    )

    # Rechunk dask array to match zarr shard size for safe writing
    # Dask needs chunks aligned with the zarr chunk grid (outer shards, not inner chunks)
    if data_array.chunksize != outer_shards:
        data_array = data_array.rechunk(outer_shards)  # type: ignore[no-untyped-call]

    # Write data using da.to_zarr (dask 2026.1.1+)
    # Suppress Dask's overly-pessimistic warning about chunk alignment
    # The warning suggests risk of data loss, but this is not actually the case
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*chunk size.*required for Dask.*")
        data_array.to_zarr(array, compute=True)  # type: ignore[no-untyped-call]


def _write_zarr_array(
    data_array: da.Array,
    path: Path,
    dataset: Dataset,
    inner_chunks: tuple[int, ...],
    outer_shards: tuple[int, ...],
    compression_level: int,
    mango_attrs: dict[str, Any] | None,
) -> None:
    """Write data as a simple Zarr V3 array with mango metadata and sharding.

    In Zarr v3 sharding:
    - inner_chunks (chunks param) = subdivisions within each shard file
    - outer_shards (shards param) = how data is split into shard files
    """
    # Ensure path has .zarr extension
    if not str(path).endswith(".zarr"):
        path = Path(str(path) + ".zarr")

    # Only use dimension names that match the actual data dimensions
    ndim = data_array.ndim
    dimension_names = (
        dataset.dimension_names[:ndim]
        if len(dataset.dimension_names) >= ndim
        else dataset.dimension_names
    )

    # Create array with sharding enabled
    array = zarr.create_array(
        path,
        shape=data_array.shape,
        chunks=inner_chunks,  # Inner chunk subdivisions
        shards=outer_shards,  # Outer shard size
        dtype=data_array.dtype,
        dimension_names=list(dimension_names),
        compressors=[ZstdCodec(level=compression_level)],
        overwrite=True,
    )

    # Add mango attributes
    if mango_attrs:
        array.attrs["mango"] = mango_attrs

    # Rechunk dask array to match zarr shard size for safe writing
    # Dask needs chunks aligned with the zarr chunk grid (outer shards, not inner chunks)
    if data_array.chunksize != outer_shards:
        data_array = data_array.rechunk(outer_shards)  # type: ignore[no-untyped-call]

    # Write data using da.to_zarr (dask 2026.1.1+)
    # Suppress Dask's overly-pessimistic warning about chunk alignment
    # The warning suggests risk of data loss, but this is not actually the case
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*chunk size.*required for Dask.*")
        data_array.to_zarr(array, compute=True)  # type: ignore[no-untyped-call]
