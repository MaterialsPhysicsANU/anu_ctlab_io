"""Write data to the ANU CTLab zarr data format."""

from datetime import datetime
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import zarr
from ome_zarr_models.common.coordinate_transformations import VectorScale
from ome_zarr_models.v05.axes import Axis
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
    max_store_size_mb: float | None = None,
    compression_level: int = 2,
    history: dict[str, str] | None = None,
    chunk_size_mb: float = 10.0,
    **extra_attrs: Any,
) -> None:
    """Write a :any:`Dataset` to Zarr format.

    :param dataset: The :any:`Dataset` to write.
    :param path: Path to write the Zarr store or directory (if splitting).
    :param datatype: The data type identifier. If None, attempts to infer from dataset.
    :param dataset_id: Unique identifier for the dataset. Auto-generated if not provided.
    :param use_ome_zarr: If True (default), writes OME-Zarr V0.5 group format.
        If False, writes simple Zarr V3 array with mango metadata.
    :param max_store_size_mb: Maximum store size in MB. If specified and data exceeds this,
        it will be split along the z-axis into multiple stores. If None, writes a single store.
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
    zdim, ydim, xdim = shape

    # Calculate chunks
    chunks = _calculate_chunks(shape, storage_dtype, chunk_size_mb)

    # Build mango metadata if datatype exists
    mango_attrs: dict[str, Any] | None = None
    if datatype is not None:
        mango_attrs = _build_mango_attrs(
            dataset, datatype, dataset_id, history, extra_attrs
        )

    # Determine if we need to split
    if max_store_size_mb is not None:
        bytes_per_slice = ydim * xdim * np.dtype(storage_dtype).itemsize
        max_bytes = max_store_size_mb * 1024 * 1024
        slices_per_store = max(1, int(max_bytes / bytes_per_slice))

        if zdim > slices_per_store:
            _write_split_zarr(
                data_array,
                path,
                dataset,
                chunks,
                compression_level,
                use_ome_zarr,
                mango_attrs,
                slices_per_store,
            )
            return

    # Write single store
    if use_ome_zarr:
        _write_ome_zarr_group(
            data_array,
            path,
            dataset,
            chunks,
            compression_level,
            mango_attrs,
        )
    else:
        _write_zarr_array(
            data_array,
            path,
            dataset,
            chunks,
            compression_level,
            mango_attrs,
        )


def _calculate_chunks(
    shape: tuple[int, ...], dtype: np.dtype[Any], target_mb: float = 10.0
) -> tuple[int, ...]:
    """Calculate appropriate chunk sizes for the data.

    Aims for chunks of approximately target_mb in size, while keeping
    y and x dimensions unchunked for better slicing performance.
    """
    zdim, ydim, xdim = shape
    bytes_per_element = np.dtype(dtype).itemsize
    bytes_per_slice = ydim * xdim * bytes_per_element
    target_bytes = target_mb * 1024 * 1024

    # Calculate z chunks, keep y and x full
    z_chunk = max(1, int(target_bytes / bytes_per_slice))
    z_chunk = min(z_chunk, zdim)

    return (z_chunk, ydim, xdim)


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
    chunks: tuple[int, ...],
    compression_level: int,
    mango_attrs: dict[str, Any] | None,
) -> None:
    """Write data as an OME-Zarr V0.5 group."""
    # Ensure path has .zarr extension
    if not str(path).endswith(".zarr"):
        path = Path(str(path) + ".zarr")

    # Create group
    root = zarr.create_group(path, overwrite=True)

    # Build OME metadata
    # Only use dimension names that match the actual data dimensions
    ndim = data_array.ndim
    dimension_names = (
        dataset.dimension_names[:ndim]
        if len(dataset.dimension_names) >= ndim
        else dataset.dimension_names
    )
    axes = [
        Axis(name=name, type="space", unit=str(dataset.voxel_unit))
        for name in dimension_names
    ]

    # Create coordinate transformations with voxel size
    voxel_size_list = [float(v) for v in dataset.voxel_size]
    coord_transforms = (VectorScale(type="scale", scale=voxel_size_list),)

    # Create dataset metadata (single resolution)
    ome_dataset = OMEDataset(
        path="0",
        coordinateTransformations=(VectorScale(type="scale", scale=[1.0, 1.0, 1.0]),),
    )

    multiscale = Multiscale(
        axes=tuple(axes),
        datasets=(ome_dataset,),
        coordinateTransformations=coord_transforms,
    )

    # Set OME attributes on root group
    root.attrs["ome"] = {"version": "0.5", "multiscales": [multiscale.model_dump()]}

    # Add mango attributes if present
    if mango_attrs:
        root.attrs["mango"] = mango_attrs

    # Create array in subgroup "0"
    array = root.create_array(
        "0",
        shape=data_array.shape,
        chunks=chunks,
        dtype=data_array.dtype,
        dimension_names=list(dimension_names),
        compressors=[ZstdCodec(level=compression_level)],
        overwrite=True,
    )

    # Write data
    computed_data: np.ndarray[Any, Any] = data_array.compute()  # type: ignore[no-untyped-call]
    array[:] = computed_data


def _write_zarr_array(
    data_array: da.Array,
    path: Path,
    dataset: Dataset,
    chunks: tuple[int, ...],
    compression_level: int,
    mango_attrs: dict[str, Any] | None,
) -> None:
    """Write data as a simple Zarr V3 array with mango metadata."""
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

    # Create array
    array = zarr.create_array(
        path,
        shape=data_array.shape,
        chunks=chunks,
        dtype=data_array.dtype,
        dimension_names=list(dimension_names),
        compressors=[ZstdCodec(level=compression_level)],
        overwrite=True,
    )

    # Add mango attributes
    if mango_attrs:
        array.attrs["mango"] = mango_attrs

    # Write data
    computed_data: np.ndarray[Any, Any] = data_array.compute()  # type: ignore[no-untyped-call]
    array[:] = computed_data


def _write_split_zarr(
    data_array: da.Array,
    base_path: Path,
    dataset: Dataset,
    chunks: tuple[int, ...],
    compression_level: int,
    use_ome_zarr: bool,
    mango_attrs: dict[str, Any] | None,
    slices_per_store: int,
) -> None:
    """Write split Zarr stores into a directory."""
    # Create directory with _zarr suffix
    path_str = str(base_path)
    if path_str.endswith(".zarr"):
        dir_path = Path(path_str[:-5] + "_zarr")
    elif path_str.endswith("_zarr"):
        dir_path = base_path
    else:
        dir_path = Path(path_str + "_zarr")

    dir_path.mkdir(parents=True, exist_ok=True)

    zdim, ydim, xdim = data_array.shape
    num_stores = (zdim + slices_per_store - 1) // slices_per_store

    # Compute data once
    computed_data: np.ndarray[Any, Any] = data_array.compute()  # type: ignore[no-untyped-call]

    for store_idx in range(num_stores):
        z_start = store_idx * slices_per_store
        z_end = min((store_idx + 1) * slices_per_store, zdim)
        store_zdim = z_end - z_start

        store_path = dir_path / f"store{store_idx:08d}.zarr"
        store_data = da.from_array(computed_data[z_start:z_end, :, :])  # type: ignore[no-untyped-call]

        # Adjust chunks for this store
        store_chunks = (min(chunks[0], store_zdim), chunks[1], chunks[2])

        # Create a modified dataset for this store
        store_dataset = Dataset(
            data=store_data,
            dimension_names=dataset.dimension_names,
            voxel_unit=dataset.voxel_unit,
            voxel_size=dataset.voxel_size,
            datatype=dataset._datatype,
            history=dataset.history if isinstance(dataset.history, dict) else {},
        )

        # Write the store
        if use_ome_zarr:
            _write_ome_zarr_group(
                store_data,
                store_path,
                store_dataset,
                store_chunks,
                compression_level,
                mango_attrs,
            )
        else:
            _write_zarr_array(
                store_data,
                store_path,
                store_dataset,
                store_chunks,
                compression_level,
                mango_attrs,
            )
