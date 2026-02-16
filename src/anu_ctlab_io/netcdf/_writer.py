"""Write data to the ANU CTLab netcdf data format."""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import dask.array as da
import netCDF4 as nc4  # type: ignore[import-not-found]
import numpy as np

from anu_ctlab_io._dataset import Dataset
from anu_ctlab_io._datatype import DataType
from anu_ctlab_io._parse_history import History, serialize_history


def _get_block_slices(array: da.Array) -> list[tuple[slice, ...]]:
    """Get slice objects for each block in a dask array.

    Args:
        array: Dask array with defined chunks

    Returns:
        List of tuples of slices, one per block
    """
    slices = []
    for block_id in np.ndindex(array.numblocks):
        block_slices = tuple(
            slice(sum(array.chunks[dim][:idx]), sum(array.chunks[dim][: idx + 1]))
            for dim, idx in enumerate(block_id)
        )
        slices.append(block_slices)
    return slices


def _write_netcdf_data(data_array: da.Array, netcdf_var: nc4.Variable) -> None:
    """Write dask array to NetCDF variable using appropriate strategy for scheduler.

    Strategy selection:
    - Distributed scheduler: Compute chunks on workers, write on client
    - Synchronous/threaded: Use LockedNetCDFVariable with current scheduler

    Args:
        data_array: Dask array to write
        netcdf_var: NetCDF variable to write to
    """
    try:
        from distributed import Lock as DistributedLock
        from distributed import as_completed, get_client

        client = get_client()  # Raises ValueError if no client

        lock = DistributedLock("netcdf_write")
        delayed_blocks = data_array.to_delayed().flatten()  # type: ignore[no-untyped-call]
        block_slices = _get_block_slices(data_array)

        future_to_slices = {
            client.compute(block): slices
            for block, slices in zip(delayed_blocks, block_slices, strict=True)
        }

        for future in as_completed(future_to_slices):
            result = future.result()  # Get computed chunk
            slices = future_to_slices[future]
            with lock:
                netcdf_var[slices] = result

    except (ImportError, ValueError):
        # No distributed client - use locked wrapper
        locked_var = LockedNetCDFVariable(netcdf_var)
        da.store(data_array, locked_var, compute=True)  # type: ignore[arg-type]


class LockedNetCDFVariable:
    """Thread-safe wrapper for NetCDF variables.

    Serializes write operations to prevent HDF5/NetCDF4 concurrent write
    conflicts while allowing dask to schedule computation in parallel.

    The HDF5 library used by NetCDF4 doesn't support multiple threads
    writing to the same file simultaneously. This wrapper uses a lock
    to serialize write operations while allowing dask to compute chunks
    in parallel with any scheduler (synchronous, threaded, or distributed).

    Args:
        netcdf_var: NetCDF variable to wrap
        lock: Optional threading.Lock. Creates new lock if not provided.

    Example:
        >>> with nc4.Dataset('file.nc', 'w') as ncfile:
        ...     var = ncfile.createVariable('data', 'f4', ('z', 'y', 'x'))
        ...     locked_var = LockedNetCDFVariable(var)
        ...     da.store(dask_array, locked_var, compute=True)
    """

    def __init__(
        self, netcdf_var: nc4.Variable, lock: threading.Lock | None = None
    ) -> None:
        self.var = netcdf_var
        self.lock = lock if lock is not None else threading.Lock()

    def __setitem__(self, key: Any, value: Any) -> None:
        """Thread-safe write to NetCDF variable."""
        with self.lock:
            self.var[key] = value

    @property
    def shape(self) -> tuple[int, ...]:
        """Pass through shape for dask compatibility."""
        return self.var.shape  # type: ignore[no-any-return]

    @property
    def dtype(self) -> np.dtype[np.generic]:
        """Pass through dtype for dask compatibility."""
        return self.var.dtype  # type: ignore[no-any-return]


def dataset_to_netcdf(
    dataset: Dataset,
    path: Path | str,
    datatype: DataType | str | None = None,
    dataset_id: str | None = None,
    max_file_size_mb: float | None = 500.0,
    compression_level: int = 0,
    history: History | None = None,
    **extra_attrs: Any,
) -> None:
    """Write a :any:`Dataset` to netcdf format.

    :param dataset: The :any:`Dataset` to write.
    :param path: Path to write the netcdf file or directory (if splitting).
    :param datatype: The data type identifier. If None, attempts to infer from dataset.
    :param dataset_id: Unique identifier for the dataset. Auto-generated if not provided.
    :param max_file_size_mb: Maximum file size in MB. If specified and data exceeds this,
        it will be split along the z-axis into multiple block files in a directory.
        Default is 500.0 MB to avoid memory issues with large datasets.
        Set to None to force single file output.
    :param compression_level: NetCDF compression level (0-9). Default is 0 (no compression),
        because NetCDF compression is really slow.
    :param history: Dictionary of history entries to add. Keys should be identifiers,
        values can be history strings or parsed history dicts (which will be serialized).
        If None, uses the dataset's history attribute.
    :param extra_attrs: Additional global attributes to include.

    .. note::
        This function uses thread-safe locking for writes to handle NetCDF4's
        parallel write limitations. The HDF5 library used by NetCDF4 doesn't
        support multiple threads writing to the same file simultaneously.

    .. note::
        By default, large datasets are split into multiple files (~500MB each) to
        reduce memory usage during writes. Small datasets (<500MB) are written as
        a single file. To force single file output, set max_file_size_mb=None.
    """
    if isinstance(path, str):
        path = Path(path)

    # Infer datatype
    if datatype is None:
        if dataset._datatype is not None:
            datatype = dataset._datatype
        else:
            raise ValueError(
                "datatype must be provided when dataset does not have _datatype"
            )
    elif isinstance(datatype, str):
        datatype = DataType.from_basename(datatype)

    # Generate dataset_id if not provided
    if dataset_id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_id = f"{timestamp}_{datatype}"

    # Handle history: use dataset history if not explicitly provided
    if history is None:
        history = dataset.history if isinstance(dataset.history, dict) else {}

    # Serialize any parsed history dicts to strings
    serialized_history: dict[str, str] = {}
    for key, value in history.items():
        if isinstance(value, dict):
            # It's a parsed history dict, serialize it
            serialized_history[key] = serialize_history(value)
        else:
            # It's already a string, use as-is
            serialized_history[key] = str(value)

    # Prepare data
    data_array = dataset.data

    # Convert to storage dtype (may need to handle signed/unsigned conversion)
    storage_dtype = datatype._dtype_uncorrected
    data_array = data_array.astype(storage_dtype)  # type: ignore[no-untyped-call]

    shape = data_array.shape
    zdim, ydim, xdim = shape

    # Build common attributes
    dtype_name = np.dtype(datatype.dtype).name
    common_attrs = {
        "data_description": f"Raw reconstructed tomogram data <{dtype_name}>",
        "voxel_size_xyz": np.array(dataset.voxel_size, dtype=np.float32),
        "voxel_unit": str(dataset.voxel_unit),
        "coord_transform": "\n",
        "intensity_f2i_offset_scale": np.array([0.0, 1.0], dtype=np.float32),
        "offset_to_coordinate_origin_xyz": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "total_grid_size_xyz": np.array([xdim, ydim, zdim], dtype=np.int32),
        "coordinate_origin_xyz": np.array([0, 0, 0], dtype=np.int32),
        "dataset_id": dataset_id,
    }
    common_attrs.update(extra_attrs)

    # Determine if we need to split
    if max_file_size_mb is not None:
        # Estimate size per z-slice
        bytes_per_slice = ydim * xdim * np.dtype(storage_dtype).itemsize
        max_bytes = max_file_size_mb * 1024 * 1024
        slices_per_file = max(1, int(max_bytes / bytes_per_slice))

        if zdim > slices_per_file:
            _write_split_netcdf(
                data_array,
                path,
                datatype,
                slices_per_file,
                common_attrs,
                compression_level,
                serialized_history,
            )
            return

    # Write single file
    _write_single_netcdf(
        data_array,
        path,
        datatype,
        common_attrs,
        compression_level,
        serialized_history,
    )


def _write_single_netcdf(
    data_array: da.Array,
    path: Path,
    datatype: DataType,
    common_attrs: dict[str, Any],
    compression_level: int,
    history: dict[str, str] | None,
) -> None:
    """Write a single netcdf file."""
    # Ensure path has .nc extension
    if not str(path).endswith(".nc"):
        path = Path(str(path) + ".nc")

    zdim, ydim, xdim = data_array.shape
    datatype_str = str(datatype)

    with nc4.Dataset(path, "w", format="NETCDF4") as ncfile:
        # Create dimensions
        ncfile.createDimension(f"{datatype_str}_zdim", zdim)
        ncfile.createDimension(f"{datatype_str}_ydim", ydim)
        ncfile.createDimension(f"{datatype_str}_xdim", xdim)

        # Set global attributes
        ncfile.setncattr("zdim_total", zdim)
        ncfile.setncattr("number_of_files", 1)
        ncfile.setncattr("zdim_range", np.array([0, zdim - 1], dtype=np.int32))
        ncfile.setncatts(common_attrs)

        # Add history attributes
        if history:
            for hist_key, hist_value in history.items():
                ncfile.setncattr(f"history_{hist_key}", hist_value)

        # Create main data variable
        storage_dtype_nc = _numpy_to_netcdf_dtype(np.dtype(datatype._dtype_uncorrected))
        data_var = ncfile.createVariable(
            datatype_str,
            storage_dtype_nc,
            (f"{datatype_str}_zdim", f"{datatype_str}_ydim", f"{datatype_str}_xdim"),
            zlib=True,
            complevel=compression_level,
        )

        # Write data using scheduler-appropriate strategy
        _write_netcdf_data(data_array, data_var)


def _write_split_netcdf(
    data_array: da.Array,
    base_path: Path,
    datatype: DataType,
    slices_per_file: int,
    common_attrs: dict[str, Any],
    compression_level: int,
    history: dict[str, str] | None,
) -> None:
    """Write split netcdf files into a directory.

    Each block is written using a thread-safe locked wrapper to handle
    NetCDF's parallel write limitations.
    """
    # Create directory with _nc suffix
    # If path ends with .nc, replace it with _nc
    path_str = str(base_path)
    if path_str.endswith(".nc"):
        dir_path = Path(path_str[:-3] + "_nc")
    elif path_str.endswith("_nc"):
        dir_path = base_path
    else:
        dir_path = Path(path_str + "_nc")

    dir_path.mkdir(parents=True, exist_ok=True)

    zdim, ydim, xdim = data_array.shape
    num_files = (zdim + slices_per_file - 1) // slices_per_file
    datatype_str = str(datatype)

    for block_idx in range(num_files):
        z_start = block_idx * slices_per_file
        z_end = min((block_idx + 1) * slices_per_file, zdim) - 1
        block_zdim = z_end - z_start + 1

        block_path = dir_path / f"block{block_idx:08d}.nc"

        with nc4.Dataset(block_path, "w", format="NETCDF4") as ncfile:
            # Create dimensions
            ncfile.createDimension(f"{datatype_str}_zdim", block_zdim)
            ncfile.createDimension(f"{datatype_str}_ydim", ydim)
            ncfile.createDimension(f"{datatype_str}_xdim", xdim)

            # Set global attributes
            ncfile.setncattr("zdim_total", zdim)
            ncfile.setncattr("number_of_files", num_files)
            ncfile.setncattr("zdim_range", np.array([z_start, z_end], dtype=np.int32))
            ncfile.setncatts(common_attrs)

            # Only first block gets history
            if block_idx == 0 and history:
                for hist_key, hist_value in history.items():
                    ncfile.setncattr(f"history_{hist_key}", hist_value)

            # Create main data variable
            storage_dtype_nc = _numpy_to_netcdf_dtype(
                np.dtype(datatype._dtype_uncorrected)
            )
            data_var = ncfile.createVariable(
                datatype_str,
                storage_dtype_nc,
                (
                    f"{datatype_str}_zdim",
                    f"{datatype_str}_ydim",
                    f"{datatype_str}_xdim",
                ),
                zlib=True,
                complevel=compression_level,
            )

            # Write block data using scheduler-appropriate strategy
            block_data = data_array[z_start : z_end + 1, :, :]
            _write_netcdf_data(block_data, data_var)


_NUMPY_TO_NETCDF_DTYPE_MAP = {
    np.dtype(np.int8): "i1",
    np.dtype(np.int16): "i2",
    np.dtype(np.int32): "i4",
    np.dtype(np.int64): "i8",
    np.dtype(np.uint8): "u1",
    np.dtype(np.uint16): "u2",
    np.dtype(np.uint32): "u4",
    np.dtype(np.uint64): "u8",
    np.dtype(np.float16): "f2",
    np.dtype(np.float32): "f4",
    np.dtype(np.float64): "f8",
}


def _numpy_to_netcdf_dtype(dtype: np.dtype) -> str:
    """Convert numpy dtype to netCDF4 dtype string."""
    dtype = np.dtype(dtype)
    try:
        return _NUMPY_TO_NETCDF_DTYPE_MAP[dtype]
    except KeyError:
        raise ValueError(f"Unsupported dtype: {dtype}") from None
