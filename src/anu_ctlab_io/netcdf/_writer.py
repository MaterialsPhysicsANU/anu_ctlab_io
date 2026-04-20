"""Write data to the ANU CTLab netcdf data format."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import dask
import dask.array as da
import h5netcdf.legacyapi as nc4  # type: ignore[import-not-found]
import numpy as np

from anu_ctlab_io._dataset import Dataset
from anu_ctlab_io._datatype import DataType
from anu_ctlab_io._parse_history import History, serialize_history


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
    :param datatype: Data type identifier. Inferred from dataset if None.
    :param dataset_id: Unique dataset identifier. Auto-generated if not provided.
    :param max_file_size_mb: Max file size in MB. Data exceeding this is split into
        multiple files along z-axis. Default 500MB. Set to None for single file.
    :param compression_level: NetCDF compression level (0-9). Default 0 (no compression).
    :param history: History entries to add. Keys are identifiers, values are strings
        or parsed history dicts. If None, uses dataset's history attribute.
    :param extra_attrs: Additional global attributes to include.
    """
    if isinstance(path, str):
        path = Path(path)

    # Infer or validate datatype
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
        history = dataset.history

    # Serialize any parsed history dicts to strings
    serialized_history: dict[str, str] = {}
    for key, value in history.items():
        if isinstance(value, dict):
            # It's a parsed history dict, serialize it
            serialized_history[key] = serialize_history(value)
        else:
            # It's already a string, use as-is
            serialized_history[key] = str(value)

    # Prepare data array
    data_array = dataset.data.astype(datatype._dtype_uncorrected)  # type: ignore[no-untyped-call]
    zdim, ydim, xdim = data_array.shape

    # Build common attributes
    common_attrs = {
        "data_description": f"Raw reconstructed tomogram data <{np.dtype(datatype.dtype).name}>",
        "voxel_size_xyz": np.array(dataset.voxel_size, dtype=np.float32),
        "voxel_unit": str(dataset.voxel_unit),
        "coord_transform": "\n",
        "intensity_f2i_offset_scale": np.array([0.0, 1.0], dtype=np.float32),
        "offset_to_coordinate_origin_xyz": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "total_grid_size_xyz": np.array([xdim, ydim, zdim], dtype=np.int32),
        "coordinate_origin_xyz": np.array([0, 0, 0], dtype=np.int32),
        "dataset_id": dataset_id,
        **extra_attrs,
    }

    # Determine split strategy
    if max_file_size_mb is not None:
        bytes_per_slice = ydim * xdim * np.dtype(datatype._dtype_uncorrected).itemsize
        slices_per_file = max(
            1, int((max_file_size_mb * 1024 * 1024) / bytes_per_slice)
        )

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

    _write_single_netcdf(
        data_array,
        path,
        datatype,
        common_attrs,
        compression_level,
        serialized_history,
    )


def _create_dimensions(
    ncfile: nc4.Dataset,
    datatype_str: str,
    zdim: int,
    ydim: int,
    xdim: int,
) -> None:
    """Create standard NetCDF dimensions."""
    ncfile.createDimension(f"{datatype_str}_zdim", zdim)
    ncfile.createDimension(f"{datatype_str}_ydim", ydim)
    ncfile.createDimension(f"{datatype_str}_xdim", xdim)


def _is_hdf5_compatible_dtype(dtype: np.dtype) -> bool:
    """Check if a numpy dtype can be stored in HDF5."""
    # HDF5 supports numeric types, bytes, and bool
    return (
        np.issubdtype(dtype, np.number)
        or np.issubdtype(dtype, np.bytes_)
        or dtype == np.bool_
    )


def _sanitise_attribute_value(value: Any) -> Any:
    """Convert attribute value to an HDF5-compatible type.

    Converts arrays and objects with incompatible dtypes to strings.
    Strings are encoded as ASCII bytes to match reference NetCDF conventions.
    """
    match value:
        # Numeric scalars are safe for HDF5
        case int() | float() | np.integer() | np.floating():
            return value
        # Numpy arrays with HDF5-compatible dtype are safe
        case np.ndarray() if _is_hdf5_compatible_dtype(value.dtype):
            return value
        # Convert list/tuple to array and recurse
        case list() | tuple():
            try:
                arr = np.array(value)
                return _sanitise_attribute_value(arr)
            except (ValueError, TypeError):
                pass

    # Everything else: encode to string bytes
    str_value = value if isinstance(value, str) else str(value)
    return np.bytes_(str_value.encode("ascii"))


def _set_global_attributes(
    ncfile: nc4.Dataset,
    zdim: int,
    zdim_total: int,
    z_start: int,
    z_end: int,
    num_files: int,
    common_attrs: dict[str, Any],
    serialized_history: dict[str, str] | None,
    include_history: bool,
) -> None:
    """Set global NetCDF attributes."""
    ncfile.setncattr("zdim_total", np.int32(zdim_total))
    ncfile.setncattr("number_of_files", np.int32(num_files))
    ncfile.setncattr("zdim_range", np.array([z_start, z_end], dtype=np.int32))
    for key, value in common_attrs.items():  # NOTE: h5netcdf lacks setncatts
        ncfile.setncattr(key, _sanitise_attribute_value(value))

    if include_history and serialized_history:
        for key, value in serialized_history.items():
            ncfile.setncattr(
                f"history_{key}",
                _sanitise_attribute_value(value),
            )


def _create_data_variable(
    ncfile: nc4.Dataset,
    datatype: DataType,
    compression_level: int,
) -> nc4.Variable:
    """Create the main data variable in a NetCDF file."""
    datatype_str = str(datatype)
    storage_dtype_nc = _numpy_to_netcdf_dtype(np.dtype(datatype._dtype_uncorrected))
    return ncfile.createVariable(
        datatype_str,
        storage_dtype_nc,
        (f"{datatype_str}_zdim", f"{datatype_str}_ydim", f"{datatype_str}_xdim"),
        zlib=True,
        complevel=compression_level,
    )


def _get_split_directory_path(base_path: Path) -> Path:
    """Get the directory path for split NetCDF files."""
    path_str = str(base_path)
    if path_str.endswith(".nc"):
        return Path(path_str[:-3] + "_nc")
    elif path_str.endswith("_nc"):
        return base_path
    else:
        return Path(path_str + "_nc")


def _write_single_netcdf(
    data_array: da.Array,
    path: Path,
    datatype: DataType,
    common_attrs: dict[str, Any],
    compression_level: int,
    serialized_history: dict[str, str] | None,
) -> None:
    """Write a single netcdf file."""
    if not str(path).endswith(".nc"):
        path = Path(str(path) + ".nc")

    zdim, _ydim, _xdim = data_array.shape
    _write_block(
        np.asarray(data_array),
        path,
        datatype,
        zdim,
        0,
        zdim - 1,
        1,
        common_attrs,
        compression_level,
        serialized_history,
        True,
    )


def _write_block(
    block_data: np.ndarray,
    block_path: Path,
    datatype: DataType,
    zdim_total: int,
    z_start: int,
    z_end: int,
    num_files: int,
    common_attrs: dict[str, Any],
    compression_level: int,
    serialized_history: dict[str, str] | None,
    include_history: bool,
) -> None:
    """Create and write a single NetCDF block file (runs inside a dask task)."""
    block_zdim, ydim, xdim = block_data.shape
    datatype_str = str(datatype)

    with nc4.Dataset(block_path, "w", format="NETCDF4", libver="earliest") as ncfile:
        _create_dimensions(ncfile, datatype_str, block_zdim, ydim, xdim)
        _set_global_attributes(
            ncfile,
            block_zdim,
            zdim_total,
            z_start,
            z_end,
            num_files,
            common_attrs,
            serialized_history,
            include_history,
        )
        data_var = _create_data_variable(ncfile, datatype, compression_level)
        data_var[:] = block_data


def _write_split_netcdf(
    data_array: da.Array,
    base_path: Path,
    datatype: DataType,
    slices_per_file: int,
    common_attrs: dict[str, Any],
    compression_level: int,
    serialized_history: dict[str, str] | None,
) -> None:
    """Write split netcdf files into a directory."""
    dir_path = _get_split_directory_path(base_path)
    dir_path.mkdir(parents=True, exist_ok=True)

    zdim, ydim, xdim = data_array.shape
    num_files = (zdim + slices_per_file - 1) // slices_per_file

    # Rechunk so each chunk corresponds to exactly one NetCDF block
    data_array = data_array.rechunk({0: slices_per_file, 1: -1, 2: -1})  # type: ignore[no-untyped-call]
    z_delayed = data_array.to_delayed().squeeze(axis=(1, 2))  # type: ignore[no-untyped-call]

    tasks = []
    for block_idx, z_block in enumerate(z_delayed):
        z_start = block_idx * slices_per_file
        z_end = min((block_idx + 1) * slices_per_file, zdim) - 1
        block_path = dir_path / f"block{block_idx:08d}.nc"

        tasks.append(
            dask.delayed(_write_block)(  # type: ignore[attr-defined]
                z_block,
                block_path,
                datatype,
                zdim,
                z_start,
                z_end,
                num_files,
                common_attrs,
                compression_level,
                serialized_history,
                block_idx == 0,
            )
        )

    dask.compute(*tasks)  # type: ignore[attr-defined,no-untyped-call]


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
