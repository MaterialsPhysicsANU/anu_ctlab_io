"""Write data to the ANU CTLab netcdf data format."""

from datetime import datetime
from pathlib import Path
from typing import Any

import dask.array as da

# Type stubs for netCDF4 don't exist, ignore import
import netCDF4 as nc4  # type: ignore[import-not-found]
import numpy as np

from anu_ctlab_io._dataset import Dataset
from anu_ctlab_io._datatype import DataType


def dataset_to_netcdf(
    dataset: Dataset,
    path: Path | str,
    datatype: DataType | str | None = None,
    dataset_id: str | None = None,
    max_file_size_mb: float | None = None,
    compression_level: int = 2,
    history: dict[str, str] | None = None,
    compute_histogram: bool = True,
    histogram_bins: int = 6000,
    **extra_attrs: Any,
) -> None:
    """Write a :any:`Dataset` to netcdf format.

    :param dataset: The :any:`Dataset` to write.
    :param path: Path to write the netcdf file or directory (if splitting).
    :param datatype: The data type identifier. If None, attempts to infer from dataset.
    :param dataset_id: Unique identifier for the dataset. Auto-generated if not provided.
    :param max_file_size_mb: Maximum file size in MB. If specified and data exceeds this,
        it will be split along the z-axis. If None, writes a single file.
    :param compression_level: NetCDF compression level (0-9). Default is 2.
    :param history: Dictionary of history entries to add. Keys should be identifiers,
        values are history strings.
    :param compute_histogram: Whether to compute and include data histogram. Default True.
    :param histogram_bins: Number of histogram bins if computing. Default 6000.
    :param extra_attrs: Additional global attributes to include.
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

    # Prepare data
    data_array = dataset.data

    # Convert to storage dtype (may need to handle signed/unsigned conversion)
    storage_dtype = datatype._dtype_uncorrected
    data_array = data_array.astype(storage_dtype)  # type: ignore[no-untyped-call]

    shape = data_array.shape
    zdim, ydim, xdim = shape

    # Compute histogram if requested
    histogram_data: np.ndarray[Any, Any] | None = None
    histogram_attrs: dict[str, Any] = {}
    if compute_histogram:
        computed_data: np.ndarray[Any, Any] = data_array.compute()  # type: ignore[no-untyped-call]
        data_min = float(np.min(computed_data))
        data_max = float(np.max(computed_data))
        histogram, _ = np.histogram(
            computed_data, bins=histogram_bins, range=(data_min, data_max)
        )
        histogram_data = histogram.astype(np.float64)
        histogram_attrs = {
            "data_min_max": np.array([data_min, data_max], dtype=np.float64),
            "data_histogram_binsize": (data_max - data_min) / histogram_bins,
            "data_histogram_offset": data_min,
        }

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
                history,
                histogram_data,
                histogram_attrs,
            )
            return

    # Write single file
    _write_single_netcdf(
        data_array,
        path,
        datatype,
        common_attrs,
        compression_level,
        history,
        histogram_data,
        histogram_attrs,
    )


def _write_single_netcdf(
    data_array: da.Array,
    path: Path,
    datatype: DataType,
    common_attrs: dict[str, Any],
    compression_level: int,
    history: dict[str, str] | None,
    histogram_data: np.ndarray | None,
    histogram_attrs: dict[str, Any],
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
        ncfile.createDimension("md5sums_dim", 16)

        # Set global attributes
        ncfile.setncattr("zdim_total", zdim)
        ncfile.setncattr("number_of_files", 1)
        ncfile.setncattr("zdim_range", np.array([0, zdim - 1], dtype=np.int32))

        for key, value in common_attrs.items():
            ncfile.setncattr(key, value)

        # Add history attributes
        if history:
            for hist_key, hist_value in history.items():
                ncfile.setncattr(f"history_{hist_key}", hist_value)

        # Add histogram attributes
        for key, value in histogram_attrs.items():
            ncfile.setncattr(key, value)

        # Create MD5 checksum variable (placeholder)
        md5_var = ncfile.createVariable(
            "md5_checksum", "i1", ("md5sums_dim",), zlib=False
        )
        md5_var[:] = np.zeros(16, dtype=np.int8)

        # Create main data variable
        storage_dtype_nc = _numpy_to_netcdf_dtype(np.dtype(datatype._dtype_uncorrected))
        data_var = ncfile.createVariable(
            datatype_str,
            storage_dtype_nc,
            (f"{datatype_str}_zdim", f"{datatype_str}_ydim", f"{datatype_str}_xdim"),
            zlib=True,
            complevel=compression_level,
        )

        # Write data in chunks to avoid memory issues
        computed_data: np.ndarray[Any, Any] = data_array.compute()  # type: ignore[no-untyped-call]
        data_var[:] = computed_data

        # Create histogram variable if present
        if histogram_data is not None:
            ncfile.createDimension("data_histogram_dim", len(histogram_data))
            hist_var = ncfile.createVariable(
                "data_histogram", "f8", ("data_histogram_dim",), zlib=False
            )
            hist_var[:] = histogram_data


def _write_split_netcdf(
    data_array: da.Array,
    base_path: Path,
    datatype: DataType,
    slices_per_file: int,
    common_attrs: dict[str, Any],
    compression_level: int,
    history: dict[str, str] | None,
    histogram_data: np.ndarray | None,
    histogram_attrs: dict[str, Any],
) -> None:
    """Write split netcdf files into a directory."""
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

    # Compute data once
    computed_data: np.ndarray[Any, Any] = data_array.compute()  # type: ignore[no-untyped-call]

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
            ncfile.createDimension("md5sums_dim", 16)

            # Set global attributes
            ncfile.setncattr("zdim_total", zdim)
            ncfile.setncattr("number_of_files", num_files)
            ncfile.setncattr("zdim_range", np.array([z_start, z_end], dtype=np.int32))

            for key, value in common_attrs.items():
                ncfile.setncattr(key, value)

            # Only first block gets history and histogram
            if block_idx == 0:
                if history:
                    for hist_key, hist_value in history.items():
                        ncfile.setncattr(f"history_{hist_key}", hist_value)

                for key, value in histogram_attrs.items():
                    ncfile.setncattr(key, value)

            # Create MD5 checksum variable (placeholder)
            md5_var = ncfile.createVariable(
                "md5_checksum", "i1", ("md5sums_dim",), zlib=False
            )
            md5_var[:] = np.zeros(16, dtype=np.int8)

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

            # Write block data
            block_data = computed_data[z_start : z_end + 1, :, :]
            data_var[:] = block_data

            # Create histogram variable only in first block
            if block_idx == 0 and histogram_data is not None:
                ncfile.createDimension("data_histogram_dim", len(histogram_data))
                hist_var = ncfile.createVariable(
                    "data_histogram", "f8", ("data_histogram_dim",), zlib=False
                )
                hist_var[:] = histogram_data


def _numpy_to_netcdf_dtype(dtype: np.dtype) -> str:
    """Convert numpy dtype to netCDF4 dtype string."""
    dtype = np.dtype(dtype)
    if dtype == np.int8:
        return "i1"
    elif dtype == np.int16:
        return "i2"
    elif dtype == np.int32:
        return "i4"
    elif dtype == np.int64:
        return "i8"
    elif dtype == np.uint8:
        return "u1"
    elif dtype == np.uint16:
        return "u2"
    elif dtype == np.uint32:
        return "u4"
    elif dtype == np.uint64:
        return "u8"
    elif dtype == np.float16:
        return "f2"
    elif dtype == np.float32:
        return "f4"
    elif dtype == np.float64:
        return "f8"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
