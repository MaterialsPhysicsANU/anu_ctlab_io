"""Read and write data from/to the ANU CTLab netcdf data format.

This is an optional extra module, and must be explicitly installed to be used (e.g., ``pip install anu_ctlab_io[netcdf]``)."""

import importlib.util
import os
import re
from pathlib import Path
from typing import Any

import dask
import dask.array as da
import numpy as np
import xarray as xr

from anu_ctlab_io._dataset import Dataset
from anu_ctlab_io._datatype import DataType
from anu_ctlab_io._parse_history import parse_history
from anu_ctlab_io._voxel_properties import VoxelUnit

if importlib.util.find_spec("h5netcdf") is None:
    raise ImportError(
        "h5netcdf is required. Install it with: pip install anu_ctlab_io[netcdf]"
    )

from anu_ctlab_io.netcdf._writer import dataset_to_netcdf

__all__ = [
    "dataset_from_netcdf",
    "dataset_to_netcdf",
]


def dataset_from_netcdf(
    path: Path, *, parse_history: bool = True, **kwargs: Any
) -> Dataset:
    """Loads a :any:`Dataset` from the path to a netcdf.

    This method is used by :any:`Dataset.from_path`, by preference call that constructor directly.

    :param Path: The path to the netcdf or directory of split netcdf blocks to be loaded.
    :param parse_history: Whether to parse the history of the netcdf file. Defaults to ``True``, but disableable because the parser is currently not guaranteed to succeed.
    :param kwargs: Currently this method consumes no kwargs, but will pass provided kwargs to ``Xarray.open_mfdataset``.
    :raises lark.exceptions.UnexpectedInput: Raised if ``parse_history=True`` and the parser fails to parse the specific history provided."""
    datatype = DataType.infer_from_path(path)
    norm_path = os.path.normpath(os.path.expanduser(path))

    if os.path.isdir(norm_path):
        possible_files = [os.path.join(norm_path, p) for p in os.listdir(norm_path)]
        files = sorted(filter(os.path.isfile, possible_files))
        try:
            return _dataset_from_h5py_dir(files, datatype, parse_history)
        except (OSError, KeyError):
            pass
        # Fallback: xarray with h5netcdf → netCDF4
        xr_ds = _read_netcdf_dir_xarray(norm_path, files, datatype, **kwargs)
    else:
        try:
            return _dataset_from_h5py_single(norm_path, datatype, parse_history)
        except (OSError, KeyError):
            pass
        # Fallback: xarray with h5netcdf → netCDF4
        xr_ds = _read_netcdf_single(norm_path, datatype, **kwargs)

    return _dataset_from_xr(xr_ds, datatype, parse_history)


def _dataset_from_h5py_dir(
    files: list[str], datatype: DataType, parse_history_p: bool
) -> Dataset:
    """Fast path: read a multi-file NetCDF directory using h5py directly.

    Only reads the HDF5 header of each file to discover z-sizes; actual data
    is deferred into per-file dask delayed tasks.
    """
    import h5py  # type: ignore[import-not-found]

    var_name = str(datatype)

    # Metadata from the first file only; capture its z-size while the file is open
    with h5py.File(files[0], "r") as f:
        var = f[var_name]
        storage_dtype = var.dtype
        yx_shape = var.shape[1:]  # (y, x)
        raw_attrs = dict(f.attrs)
        first_z = int(var.shape[0])

    z_sizes = [first_z] + [_h5_z_size(fp, var_name) for fp in files[1:]]

    # Build one dask slab per file; data is not read until compute()
    def _read_slab(fpath: str) -> np.ndarray:
        with h5py.File(fpath, "r") as f:
            return f[var_name][:]  # type: ignore[no-any-return]

    slabs = [
        da.from_delayed(  # type: ignore[no-untyped-call]
            dask.delayed(_read_slab)(fp),  # type: ignore[attr-defined]
            shape=(z_sz, *yx_shape),
            dtype=storage_dtype,
        )
        for fp, z_sz in zip(files, z_sizes, strict=False)
    ]

    return _finish_h5py_dataset(
        slabs, raw_attrs, storage_dtype, datatype, parse_history_p
    )


def _dataset_from_h5py_single(
    fpath: str, datatype: DataType, parse_history_p: bool
) -> Dataset:
    """Fast path: read a single NetCDF file using h5py directly."""
    import h5py

    var_name = str(datatype)

    with h5py.File(fpath, "r") as f:
        var = f[var_name]
        storage_dtype = var.dtype
        shape = var.shape
        native_chunks = var.chunks  # None if dataset is not chunked
        raw_attrs = dict(f.attrs)

    z_total, y_size, x_size = shape
    z_chunk = native_chunks[0] if native_chunks is not None else z_total

    def _read_slab(fp: str, z_start: int, z_end: int) -> np.ndarray:
        with h5py.File(fp, "r") as f:
            return f[var_name][z_start:z_end]  # type: ignore[no-any-return]

    slabs = [
        da.from_delayed(  # type: ignore[no-untyped-call]
            dask.delayed(_read_slab)(fpath, z_start, min(z_start + z_chunk, z_total)),  # type: ignore[attr-defined]
            shape=(min(z_chunk, z_total - z_start), y_size, x_size),
            dtype=storage_dtype,
        )
        for z_start in range(0, z_total, z_chunk)
    ]

    return _finish_h5py_dataset(
        slabs, raw_attrs, storage_dtype, datatype, parse_history_p
    )


def _finish_h5py_dataset(
    slabs: list[da.Array],
    raw_attrs: dict[str, Any],
    storage_dtype: np.dtype,
    datatype: DataType,
    parse_history_p: bool,
) -> Dataset:
    data = da.concatenate(slabs, axis=0)  # type: ignore[no-untyped-call]
    if storage_dtype != datatype.dtype:
        data = data.astype(datatype.dtype)
    processed_attrs = _update_attrs(_decode_h5_attrs(raw_attrs), parse_history_p)
    return Dataset(
        data=data,
        dimension_names=("z", "y", "x"),
        datatype=datatype,
        voxel_unit=VoxelUnit.from_str(processed_attrs["voxel_unit"]),
        voxel_size=processed_attrs["voxel_size"],
        history=processed_attrs.get("history", {}),
        dataset_id=processed_attrs.get("dataset_id"),
    )


def _h5_z_size(fpath: str, var_name: str) -> int:
    import h5py

    with h5py.File(fpath, "r") as f:
        return int(f[var_name].shape[0])


def _decode_h5_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    """Decode bytes attribute values from h5py to str, matching xarray behaviour."""
    import h5py

    result = {}
    for k, v in attrs.items():
        if isinstance(v, h5py.Empty):
            result[k] = ""
        elif isinstance(v, bytes):
            result[k] = v.decode("ascii", errors="replace")
        else:
            result[k] = v
    return result


def _dataset_from_xr(
    xr_ds: xr.Dataset, datatype: DataType, parse_history_p: bool
) -> Dataset:
    xr_ds = xr_ds.rename(_transform_data_vars(xr_ds, datatype))
    xr_ds["data"] = xr_ds.data.astype(datatype.dtype)
    xr_ds.attrs = _update_attrs(xr_ds.attrs, parse_history_p)
    return Dataset(
        data=xr_ds.data.data,
        dimension_names=tuple(map(str, xr_ds.data.dims)),
        datatype=datatype,
        voxel_unit=VoxelUnit.from_str(xr_ds.attrs["voxel_unit"]),
        voxel_size=xr_ds.attrs["voxel_size"],
        history=xr_ds.history,
        dataset_id=xr_ds.attrs.get("dataset_id"),
    )


def _transform_data_vars(dataset: xr.Dataset, datatype: DataType) -> dict[str, str]:
    attr_transform = {f"{datatype}_{dim}dim": dim for dim in ["x", "y", "z"]}
    for k in dataset.data_vars.keys():
        match k:
            case a if isinstance(a, str) and a.find(str(datatype)) == 0:
                attr_transform[k] = "data"
    return attr_transform


def _update_attrs(attrs: dict[str, Any], parse_history_p: bool) -> dict[str, Any]:
    new_attrs: dict[str, Any] = {"history": {}}
    for k, v in attrs.items():
        match k:
            case a if a.find("history") == 0:
                if parse_history_p:
                    new_attrs["history"][k[len("history") + 1 :]] = parse_history(v)
                else:
                    new_attrs["history"][k[len("history") + 1 :]] = v
            case a if a.find("dim") != -1:
                new_attrs[re.sub("([x|y|z])dim", "\\1", k)] = v
            case a if a in ["number_of_files", "zdim_total", "total_grid_size_xyz"]:
                pass
            case a if a.find("_xyz"):
                new_attrs[re.sub("(.*)_xyz", "\\1", k)] = v
            case _:
                new_attrs[k] = attrs[k]
    return new_attrs


def _read_netcdf_dir_xarray(
    norm_path: str, files: list[str], datatype: DataType, **kwargs: Any
) -> xr.Dataset:
    last_exc: OSError | None = None
    for engine in _read_engines():
        try:
            return xr.open_mfdataset(
                files,
                engine=engine,
                combine="nested",
                concat_dim=[f"{datatype}_zdim"],
                combine_attrs="drop_conflicts",
                coords="minimal",
                compat="override",
                mask_and_scale=False,
                data_vars=[f"{datatype}"],
                **kwargs,
            )
        except OSError as e:
            last_exc = e
    raise OSError(
        f"Could not read netCDF files in {norm_path} with any available engine."
    ) from last_exc


def _read_netcdf_single(
    norm_path: str, datatype: DataType, **kwargs: Any
) -> xr.Dataset:
    chunks = kwargs.pop("chunks", -1)
    last_exc: OSError | None = None
    for engine in _read_engines():
        try:
            return xr.open_dataset(
                norm_path,
                engine=engine,
                mask_and_scale=False,
                chunks=chunks,
                **kwargs,
            )
        except OSError as e:
            last_exc = e
    raise OSError(
        f"Could not read netCDF file {norm_path} with any available engine."
    ) from last_exc


# Maybe netCDF3 support becomes optional at some point?
def _read_engines() -> list[str]:
    engines = ["h5netcdf"]
    if importlib.util.find_spec("netCDF4") is not None:
        engines.append("netCDF4")
    return engines
