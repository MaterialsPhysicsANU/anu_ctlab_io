import deprecation  # type: ignore
import importlib.util
import os
import re
from pathlib import Path

import xarray as xr

from anu_ctlab_io._dataset import Dataset
from anu_ctlab_io._datatype import DataType
from anu_ctlab_io._parse_history import parse_history
from anu_ctlab_io._version import version
from anu_ctlab_io._voxel_properties import VoxelUnit

if importlib.util.find_spec("netCDF4") is None and importlib.util.find_spec("h5netcdf") is None:
    raise ImportError("Neither netCDF4 nor h5netcdf could be imported.")

__all__ = ["NetCDFDataset", "dataset_from_netcdf"]


def dataset_from_netcdf(path: Path, *, parse_history: bool, **kwargs):
    datatype = DataType.infer_from_path(path)
    dataset = read_netcdf(path, datatype, **kwargs)
    dataset = dataset.rename(_transform_data_vars(dataset, datatype))
    dataset["data"] = dataset.data.astype(datatype.dtype)
    dataset.attrs = _update_attrs(dataset.attrs, parse_history)
    return Dataset(
        data=dataset.data.data,
        dimension_names=dataset.dims,
        datatype=datatype,
        voxel_unit=VoxelUnit.from_str(dataset.attrs["voxel_unit"]),
        voxel_size=dataset.attrs["voxel_size"],
        history=dataset.history,
    )


def _transform_data_vars(dataset: xr.Dataset, datatype: DataType) -> dict[str, str]:
    attr_transform = {f"{datatype}_{dim}dim": dim for dim in ["x", "y", "z"]}
    for k in dataset.data_vars.keys():
        match k:
            case a if a.find(str(datatype)) == 0:
                attr_transform[k] = "data"
    return attr_transform


def _update_attrs(attrs: dict, parse_history_p: bool) -> dict:
    new_attrs: dict = {"history": {}}
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


def read_netcdf(path: os.PathLike, datatype: DataType, **kwargs):
    path = os.path.normpath(os.path.expanduser(path))
    if os.path.isdir(path):
        possible_files = [os.path.join(path, p) for p in os.listdir(path)]
        files = sorted(list(filter(os.path.isfile, possible_files)))
        dataset = xr.open_mfdataset(
            files,
            combine="nested",
            concat_dim=[f"{datatype}_zdim"],
            combine_attrs="drop_conflicts",
            coords="minimal",
            compat="override",
            mask_and_scale=False,
            **kwargs,
        )
    else:
        dataset = xr.open_dataset(
            path, mask_and_scale=False, chunks=kwargs.pop("chunks", -1), **kwargs
        )
    return dataset


class NetCDFDataset(Dataset):
    @deprecation.deprecated(
        deprecated_in="0.2",
        removed_in="1.0",
        current_version=version,
        details="Use the from_path method on `Dataset` rather than `NetCDFDataset`.",
    )
    @staticmethod
    def from_path(path: Path, **kwargs):
        return Dataset.from_path(path, **kwargs)
