from typing import Self
import os
import re

from anu_ctlab_io.netcdf.parse_history import parse_history
from anu_ctlab_io import VoxelUnit, VoxelSize, DataType, storage_dtypes
import xarray as xr
import dictdiffer


__all__ = ["NetCDFDataset"]


def _transform_data_vars(dataset: xr.Dataset, datatype: DataType) -> dict[str, str]:
    attr_transform = {f"{datatype}_{dim}dim": dim for dim in ["x", "y", "z"]}
    for k in dataset.data_vars.keys():
        match k:
            case a if a.find(str(datatype)) == 0:
                attr_transform[k] = "data"
    return attr_transform


class NetCDFDataset:
    # _voxel_unit: VoxelUnit
    # _voxel_size: VoxelSize
    _dataType: DataType
    _dataset: xr.Dataset
    _attr_diff: dict
    _data_var_tx: dict[str, str]
    _parse_history: bool

    def __init__(self, dataset: xr.Dataset, dataType: DataType, *, preserve_history = True) -> None:
        self._dataset = dataset
        self._dataType = dataType
        self._parse_history = preserve_history
        self._transform_from_anunetcdf_format()

    def __repr__(
        self,
    ):  # TODO: this is useful but a misleading representation of the class, fix
        return "<CTLabDataset>" + self._dataset.__repr__()

    def _transform_from_anunetcdf_format(self):
        new_attrs = _update_attrs(self._dataset.attrs)
        self._attr_diff = dictdiffer.diff(self._dataset.attrs, new_attrs)
        self._dataset.attrs.update(new_attrs)

        self._data_var_tx = _transform_data_vars(self._dataset, self._dataType)
        self._dataset = self._dataset.rename(self._data_var_tx)
        self._dataset["data"] = self._dataset.data.astype(self._dataType.dtype)

    def _update_attrs(self, attrs: dict) -> dict:
        new_attrs: dict = {"history": {}}
        for k, v in attrs.items():
            match k:
                case a if a.find("history") == 0:
                    if self._parse_history:
                        new_attrs["history"][k[len("history") + 1 :]] = parse_history(v)
                    else:
                        new_attrs["history"][k[len("history") + 1 :]] = v
                case a if a.find("dim") != -1:
                    new_attrs[re.sub("([x|y|z])dim", "\\1", k)] = v
                case a if a in ["number_of_files", "zdim_total", "total_grid_size_xyz"]:
                    pass
                case _:
                    new_attrs[k] = attrs[k]
        return new_attrs

    def _restore_to_anunetcdf_format(self) -> xr.Dataset:
        """Note this returns a new dataset rather than mutating the CTLabDataset instance."""
        restored_dataset = self._dataset.rename(
            {v: k for k, v in self._data_var_tx.items()}
        )
        restored_dataset.attrs.update(
            dictdiffer.revert(restored_dataset.attrs, self._attr_diff)
        )
        restored_dataset[str(self._dataType)] = restored_dataset[
            str(self._dataType)
        ].astype(self._dataType._dtype_uncorrected)

        return restored_dataset

    @classmethod
    def from_path(cls, path: os.PathLike, **kwargs):
        path = os.path.normpath(os.path.expanduser(path))
        dataType = DataType.infer_from_path(path)
        if os.path.isdir(path):
            possible_files = [os.path.join(path, p) for p in os.listdir(path)]
            files = sorted(list(filter(os.path.isfile, possible_files)))
            dataset = xr.open_mfdataset(
                files,
                combine="nested",
                concat_dim=[f"{dataType}_zdim"],
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
        return cls(dataset, dataType)

    @classmethod
    # TODO: properly infer datatype!
    def from_xarray_dataset(cls, dataset: xr.Dataset) -> Self:
        return cls(dataset, DataType[dataset.attrs["DataType"]])

    # @classmethod
    # def from_array(
    #     cls,
    #     array: xr.DataArray | np.ndarray,
    #     data_type: DataType,
    #     voxel_size: VoxelSize,
    #     voxel_unit: VoxelUnit,
    #     history: dict[str, str],
    # ) -> Self: ...

    @property
    def voxel_size(self) -> VoxelSize:
        return self._dataset.attrs["voxel_size"]

    @property
    def voxel_unit(self) -> VoxelUnit:
        return self._dataset.attrs["voxel_unit"]

    @property
    def history(self) -> dict:
        return self._dataset.attrs.get("history", {})

    @property
    def mask_value(self) -> storage_dtypes | None:
        return self._dataType.mask_value

    def as_xarray_dataarray(self) -> xr.DataArray:
        return self._dataset.data

    def as_xarray_dataset(self) -> xr.Dataset:
        return self._dataset

    # def as_xarray_datatree(self) -> xr.DataTree:
    #    ...

    # def as_numpy_array(self) -> np.ndarray: ...

    # def save(self, path: os.PathLike, options) -> None: ...
