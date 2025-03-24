from enum import Enum, auto
from dataclasses import dataclass
from typing import Self
import os
import re

from anu_ctlab_io.netcdf.parse_history import parse_history
import xarray as xr
import dictdiffer
import hidefix  # noqa: F401
import numpy as np


type VoxelSize = tuple[np.float32, np.float32, np.float32]


class VoxelUnit(Enum):
    """
    The unit of size of a voxel. NOTE: Consider parameterizing this?
    TODO: double check mango supported units
    """

    meter = auto()
    millimeter = auto()
    micrometer = auto()
    nanometer = auto()
    voxels = auto()

    @classmethod
    def from_str(cls, string: str):
        return {
            # short names
            "m": cls.meter,
            "mm": cls.millimeter,
            "um": cls.micrometer,
            "nm": cls.nanometer,
            # long names
            "meter": cls.meter,
            "millimeter": cls.millimeter,
            "micrometer": cls.micrometer,
            "nanometer": cls.nanometer,
            # alternative symbols
            "µm": cls.micrometer,
        }.get(string, cls.voxels)


type storage_dtypes = (
    np.uint8 | np.uint16 | np.uint32 | np.uint64 | np.float16 | np.float32 | np.float64
)


@dataclass
class DataTypeProperties:
    """
    The properties of a DataType in the ANU NetCDF format.
    """

    discrete: bool
    dtype: type
    dtype_uncorrected: type
    mask_value: int | float | None


DATATYPE_PROPERTIES: dict[str, DataTypeProperties] = {
    "proju16": DataTypeProperties(False, np.uint16, np.int16, None),
    "projf32": DataTypeProperties(False, np.float32, np.float32, None),
    "tomo_float": DataTypeProperties(False, np.float32, np.float32, 1.0e30),
    "tomo": DataTypeProperties(False, np.uint16, np.int16, 65535),
    "float16": DataTypeProperties(False, np.float16, np.float16, None),
    "float64": DataTypeProperties(False, np.float64, np.float64, 1.0e300),
    "segmented": DataTypeProperties(True, np.uint8, np.int8, 255),
    "distance_map": DataTypeProperties(False, np.float32, np.float32, -2.0),
    "labels": DataTypeProperties(True, np.int32, np.int32, 2147483647),
    "rgba8": DataTypeProperties(False, np.uint8, np.int8, None),
}


class DataType(Enum):
    proju16 = auto()
    projf32 = auto()
    # tomo_float is above tomo, to ensure it is checked first when iterating over DataType
    tomo_float = auto()
    tomo = auto()
    float16 = auto()
    float64 = auto()
    segmented = auto()
    distance_map = auto()
    labels = auto()
    rgba8 = auto()

    def __str__(self) -> str:
        return self.__dict__["_name_"]

    @property
    def is_discrete(self) -> bool:
        return DATATYPE_PROPERTIES[str(self)].discrete

    @property
    def dtype(self) -> type:
        return DATATYPE_PROPERTIES[str(self)].dtype

    @property
    def _dtype_uncorrected(self) -> type:
        return DATATYPE_PROPERTIES[str(self)].dtype_uncorrected

    def _mask_value(self, uncorrected: bool = False) -> storage_dtypes | None:
        idx = "dtype_uncorrected" if uncorrected else "dtype"
        props = DATATYPE_PROPERTIES[str(self)]
        if props.__dict__[idx] is None:
            return None
        return props.__dict__[idx](
            np.array(props.mask_value).astype(
                props.__dict__[idx]
            )  # req'd to do overflows as desired
        )

    @property
    def mask_value(self):
        return self._mask_value()

    @property
    def _mask_value_uncorrected(self):
        return self._mask_value(True)

    @classmethod
    def infer_from_path(cls, path: os.PathLike) -> Self:
        basename = os.path.basename(os.path.normpath(path)).removeprefix("cntr_")
        for data_type in DataType:
            if basename.startswith(str(data_type)):
                return cls[str(data_type)]
        raise RuntimeError("File datatype not recognised from name.")

    @classmethod
    def from_basename(cls, basename: str) -> Self:
        try:
            return cls[basename]
        except KeyError as e:
            raise RuntimeError(f"Basename {basename} not recognized.", e)

    # @classmethod
    # def from_dataset(cls, dataset: xr.Dataset) -> Self: ...


def update_attrs(attrs: dict) -> dict:
    new_attrs: dict = {"history": {}}
    for k, v in attrs.items():
        match k:
            case a if a.find("history") == 0:
                new_attrs["history"][k[len("history") + 1 :]] = parse_history(v)
            case a if a.find("dim") != -1:
                new_attrs[re.sub("([x|y|z])dim", "\\1", k)] = v
            case a if a in ["number_of_files", "zdim_total", "total_grid_size_xyz"]:
                pass
            case _:
                new_attrs[k] = attrs[k]
    return new_attrs


def transform_data_vars(dataset: xr.Dataset, datatype: DataType) -> dict[str, str]:
    attr_transform = {f"{datatype}_{dim}dim": dim for dim in ["x", "y", "z"]}
    for k in dataset.data_vars.keys():
        match k:
            case a if a.find(str(datatype)) == 0:
                attr_transform[k] = "data"
    return attr_transform


class CTLabDataset:
    # _voxel_unit: VoxelUnit
    # _voxel_size: VoxelSize
    _dataType: DataType
    _dataset: xr.Dataset
    _attr_diff: dict
    _data_var_tx: dict[str, str]

    def __init__(self, dataset: xr.Dataset, dataType: DataType) -> None:
        self._dataset = dataset
        self._dataType = dataType
        self._transform_from_anunetcdf_format()

    def __repr__(
        self,
    ):  # TODO: this is useful but a misleading representation of the class, fix
        return "<CTLabDataset>" + self._dataset.__repr__()

    def _transform_from_anunetcdf_format(self):
        new_attrs = update_attrs(self._dataset.attrs)
        self._attr_diff = dictdiffer.diff(self._dataset.attrs, new_attrs)
        self._dataset.attrs.update(new_attrs)

        self._data_var_tx = transform_data_vars(self._dataset, self._dataType)
        self._dataset = self._dataset.rename(self._data_var_tx)
        self._dataset["data"] = self._dataset.data.astype(self._dataType.dtype)

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
    def from_netcdf(cls, path: os.PathLike):
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
                engine="hidefix",
            )
        else:
            dataset = xr.open_dataset(path, mask_and_scale=False, engine="hidefix")
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
    def history(self) -> dict[str, str | dict] | None:
        return self._dataset.attrs["history"]

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
