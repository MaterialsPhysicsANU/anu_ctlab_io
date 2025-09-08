from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path
from typing import Any

import dask.array as da
import deprecation  # type: ignore
import xarray as xr


from anu_ctlab_io.datatype import DataType, StorageDType
from anu_ctlab_io.version import version
from anu_ctlab_io.voxel_properties import VoxelUnit, VoxelSize


class AbstractDataset(ABC):
    @classmethod
    @abstractmethod
    def from_path(cls, path: Path, *, parse_history=True, **kwargs):
        pass

    @abstractproperty
    def voxel_size(self) -> VoxelSize:
        pass

    @abstractproperty
    def voxel_unit(self) -> VoxelUnit:
        pass

    @abstractproperty
    def dimension_names(self) -> tuple[str, ...]:
        pass

    @abstractproperty
    def history(self) -> dict:
        pass

    @abstractproperty
    def mask_value(self) -> StorageDType | None:
        pass

    @abstractproperty
    def data(self) -> da.Array:
        pass

    @abstractmethod
    def as_xarray_dataarray(self) -> xr.DataArray:
        pass

    @abstractmethod
    def as_xarray_dataset(self) -> xr.Dataset:
        pass


class DatasetFormatException(Exception):
    pass


class Dataset(AbstractDataset):
    _data: da.Array
    _datatype: DataType
    _voxel_unit: VoxelUnit
    _voxel_size: VoxelSize
    _history: dict[str, Any]

    def __init__(
        self,
        data: da.Array,
        *,
        dimension_names: tuple[str, ...],
        voxel_unit: VoxelUnit,
        voxel_size: VoxelSize,
        datatype: DataType,
        history: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        self._data = data
        self._dimension_names = dimension_names
        self._datatype = datatype
        self._voxel_unit = voxel_unit
        self._voxel_size = voxel_size
        self._history = history

    @classmethod
    def from_path(
        cls,
        path: Path | str,
        *,
        filetype: str = "auto",
        parse_history: bool = True,
        **kwargs: Any,
    ):
        # function level imports avoid the circular dependancy
        from anu_ctlab_io.netcdf import dataset_from_netcdf
        from anu_ctlab_io.zarr import dataset_from_zarr

        if isinstance(path, str):
            path = Path(path)

        match filetype:
            case "NetCDF":
                return dataset_from_netcdf(path, parse_history=parse_history, **kwargs)
            case "zarr":
                return dataset_from_zarr(path, parse_history=parse_history, **kwargs)
            case "auto":
                if path.name[-2:] == "nc":
                    return dataset_from_netcdf(
                        path, parse_history=parse_history, **kwargs
                    )

                if path.name[-4:] == "zarr":
                    return dataset_from_zarr(
                        path, parse_history=parse_history, **kwargs
                    )

                else:
                    raise (
                        DatasetFormatException(
                            "Unable to construct Dataset from given `path`, perhaps specify `filetype`?",
                            path,
                        )
                    )

    @property
    def voxel_size(self) -> VoxelSize:
        return self._voxel_size

    @property
    def voxel_unit(self) -> VoxelUnit:
        return self._voxel_unit

    @property
    def dimension_names(self) -> tuple[str, ...]:
        return self._dimension_names

    @property
    def history(self) -> dict:
        return self._history

    @property
    def mask_value(self) -> StorageDType | None:
        return self._datatype.mask_value

    @property
    def data(self) -> da.Array:
        return self._data

    @deprecation.deprecated(
        deprecated_in="0.2",
        removed_in="1.0",
        current_version=version,
        details="Used `Dataset.data` to access a dask array. Xarray support is being removed.",
    )
    def as_xarray_dataarray(self) -> xr.DataArray:
        xa = xr.DataArray(self.data, dims={"z", "y", "x"})
        return xa

    @deprecation.deprecated(
        deprecated_in="0.2",
        removed_in="1.0",
        current_version=version,
        details="Used `Dataset.data` to access a dask array. Xarray support is being removed.",
    )
    def as_xarray_dataset(self) -> xr.Dataset:
        xa = xr.DataArray(self.data, dims={"z", "y", "x"})
        ds = xr.Dataset({"data": xa})
        return ds


AbstractDataset.register(Dataset)
