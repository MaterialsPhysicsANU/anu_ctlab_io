from abc import ABC, abstractmethod, abstractproperty
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, Self

import dask.array as da
import numpy as np

from anu_ctlab_io._datatype import DataType, StorageDType
from anu_ctlab_io._voxel_properties import VoxelUnit


class AbstractDataset(ABC):
    @classmethod
    @abstractmethod
    def from_path(
        cls, path: Path, *, parse_history: bool = True, **kwargs: Any
    ) -> Self:
        pass

    @abstractproperty
    def voxel_size(self) -> tuple[np.float32, np.float32, np.float32]:
        pass

    @abstractproperty
    def voxel_unit(self) -> VoxelUnit:
        pass

    @abstractproperty
    def dimension_names(self) -> tuple[str, ...]:
        pass

    @abstractproperty
    def history(self) -> dict[Any, Any] | str:
        pass

    @abstractproperty
    def mask_value(self) -> StorageDType | None:
        pass

    @abstractproperty
    def data(self) -> da.Array:
        pass


class DatasetFormatException(Exception):
    pass


class Dataset(AbstractDataset):
    """A :any:`Dataset`, containing the data and metadata read from one of the ANU CTLab file formats.

    :any:`Dataset`\\ s are the primary interface to the :py:mod:`anu_ctlab_io` package, and should generally be
    constructed by users via the :any:`Dataset.from_path` classmethod.

    The initializer of this class should only be used when manually constructing a :any:`Dataset`, which is not
    the primary usage of this library.
    """

    _data: da.Array
    _datatype: DataType
    _voxel_unit: VoxelUnit
    _voxel_size: tuple[np.float32, np.float32, np.float32]
    _history: dict[Any, Any] | str

    def __init__(
        self,
        data: da.Array,
        *,
        dimension_names: tuple[str, ...],
        voxel_unit: VoxelUnit,
        voxel_size: tuple[np.float32, np.float32, np.float32],
        datatype: DataType,
        history: dict[str, Any],
    ) -> None:
        """
        Manually constructs a :any:`Dataset`.

        :param data: The data contained in the :any:`Dataset`.
        :param dimension_names: The names of the dimensions of the :any:`Dataset`.
        :param voxel_unit: The unit the `voxel_size` is in terms of.
        :param voxel_size: The size of each voxel in the :any:`Dataset`.
        :param datatype: The mango datatype of the data. This is an implementation detail only required for parsing NetCDF files.
        :param history: The history of the :any:`Dataset`.
        """
        self._data = data
        self._dimension_names = dimension_names
        self._datatype = datatype
        self._voxel_unit = voxel_unit
        self._voxel_size = voxel_size
        self._history = history

    @staticmethod
    def _import_with_extra(module: str, extra: str) -> ModuleType:
        try:
            return import_module(module)
        except ImportError as e:
            raise ImportError(
                f"{module} is missing. Please install with the '{extra}' extra: pip install anu-ctlab-io[{extra}]"
            ) from e

    @classmethod
    def from_path(
        cls,
        path: Path | str,
        *,
        filetype: str = "auto",
        parse_history: bool = True,
        **kwargs: Any,
    ) -> "Dataset":
        """Creates a :any:`Dataset` from the data at the given ``path``.

        The data at ``path`` must be in one of the ANU mass data storage formats, and the optional dependencies required for the specific
        file format must be installed.

        :param path: The ``path`` to read data from.
        :rtype: :any:`Dataset`
        """
        if isinstance(path, str):
            path = Path(path)

        match filetype:
            case "NetCDF":
                netcdf_mod = cls._import_with_extra("anu_ctlab_io.netcdf", "netcdf")
                return netcdf_mod.dataset_from_netcdf(  # type: ignore[no-any-return]
                    path, parse_history=parse_history, **kwargs
                )
            case "zarr":
                zarr_mod = cls._import_with_extra("anu_ctlab_io.zarr", "zarr")
                return zarr_mod.dataset_from_zarr(  # type: ignore[no-any-return]
                    path, parse_history=parse_history, **kwargs
                )
            case "auto":
                if path.name[-2:] == "nc":
                    netcdf_mod = cls._import_with_extra("anu_ctlab_io.netcdf", "netcdf")
                    return netcdf_mod.dataset_from_netcdf(  # type: ignore[no-any-return]
                        path, parse_history=parse_history, **kwargs
                    )

                if path.name[-4:] == "zarr":
                    zarr_mod = cls._import_with_extra("anu_ctlab_io.zarr", "zarr")
                    return zarr_mod.dataset_from_zarr(  # type: ignore[no-any-return]
                        path, parse_history=parse_history, **kwargs
                    )

        raise (
            DatasetFormatException(
                "Unable to construct Dataset from given `path`, perhaps specify `filetype`?",
                path,
            )
        )

    @property
    def voxel_size(self) -> tuple[np.float32, np.float32, np.float32]:
        """The voxel size of the data."""
        return self._voxel_size

    @property
    def voxel_unit(self) -> VoxelUnit:
        """The unit the data's voxel size is in."""
        return self._voxel_unit

    @property
    def dimension_names(self) -> tuple[str, ...]:
        """The names of the data's dimensions. Usually ``("z", "y", "x")``."""
        return self._dimension_names

    @property
    def history(self) -> dict[Any, Any] | str:
        """The history metadata associated with the :any:`Dataset`.

        If parsing is enabled this will be a nested dict, otherwise it will be a dictionary
        without any guaranteed structure."""
        return self._history

    @property
    def mask_value(self) -> StorageDType | None:
        """The mask value being used by the data."""
        return self._datatype.mask_value

    @property
    def data(self) -> da.Array:
        """The data contained within the :any:`Dataset`.

        This is a `Dask Array <https://docs.dask.org/en/stable/array.html>`_."""
        return self._data


AbstractDataset.register(Dataset)
