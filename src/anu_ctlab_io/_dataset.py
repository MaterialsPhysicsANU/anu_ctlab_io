import re
from abc import ABC, abstractmethod
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, Self, cast

import dask.array as da
import numpy as np

from anu_ctlab_io._datatype import DataType, StorageDType
from anu_ctlab_io._voxel_properties import VoxelUnit


def _extract_base_name_from_dataset_id(dataset_id: str) -> str:
    """Strip old-format timestamp from dataset_id for filename generation.

    Old format "20250314_012913_basename" → returns "basename"
    New format "0-00000_gb1" → returns "0-00000_gb1" (no match, return as-is)

    :param dataset_id: The dataset identifier
    :return: The base name without timestamp prefix
    """
    match = re.match(r"^(\d{8}_\d{6})_(.+)$", dataset_id)
    if match:
        return match.group(2)  # Everything after second underscore
    return dataset_id  # Not old format, use as-is


class AbstractDataset(ABC):
    @classmethod
    @abstractmethod
    def from_path(
        cls, path: Path, *, parse_history: bool = True, **kwargs: Any
    ) -> Self:
        pass

    @abstractmethod
    def to_path(self, path: Path, *, filetype: str = "auto", **kwargs: Any) -> None:
        pass

    @property
    @abstractmethod
    def voxel_size(self) -> tuple[np.float32, np.float32, np.float32]: ...

    @property
    @abstractmethod
    def voxel_unit(self) -> VoxelUnit: ...

    @property
    @abstractmethod
    def dimension_names(self) -> tuple[str, ...]: ...

    @property
    @abstractmethod
    def history(self) -> dict[Any, Any] | str: ...

    @property
    @abstractmethod
    def mask_value(self) -> StorageDType | None: ...

    @property
    @abstractmethod
    def data(self) -> da.Array: ...

    @property
    @abstractmethod
    def mask(self) -> da.Array: ...

    @property
    @abstractmethod
    def masked_data(self) -> da.Array: ...


class Dataset(AbstractDataset):
    """A :any:`Dataset`, containing the data and metadata read from one of the ANU CTLab file formats.

    :any:`Dataset`\\ s are the primary interface to the :py:mod:`anu_ctlab_io` package, and should generally be
    constructed by users via the :any:`Dataset.from_path` classmethod. Note that the relevant extra (:any:`netcdf` or :any:`zarr`)
    must be installed.

    The initializer of this class should only be used when manually constructing a :any:`Dataset`, which is not
    the primary usage of this library.
    """

    _data: da.Array
    _datatype: DataType | None
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
        datatype: DataType | None = None,
        history: dict[str, Any] | None = None,
        dataset_id: str | None = None,
        source_format: str | None = None,
    ) -> None:
        """
        Manually constructs a :any:`Dataset`.

        :param data: The data contained in the :any:`Dataset`.
        :param dimension_names: The names of the dimensions of the :any:`Dataset`.
        :param voxel_unit: The unit the `voxel_size` is in terms of.
        :param voxel_size: The size of each voxel in the :any:`Dataset`.
        :param datatype: The mango datatype of the data. This is an implementation detail only required for parsing NetCDF files.
        :param history: The history of the :any:`Dataset`.
        :param dataset_id: The dataset identifier from the source file.
        :param source_format: The format the dataset was read from ("netcdf" or "zarr").
        """
        if history is None:
            history = {}

        self._data = data
        self._dimension_names = dimension_names
        self._datatype = datatype
        self._voxel_unit = voxel_unit
        self._voxel_size = voxel_size
        self._history = history
        self._dataset_id = dataset_id
        self._source_format = source_format

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

        The data at ``path`` must be in one of the ANU mass data storage formats, and the optional extras required for the specific
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
            ValueError(
                "Unable to construct Dataset from given `path`, perhaps specify `filetype`?",
                path,
            )
        )

    def to_path(
        self,
        path: Path | str,
        *,
        filetype: str = "auto",
        dataset_id: str | None = "auto",
        **kwargs: Any,
    ) -> None:
        """Writes the :any:`Dataset` to the given ``path``.

        The data will be written in one of the ANU mass data storage formats, and the optional extras required for the specific
        file format must be installed.

        :param path: The ``path`` to write data to.
        :param filetype: The format to write ("NetCDF", "zarr", or "auto"). If "auto", format is inferred from path extension.
            When inferring, NetCDF is assumed for paths ending in ``.nc`` or ``_nc``, and Zarr for paths ending in ``.zarr``.
            If datatype is present in filename (e.g., "tomo_output"), NetCDF is assumed.
        :param dataset_id: Dataset identifier to write to file metadata. Options:
            - "auto" (default): Use self.dataset_id if available, otherwise generate new
            - str: Use this exact value
            - None: Generate new (legacy behavior)
        :param kwargs: Additional keyword arguments passed to the format-specific writer.
        """
        if isinstance(path, str):
            path = Path(path)

        # Handle dataset_id parameter
        if dataset_id == "auto":
            resolved_dataset_id = self._dataset_id
        else:
            resolved_dataset_id = dataset_id

        # Add resolved dataset_id to kwargs if not already present
        if "dataset_id" not in kwargs:
            kwargs["dataset_id"] = resolved_dataset_id

        match filetype:
            case "NetCDF":
                netcdf_mod = self._import_with_extra("anu_ctlab_io.netcdf", "netcdf")
                netcdf_mod.dataset_to_netcdf(self, path, **kwargs)
                return
            case "zarr":
                zarr_mod = self._import_with_extra("anu_ctlab_io.zarr", "zarr")
                zarr_mod.dataset_to_zarr(self, path, **kwargs)
                return
            case "auto":
                # Check for explicit extensions
                if path.name.endswith(".nc") or path.name.endswith("_nc"):
                    netcdf_mod = self._import_with_extra(
                        "anu_ctlab_io.netcdf", "netcdf"
                    )
                    netcdf_mod.dataset_to_netcdf(self, path, **kwargs)
                    return

                if path.name.endswith(".zarr"):
                    zarr_mod = self._import_with_extra("anu_ctlab_io.zarr", "zarr")
                    zarr_mod.dataset_to_zarr(self, path, **kwargs)
                    return

                # Check if datatype is in filename (Mango convention)
                if self._datatype is not None:
                    datatype_str = str(self._datatype)
                    if datatype_str in path.name:
                        netcdf_mod = self._import_with_extra(
                            "anu_ctlab_io.netcdf", "netcdf"
                        )
                        netcdf_mod.dataset_to_netcdf(self, path, **kwargs)
                        return

        raise ValueError(
            "Unable to determine output format from given `path`, perhaps specify `filetype`?",
            path,
        )

    def save(
        self,
        suffix: str = "_CTLAB_IO",
        format: str | None = None,
        directory: Path | str = ".",
        **kwargs: Any,
    ) -> Path:
        """Write dataset with auto-generated filename.

        Generates a filename based on the dataset_id, stripping old-format timestamps
        for cleaner paths while preserving them in file metadata for provenance.

        :param suffix: Suffix to append to the base name. Defaults to "_CTLAB_IO".
        :param format: Output format ("netcdf" or "zarr"). If None, uses source_format or defaults to "netcdf".
        :param directory: Directory to write the file to. Defaults to current directory.
        :param kwargs: Additional arguments passed to the format writer.
        :return: Path to the written file.
        :raises ValueError: If dataset_id is not available (required for auto-path generation).

        Example::

            # Old format: "20250314_012913_tomoLoRes_SS"
            # output filename: "tomoLoRes_SS_CTLAB_IO.nc"
            # dataset_id in file: "20250314_012913_tomoLoRes_SS"
            ds = Dataset.from_path("file.nc")
            output_path = ds.save()

            # New format: "0-00000_gb1"
            # Output filename: "0-00000_gb1__processed.zarr"
            ds2 = Dataset.from_path("file2.nc")
            output_path = ds2.save(suffix="__processed", format="zarr")
        """
        # Determine format
        if format is None:
            format = self._source_format or "netcdf"

        # Normalize format to match to_path expectations
        if format.lower() == "netcdf":
            filetype = "NetCDF"
        elif format.lower() == "zarr":
            filetype = "zarr"
        else:
            filetype = format

        # Get base name (strip timestamp if old format)
        if self._dataset_id is not None:
            base_name = _extract_base_name_from_dataset_id(self._dataset_id)
        else:
            raise ValueError(
                "Cannot auto-generate filename: dataset_id is not available. "
                "Use to_path() with an explicit path instead."
            )

        # Construct filename using stripped basename
        extension = ".zarr" if format.lower() == "zarr" else ".nc"
        filename = f"{base_name}{suffix}{extension}"
        output_path = Path(directory) / filename

        # Write with ORIGINAL dataset_id preserved in metadata
        # Use dataset_id="auto" to let to_path handle it
        self.to_path(output_path, filetype=filetype, **kwargs)

        return output_path

    @property
    def voxel_size(self) -> tuple[np.float32, np.float32, np.float32]:
        """The voxel size of the data in the dataset's native unit."""
        return self._voxel_size

    def voxel_size_with_unit(
        self, voxel_unit: VoxelUnit
    ) -> tuple[np.float32, np.float32, np.float32]:
        """Get the voxel size of the data converted to a target unit.

        :param voxel_unit: The unit to convert the voxel size to.
        :return: The voxel size as a tuple of three float32 values.
        :raises ValueError: If unit conversion is requested but the source or target unit is VOXEL.
        """
        if voxel_unit == self._voxel_unit:
            return self._voxel_size

        conversion_factor = self._voxel_unit._conversion_factor(voxel_unit)
        return (
            np.float32(self._voxel_size[0] * conversion_factor),
            np.float32(self._voxel_size[1] * conversion_factor),
            np.float32(self._voxel_size[2] * conversion_factor),
        )

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
        return None if self._datatype is None else self._datatype.mask_value

    @property
    def data(self) -> da.Array:
        """The data contained within the :any:`Dataset`.

        This is a `Dask Array <https://docs.dask.org/en/stable/array.html>`_."""
        return self._data

    @property
    def dataset_id(self) -> str | None:
        """The dataset identifier from the source file, if available."""
        return self._dataset_id

    @property
    def source_format(self) -> str | None:
        """The format the dataset was read from ("netcdf" or "zarr"), if available."""
        return self._source_format

    @property
    def mask(self) -> da.Array:
        """The masked areas of the :any:`Dataset`, as a boolean array.

        This has the same dimensions as the data, and will be all-zero if no mask value exists."""
        return cast(
            da.Array,
            da.zeros_like(self._data, dtype=bool)  # type: ignore [no-untyped-call]
            if self._datatype is None
            else self._data == self._datatype.mask_value,
        )

    @property
    def masked_data(self) -> da.Array:
        """The data contained within the :any:`Dataset`, as a masked array.

        This has better performance than manually creating a masked_array using `mask` in the case
        that the loaded datatype has no mask (i.e., OME-Zarr data), as it creates a masked array
        with `nomask` in these situations."""
        print(self._datatype)
        return cast(
            da.Array,
            da.ma.masked_array(self._data, mask=self.mask)
            if self._datatype is not None
            else da.ma.masked_array(self._data),
        )

    def add_to_history(self, key: str, value: dict[str, Any] | str) -> None:
        """Add an entry to the dataset's history metadata.

        This method mutates the dataset in-place by adding a new history entry.
        The history will be automatically serialized when writing to NetCDF or Zarr formats.

        :param key: The history key/identifier. Convention is to use timestamps
            (e.g., "20260128_150530_crop") but any string is valid.
        :param value: The history entry value. Can be a dict with operation details
            (recommended) or a string. Dicts will be serialized to structured format.

        Example::

            ds = Dataset.from_path("data.nc")
            ds.add_to_history("20260128_crop", {
                "operation": "crop",
                "z_range": [10, 50],
                "reason": "Focus on region of interest"
            })
            ds.to_path("cropped.nc")  # History is preserved
        """
        if not isinstance(self._history, dict):
            self._history = {}
        self._history[key] = value

    def update_history(self, entries: dict[str, dict[str, Any] | str]) -> None:
        """Update the dataset's history with multiple entries at once.

        This method mutates the dataset in-place by adding multiple history entries.
        Equivalent to calling :any:`add_to_history` multiple times.

        :param entries: Dictionary of history entries to add. Keys are history identifiers,
            values are the entry data (dicts or strings).

        Example::

            ds.update_history({
                "20260128_150530_crop": {"operation": "crop", "z_range": [10, 50]},
                "20260128_150545_filter": {"operation": "gaussian_filter", "sigma": 2.0}
            })
        """
        if not isinstance(self._history, dict):
            self._history = {}
        self._history.update(entries)

    @classmethod
    def from_modified(
        cls,
        source: "Dataset",
        *,
        data: da.Array | None = None,
        voxel_size: tuple[np.float32, np.float32, np.float32] | None = None,
        voxel_unit: VoxelUnit | None = None,
        dimension_names: tuple[str, ...] | None = None,
        datatype: DataType | None = None,
        history_entry: dict[str, Any] | str | None = None,
        history_key: str | None = None,
    ) -> "Dataset":
        """Create a new Dataset from a modified version of an existing one.

        This factory method creates a new Dataset instance with selected attributes
        modified, while preserving the rest from the source. Optionally adds a history
        entry documenting the modification. This follows an immutable pattern where
        the source dataset is not modified.

        :param source: The source Dataset to create a modified copy from.
        :param data: New data array. If None, uses source's data.
        :param voxel_size: New voxel size. If None, uses source's voxel_size.
        :param voxel_unit: New voxel unit. If None, uses source's voxel_unit.
        :param dimension_names: New dimension names. If None, uses source's dimension_names.
        :param datatype: New datatype. If None, uses source's datatype.
        :param history_entry: History entry to add documenting the modification.
            If provided, a new history entry is added with the given key.
        :param history_key: Key for the history entry. If None and history_entry is provided,
            auto-generates a timestamp-based key like "20260128_150530_modification".
        :return: New Dataset instance with modifications applied.

        Example::

            ds = Dataset.from_path("data.nc")

            # Create cropped version with automatic history
            cropped = Dataset.from_modified(
                ds,
                data=ds.data[10:50, :, :],
                history_entry={"operation": "crop", "z_range": [10, 50]},
                history_key="20260128_crop"
            )

            # Chain modifications
            scaled = Dataset.from_modified(
                cropped,
                voxel_size=(0.1, 0.1, 0.1),
                history_entry={"operation": "rescale", "new_voxel_size": [0.1, 0.1, 0.1]}
            )
        """
        from datetime import datetime

        # Copy history from source
        if isinstance(source._history, dict):
            new_history = source._history.copy()
        else:
            new_history = {}

        # Add new history entry if provided
        if history_entry is not None:
            if history_key is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                history_key = f"{timestamp}_modification"
            new_history[history_key] = history_entry

        return cls(
            data=data if data is not None else source._data,
            dimension_names=(
                dimension_names
                if dimension_names is not None
                else source._dimension_names
            ),
            voxel_unit=voxel_unit if voxel_unit is not None else source._voxel_unit,
            voxel_size=voxel_size if voxel_size is not None else source._voxel_size,
            datatype=datatype if datatype is not None else source._datatype,
            history=new_history,
            dataset_id=source._dataset_id,
            source_format=source._source_format,
        )


AbstractDataset.register(Dataset)
