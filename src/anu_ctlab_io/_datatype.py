import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Self

import numpy as np

__all__ = ["StorageDType", "DataType"]


type StorageDType = (
    np.uint8 | np.uint16 | np.uint32 | np.uint64 | np.float16 | np.float32 | np.float64
)


@dataclass
class _DataTypeProperties:
    """
    The properties of a DataType in the ANU NetCDF format.
    """

    discrete: bool
    dtype: type
    dtype_uncorrected: type
    mask_value: int | float | None


_DATATYPE_PROPERTIES: dict[str, _DataTypeProperties] = {
    "proju16": _DataTypeProperties(False, np.uint16, np.int16, None),
    "projf32": _DataTypeProperties(False, np.float32, np.float32, None),
    "tomo_float": _DataTypeProperties(False, np.float32, np.float32, 1.0e30),
    "tomo": _DataTypeProperties(False, np.uint16, np.int16, 65535),
    "float16": _DataTypeProperties(False, np.float16, np.float16, None),
    "float64": _DataTypeProperties(False, np.float64, np.float64, 1.0e300),
    "segmented": _DataTypeProperties(True, np.uint8, np.int8, 255),
    "distance_map": _DataTypeProperties(False, np.float32, np.float32, -2.0),
    "labels": _DataTypeProperties(True, np.int32, np.int32, 2147483647),
    "rgba8": _DataTypeProperties(False, np.uint8, np.int8, None),
}


class DataType(Enum):
    """An ``Enum`` representing the datatypes produced by MANGO.

    This is used when parsing metadata to construct a :any:`Dataset`, and generally should not need
    to be constructed by a user (use the :any:`Dataset.from_path` classmethod instead).

    When needed, :any:`DataType`\\ s should be constructed via either the :any:`infer_from_path` or
    the :any:`from_basename` classmethods."""

    _PROJU16 = "proju16"
    _PROJF32 = "projf32"
    # tomo_float is above tomo, to ensure it is checked first when iterating over DataType
    _TOMO_FLOAT = "tomo_float"
    _TOMO = "tomo"
    _FLOAT16 = "float16"
    _FLOAT64 = "float64"
    _SEGMENTED = "segmented"
    _DISTANCE_MAP = "distance_map"
    _LABELS = "labels"
    _RGBA8 = "rgba8"

    def __str__(self) -> str:
        return self.value

    @property
    def is_discrete(self) -> bool:
        """Whether the :any:`DataType` is discrete."""
        return _DATATYPE_PROPERTIES[str(self)].discrete

    @property
    def dtype(self) -> type:
        """The numpy ``dtype`` appropriate for storing data of the :any:`DataType`.

        Because of a historical decision in MANGO, the datatype listed in ANU CTLab NetCDFs is not
        guaranteed to have the correct signed/unsigned type -- for some MANGO datatypes, data recorded
        in the NetCDF as an integer type is really an unsigned integer stored in an integer.
        The :any:`dtype` is the real datatype of the data, regardless of whether a loaded NetCDF
        exhibits this behaviour (trust this value, not the NetCDF header)."""
        return _DATATYPE_PROPERTIES[str(self)].dtype

    @property
    def _dtype_uncorrected(self) -> type:
        return _DATATYPE_PROPERTIES[str(self)].dtype_uncorrected

    def _mask_value(self, uncorrected: bool = False) -> StorageDType | None:
        props = _DATATYPE_PROPERTIES[str(self)]
        if props.mask_value is None:
            return None

        dtype = props.dtype_uncorrected if uncorrected else props.dtype
        val: StorageDType = dtype(props.mask_value)
        return val

    @property
    def mask_value(self) -> StorageDType | None:
        """The mask value of the :any:`DataType`.

        This value is corrected for signedness if required (see :any:`dtype`\\ )."""
        return self._mask_value()

    @property
    def _mask_value_uncorrected(self) -> StorageDType | None:
        return self._mask_value(True)

    @classmethod
    def infer_from_path(cls, path: str | Path) -> Self:
        """Create a :any:`DataType` object by inferring it from the path to the data being loaded.

        Relies on MANGO's standardised file naming.

        :rtype: :any:`DataType`"""
        basename = os.path.basename(os.path.normpath(path)).removeprefix("cntr_")
        for data_type in DataType:
            if basename.startswith(str(data_type)):
                return cls(data_type)
        raise RuntimeError("File datatype not recognised from name.")

    @classmethod
    def from_basename(cls, basename: str) -> Self:
        """Create a :any:`DataType` object from it's name as a string.

        E.g., ``DataType.from_basename("tomo")``

        :rtype: :any:`DataType`"""
        try:
            return cls(basename)
        except KeyError as e:
            raise RuntimeError(f"Basename {basename} not recognized.", e) from e
