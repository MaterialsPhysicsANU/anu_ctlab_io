import os
from typing import Self, TypeAlias


from enum import Enum
from dataclasses import dataclass
import numpy as np


__all__ = ["StorageDType", "DataType"]


StorageDType: TypeAlias = (
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
    PROJU16 = "proju16"
    PROJF32 = "projf32"
    # tomo_float is above tomo, to ensure it is checked first when iterating over DataType
    TOMO_FLOAT = "tomo_float"
    TOMO = "tomo"
    FLOAT16 = "float16"
    FLOAT64 = "float64"
    SEGMENTED = "segmented"
    DISTANCE_MAP = "distance_map"
    LABELS = "labels"
    RGBA8 = "rgba8"

    def __str__(self) -> str:
        return self.value

    @property
    def is_discrete(self) -> bool:
        return _DATATYPE_PROPERTIES[str(self)].discrete

    @property
    def dtype(self) -> type:
        return _DATATYPE_PROPERTIES[str(self)].dtype

    @property
    def _dtype_uncorrected(self) -> type:
        return _DATATYPE_PROPERTIES[str(self)].dtype_uncorrected

    def _mask_value(self, uncorrected: bool = False) -> StorageDType | None:
        props = _DATATYPE_PROPERTIES[str(self)]
        if props.mask_value is None:
            return None

        dtype = props.dtype_uncorrected if uncorrected else props.dtype
        return (dtype)(props.mask_value)

    @property
    def mask_value(self) -> StorageDType | None:
        return self._mask_value()

    @property
    def _mask_value_uncorrected(self) -> StorageDType | None:
        return self._mask_value(True)

    @classmethod
    def infer_from_path(cls, path: os.PathLike) -> Self:
        basename = os.path.basename(os.path.normpath(path)).removeprefix("cntr_")
        for data_type in DataType:
            if basename.startswith(str(data_type)):
                return cls(data_type)
        raise RuntimeError("File datatype not recognised from name.")

    @classmethod
    def from_basename(cls, basename: str) -> Self:
        try:
            return cls[basename]
        except KeyError as e:
            raise RuntimeError(f"Basename {basename} not recognized.", e)
