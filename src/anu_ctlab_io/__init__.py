import os
import numpy as np
from typing import Self
from enum import Enum, auto
from dataclasses import dataclass


__all__ = ["VoxelSize", "VoxelUnit", "DataType", "storage_dtypes"]


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
