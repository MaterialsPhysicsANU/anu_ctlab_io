from enum import Enum, auto
from typing import Any


class UnknownVoxelUnitException(Exception):
    pass


class VoxelUnit(Enum):
    """
    The unit of size of a voxel.
    TODO: double check mango supported units
    """

    _m = auto()
    _mm = auto()
    _um = auto()
    _nm = auto()

    @classmethod
    def from_str(cls, string: str) -> "VoxelUnit":
        units_lut: dict[str, VoxelUnit] = {
            # short names
            "m": cls._m,
            "mm": cls._mm,
            "um": cls._um,
            "nm": cls._nm,
            # long names
            "meter": cls._m,
            "millimeter": cls._mm,
            "micrometer": cls._um,
            "nanometer": cls._nm,
            # alternative symbols
            "µm": cls._um,
        }
        try:
            return units_lut[string]
        except KeyError as e:
            raise UnknownVoxelUnitException(f"Unknown VoxelUnit {string}", e) from e

    def __eq__(self, item: Any) -> bool:
        if isinstance(item, str):
            try:
                return self == VoxelUnit.from_str(item)
            except UnknownVoxelUnitException:
                return False
        if isinstance(item, VoxelUnit):
            return item._value_ == self._value_
        else:
            return False

    def __str__(self) -> str:
        return self._name_[1:]
