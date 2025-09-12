from enum import Enum, auto
from typing import Any


class UnknownVoxelUnitException(Exception):
    pass


class VoxelUnit(Enum):
    """The unit of size of a voxel."""

    m = auto()
    cm = auto()
    mm = auto()
    um = auto()
    nm = auto()
    angstrom = auto()
    voxel = auto()

    @classmethod
    def from_str(cls, string: str) -> "VoxelUnit":
        """Create a VoxelUnit from the string name of the unit.

        Accepts a wide range of standard representations of each unit, and is case insensitive."""
        units_lut: dict[str, VoxelUnit] = {
            # short names
            "m": cls.m,
            "cm": cls.cm,
            "mm": cls.mm,
            "um": cls.um,
            "nm": cls.nm,
            "a": cls.angstrom,
            # long names
            "meter": cls.m,
            "centimeter": cls.cm,
            "millimeter": cls.mm,
            "micrometer": cls.um,
            "nanometer": cls.nm,
            "angstrom": cls.angstrom,
            "voxel": cls.voxel,
            # alternative symbols
            "µm": cls.um,
            "å": cls.angstrom,
            "au": cls.angstrom,
            "a.u.": cls.angstrom,
        }
        try:
            return units_lut[string.lower()]
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
        return self._name_
