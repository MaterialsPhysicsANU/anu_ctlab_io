from enum import Enum
from typing import Any


class VoxelUnit(Enum):
    """The unit of size of a voxel."""

    M = "m"
    CM = "cm"
    MM = "mm"
    UM = "um"
    NM = "nm"
    ANGSTROM = "angstrom"
    VOXEL = "voxel"

    @classmethod
    def from_str(cls, string: str) -> "VoxelUnit":
        """Create a VoxelUnit from the string name of the unit.

        Accepts a wide range of standard representations of each unit, and is case insensitive."""
        units_lut: dict[str, VoxelUnit] = {
            # short names
            "m": cls.M,
            "cm": cls.CM,
            "mm": cls.MM,
            "um": cls.UM,
            "nm": cls.NM,
            "a": cls.ANGSTROM,
            # long names
            "meter": cls.M,
            "centimeter": cls.CM,
            "millimeter": cls.MM,
            "micrometer": cls.UM,
            "nanometer": cls.NM,
            "angstrom": cls.ANGSTROM,
            "voxel": cls.VOXEL,
            # alternative symbols
            "µm": cls.UM,
            "å": cls.ANGSTROM,
            "au": cls.ANGSTROM,
            "a.u.": cls.ANGSTROM,
        }
        try:
            return units_lut[string.lower()]
        except KeyError as e:
            raise ValueError(f"Unknown VoxelUnit {string}", e) from e

    def __eq__(self, item: Any) -> bool:
        if isinstance(item, str):
            return self.value == item
        if isinstance(item, VoxelUnit):
            return self.value == item.value
        else:
            return False

    def __str__(self) -> str:
        return self.value
