from enum import Enum
from typing import Any, Self


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

    def __hash__(self) -> int:
        return hash(self.value)

    def __str__(self) -> str:
        return self.value

    def to_full_name(self) -> str:
        """Return the full unit name for OME-Zarr metadata.

        OME-Zarr specification requires full unit names (e.g., "millimeter")
        rather than abbreviated forms (e.g., "mm").

        :return: Full unit name as string.
        """
        full_names = {
            VoxelUnit.M: "meter",
            VoxelUnit.CM: "centimeter",
            VoxelUnit.MM: "millimeter",
            VoxelUnit.UM: "micrometer",
            VoxelUnit.NM: "nanometer",
            VoxelUnit.ANGSTROM: "angstrom",
            VoxelUnit.VOXEL: "voxel",
        }
        return full_names[self]

    def _conversion_factor(self, target_unit: Self) -> float:
        """Get the conversion factor from this unit to the target unit.

        :param target_unit: The unit to convert to.
        :return: The multiplication factor to convert from this unit to the target unit.
        :raises ValueError: If either unit is VOXEL or if conversion is not supported.
        """
        if self == VoxelUnit.VOXEL or target_unit == VoxelUnit.VOXEL:
            raise ValueError("Cannot convert to/from voxel unit")

        # Conversion factors to meters
        to_meters: dict[VoxelUnit, float] = {
            VoxelUnit.M: 1.0,
            VoxelUnit.CM: 1e-2,
            VoxelUnit.MM: 1e-3,
            VoxelUnit.UM: 1e-6,
            VoxelUnit.NM: 1e-9,
            VoxelUnit.ANGSTROM: 1e-10,
        }

        return to_meters[self] / to_meters[target_unit]
