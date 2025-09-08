from enum import Enum, auto
from typing import Any, TypeAlias

import numpy as np


VoxelSize: TypeAlias = tuple[np.float32, np.float32, np.float32]


class VoxelUnit(Enum):
    """
    The unit of size of a voxel. NOTE: Consider parameterizing this?
    TODO: double check mango supported units
    """

    m = auto()
    mm = auto()
    um = auto()
    nm = auto()
    voxels = auto()

    @classmethod
    def from_str(cls, string: str):
        return {
            # short names
            "m": cls.m,
            "mm": cls.mm,
            "um": cls.um,
            "nm": cls.nm,
            # long names
            "meter": cls.m,
            "millimeter": cls.mm,
            "micrometer": cls.um,
            "nanometer": cls.nm,
            # alternative symbols
            "µm": cls.um,
        }.get(string, cls.voxels)

    def __eq__(self, item: Any) -> bool:
        if isinstance(item, str):
            return self == VoxelUnit.from_str(item)
        if isinstance(item, VoxelUnit):
            return item._value_ == self._value_
        else:
            return False

    def __str__(self):
        return self._name_
