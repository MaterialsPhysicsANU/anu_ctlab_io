"""
python I/O for the ANU CTLab array storage format(s).
"""
from contextlib import suppress

from anu_ctlab_io._dataset import Dataset
from anu_ctlab_io._datatype import DataType, StorageDType
from anu_ctlab_io._version import version as __version__
from anu_ctlab_io._voxel_properties import VoxelUnit

with suppress(ImportError):
    import anu_ctlab_io.netcdf as netcdf

with suppress(ImportError):
    import anu_ctlab_io.zarr as zarr

__all__ = [
    "Dataset",  # out of sorted order so it comes first in the docs
    "DataType",
    "StorageDType",
    "VoxelUnit",
    "__version__",
    "netcdf",
    "zarr",
]
