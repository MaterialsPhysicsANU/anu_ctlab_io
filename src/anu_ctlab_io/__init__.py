from anu_ctlab_io._dataset import Dataset
from anu_ctlab_io._datatype import DataType, StorageDType
from anu_ctlab_io._version import version as __version__
from anu_ctlab_io._voxel_properties import VoxelSize, VoxelUnit
from contextlib import suppress

with suppress(ImportError):
    import anu_ctlab_io.netcdf as netcdf

with suppress(ImportError):
    import anu_ctlab_io.zarr as zarr

__all__ = [
    "VoxelSize",
    "VoxelUnit",
    "DataType",
    "StorageDType",
    "Dataset",
    "__version__",
    "zarr",
    "netcdf",
]
