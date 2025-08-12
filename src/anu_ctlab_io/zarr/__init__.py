from pathlib import Path
from typing import Any

import dask.array as da
import zarr

from anu_ctlab_io.dataset import Dataset
from anu_ctlab_io.datatype import DataType
from anu_ctlab_io.voxel_properties import VoxelUnit


def dataset_from_zarr(path: Path, **kwargs) -> Dataset:
    datatype = DataType.infer_from_path(path)
    try:
        ome = False
        data = da.from_zarr(path, **kwargs)
    except zarr.errors.NodeTypeValidationError:  # happens if this is an ome
        ome = True
        data = da.from_zarr(path, component="0", **kwargs)

    if ome:
        zg = zarr.open_group(path, zarr_format=3)
        attrs: dict[str, Any] = dict(zg.attrs)["mango"]  # type: ignore # -- because Zarr has iffy typehinting
        dimension_names: tuple[str, ...] = tuple(
            map(
                lambda x: x["name"],
                zg.metadata.attributes["ome"]["multiscales"][0]["axes"],
            )
        )  # type: ignore # -- because Zarr has iffy typehinting

    else:
        za = zarr.open_array(path, zarr_format=3)
        attrs: dict[str, Any] = dict(za.attrs)["mango"]  # type: ignore # -- because Zarr has iffy typehinting
        dimension_names: tuple[str, ...] = za.metadata.dimension_names  # type: ignore # -- because Zarr has iffy typehinting

    return Dataset(
        data=data,
        dimension_names=dimension_names,
        datatype=datatype,
        voxel_unit=VoxelUnit.from_str(attrs["voxel_unit"]),
        voxel_size=attrs["voxel_size_xyz"],
        history=attrs["history"],
    )
