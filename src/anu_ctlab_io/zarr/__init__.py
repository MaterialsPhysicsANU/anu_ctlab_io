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

    if ome:
        zg = zarr.open_group(path, zarr_format=3)
        multiscale = zg.metadata.attributes["ome"]["multiscales"][0]
        component_path = multiscale["datasets"][0]["path"]
        data = da.from_zarr(path, component=component_path, **kwargs)
        attrs: dict[str, Any] = dict(zg.attrs)["mango"]  # type: ignore # -- because Zarr has iffy typehinting
        dimension_names: tuple[str, ...] = tuple(
            [x["name"] for x in multiscale["axes"]]
        )  # type: ignore # -- because Zarr has iffy typehinting
        voxel_unit_list: tuple[str, ...] = tuple(
            [x["unit"] for x in multiscale["axes"]]
        )  # type: ignore # -- because Zarr has iffy typehinting
        assert (
            voxel_unit_list[0] == voxel_unit_list[1]
            and voxel_unit_list[1] == voxel_unit_list[2]
        )  # we only use one unit, so this must hold
        voxel_unit = VoxelUnit.from_str(voxel_unit_list[0])

        voxel_size = multiscale["datasets"][0]["coordinateTransformations"][0]["scale"]

    else:
        za = zarr.open_array(path, zarr_format=3)
        attrs: dict[str, Any] = dict(za.attrs)["mango"]  # type: ignore # -- because Zarr has iffy typehinting
        dimension_names: tuple[str, ...] = za.metadata.dimension_names  # type: ignore # -- because Zarr has iffy typehinting
        voxel_unit = VoxelUnit.from_str(attrs["voxel_unit"])
        voxel_size = attrs["voxel_size_xyz"]

    return Dataset(
        data=data,
        dimension_names=dimension_names,
        datatype=datatype,
        voxel_unit=voxel_unit,
        voxel_size=voxel_size,
        history=attrs["history"],
    )
