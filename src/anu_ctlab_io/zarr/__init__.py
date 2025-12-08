"""Read data from the ANU CTLab zarr data format.

This is an optional extra module, and must be explicitly installed to be used (e.g., ``pip install anu_ctlab_io[zarr]``)."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import zarr
from ome_zarr_models.common.coordinate_transformations import VectorScale
from ome_zarr_models.v05.image import Image
from ome_zarr_models.v05.multiscales import ValidTransform

from anu_ctlab_io._dataset import Dataset
from anu_ctlab_io._datatype import DataType
from anu_ctlab_io._voxel_properties import VoxelUnit

__all__ = ["dataset_from_zarr"]


def dataset_from_zarr(path: Path, **kwargs: Any) -> Dataset:
    """Loads a :any:`Dataset` from the path to a zarr.

    This method is used by :any:`Dataset.from_path`, by preference call that constructor directly.

    :param Path: The path to the zarr to be loaded
    :param kwargs: Currently this method consumes no kwargs, but will pass provided kwargs to ``dask.Array.from_path``."""
    try:
        data: da.Array = da.from_zarr(path, **kwargs)  # type: ignore[no-untyped-call]
        za = zarr.open_array(
            path, zarr_format=3
        )  # can't get a dask array directly from this
        attrs: dict[str, Any] = dict(za.attrs)["mango"]  # type: ignore[assignment]
        dimension_names: tuple[str, ...] = za.metadata.dimension_names  # type: ignore[assignment, union-attr]
        voxel_unit = VoxelUnit.from_str(attrs["voxel_unit"])
        voxel_size = attrs["voxel_size_xyz"]
        datatype = DataType.from_basename(attrs["basename"])
        history = attrs["history"]  # type: ignore[assignment]

    except zarr.errors.NodeTypeValidationError:  # happens if this is an ome
        zg = zarr.open_group(path, zarr_format=3)
        ome = Image.from_zarr(zg)
        multiscales = ome.ome_attributes.multiscales
        if len(multiscales) > 1:
            raise NotImplementedError(
                "Only single multiscale images are currently supported."
            ) from None

        # Extract the first multiscale and first dataset
        # These are guaranteed to exist by ome_zarr_metadata validation
        # Assumes the first dataset is the full resolution one
        multiscale = multiscales[0]
        dataset = multiscale.datasets[0]
        data = da.from_zarr(path, component=dataset.path, **kwargs)  # type: ignore[no-untyped-call]

        if len(multiscale.axes) != 3:
            raise ValueError(
                f"Provided zarr has {len(multiscale.axes)} axes, should have 3."
            ) from None

        dimension_names = tuple(
            [str(axis.name) for axis in multiscale.axes]
        )  # NOTE: str cast needed due to a mismatch in ome_zarr_models with the ome_zarr spec

        def _cast_unit(unit: str | Any | None) -> str | None:
            if unit is None:
                return None
            if isinstance(unit, str):
                return unit
            raise ValueError(f"Unsupported unit type: {unit}")

        voxel_unit_list: tuple[str | None, ...] = tuple(
            [_cast_unit(axis.unit) for axis in multiscale.axes]
        )
        if not (
            voxel_unit_list[0] == voxel_unit_list[1]
            and voxel_unit_list[1] == voxel_unit_list[2]
        ):
            raise ValueError(
                f"Provided zarr has differing units {voxel_unit_list}, these should all be equal."
            ) from None
        if voxel_unit_list[0] is None:
            voxel_unit = VoxelUnit.VOXEL
        else:
            voxel_unit = VoxelUnit.from_str(voxel_unit_list[0])

        # Calculate the voxel size from the transformations
        def extract_vector_scale(
            transformations: ValidTransform | None,
        ) -> tuple[float, float, float]:
            """Extracts the scale from a list of coordinate transformations, expects VectorScale."""
            if transformations and len(transformations) > 0:
                scale_transform = transformations[0]
                if isinstance(scale_transform, VectorScale):
                    scale = scale_transform.scale
                    if len(scale) != 3:
                        raise ValueError(
                            f"Provided zarr has {len(scale)} scale factors provided, should have 3."
                        ) from None
                    return (scale[0], scale[1], scale[2])
                else:
                    raise NotImplementedError(
                        "Only vector scales are currently supported for coordinate transformations."
                    ) from None
            return (1.0, 1.0, 1.0)

        voxel_size_dataset = extract_vector_scale(dataset.coordinateTransformations)
        voxel_size_root = extract_vector_scale(multiscale.coordinateTransformations)
        voxel_size = tuple(np.array(voxel_size_dataset) * np.array(voxel_size_root))

        if "mango" not in zg.attrs:
            # Handle a plain OME-Zarr dataset that has no mango attributes
            datatype = None
            history: dict[str, Any] = {}
        else:
            mango_attrs = zg.attrs["mango"]
            assert isinstance(mango_attrs, Mapping)
            basename = mango_attrs["basename"]
            assert isinstance(basename, str)
            datatype = DataType.from_basename(basename)
            # NOTE: Should refine history from Any to JSON
            history = mango_attrs["history"]  # type: ignore[assignment]

    return Dataset(
        data=data,
        dimension_names=dimension_names,
        datatype=datatype,
        voxel_unit=voxel_unit,
        voxel_size=voxel_size,
        history=history,
    )
