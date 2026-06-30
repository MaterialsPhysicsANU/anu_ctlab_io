"""Create and write OME-Zarr groups."""

import logging
from dataclasses import dataclass
from enum import Enum
from math import prod
from pathlib import Path
from typing import Any

import dask.array as da
import zarr
from dask.delayed import Delayed
from ome_zarr_models.v05.axes import Axis
from ome_zarr_models.v05.coordinate_transformations import (
    VectorScale,
    VectorTranslation,
)
from ome_zarr_models.v05.multiscales import Dataset as OMEDataset
from ome_zarr_models.v05.multiscales import Multiscale

from anu_ctlab_io._dataset import Dataset
from anu_ctlab_io.zarr._layout import (
    ChunkShape,
    ChunkSpec,
    _chunk_key_encoding,
    _resolve_zarr_layout,
)
from anu_ctlab_io.zarr._multiscale import (
    DownsampleMethod,
    _array_fits_in_store_unit,
    _build_direct_ome_zarr_write,
    _can_downsample_shape_by_two,
    _chunk_grid_preserves_downsample_alignment,
    _disable_single_subchunk_sharding,
    _downsample_array_by_two,
    _downsampled_shape,
    _safe_downsample_rechunk_shape,
)

logger = logging.getLogger(__name__)


class OMEZarrVersion(Enum):
    """OME-Zarr specification version to use when writing."""

    v05 = "0.5"


@dataclass(frozen=True)
class _OMELevelSpec:
    shape: ChunkShape
    chunks: ChunkShape
    subchunks: ChunkShape | None

    @property
    def layout(self) -> tuple[ChunkShape, ChunkShape | None]:
        return self.chunks, self.subchunks


def _resolve_ome_level_layout(
    *,
    level_shape: ChunkShape,
    level_array: da.Array,
    chunks: ChunkSpec,
    subchunks: ChunkSpec | None,
    rechunk_before_store: bool,
    preferred_chunks: ChunkShape | None = None,
) -> tuple[ChunkShape, ChunkShape | None]:
    if rechunk_before_store:
        return _resolve_zarr_layout(
            shape=level_shape,
            chunks=chunks,
            subchunks=subchunks,
            aligned_chunks=None,
        )

    if chunks == "auto" and subchunks is not None and preferred_chunks is not None:
        try:
            return _resolve_zarr_layout(
                shape=level_shape,
                chunks=preferred_chunks,
                subchunks=subchunks,
                aligned_chunks=level_array.chunks,
            )
        except ValueError:
            logger.debug(
                "Ignoring preferred OME-Zarr level layout=%s for shape=%s "
                "because it is incompatible with the downsampled chunk grid.",
                preferred_chunks,
                level_shape,
            )

    try:
        return _resolve_zarr_layout(
            shape=level_shape,
            chunks=chunks,
            subchunks=subchunks,
            aligned_chunks=level_array.chunks,
        )
    except ValueError:
        logger.debug(
            "Falling back to rechunked storage for OME-Zarr level shape=%s "
            "because the layout is incompatible with the downsampled chunk grid.",
            level_shape,
        )
        return _resolve_zarr_layout(
            shape=level_shape,
            chunks=chunks,
            subchunks=subchunks,
            aligned_chunks=None,
        )


def _write_ome_zarr_group(
    data_array: da.Array,
    path: Path,
    dataset: Dataset,
    *,
    chunks: ChunkSpec,
    subchunks: ChunkSpec | None,
    create_array_kwargs: dict[str, Any],
    mango_attrs: dict[str, Any] | None,
    ome_zarr_version: OMEZarrVersion,
    rechunk_before_store: bool,
    multiscale: bool,
    downsample_method: DownsampleMethod,
    dimension_separator_threshold: int | None,
    compute: bool = True,
) -> Delayed | None:
    """Write data as an OME-Zarr group, optionally using Zarr v3 sharding.

    In Zarr v3 sharding:
    - chunks = primary write chunks
    - subchunks = subdivisions within each sharded chunk

    If ``subchunks`` is ``None``, the array is written without the sharding codec.
    """

    if not str(path).endswith(".zarr"):
        path = Path(str(path) + ".zarr")

    root = zarr.create_group(path, overwrite=True)

    ndim = data_array.ndim
    dimension_names = (
        dataset.dimension_names[:ndim]
        if len(dataset.dimension_names) >= ndim
        else dataset.dimension_names
    )

    axes = [
        Axis(name=name, type="space", unit=dataset.voxel_unit.to_full_name())
        for name in dimension_names
    ]

    voxel_size_list = [float(v) for v in dataset.voxel_size]
    scale_transform = VectorScale(type="scale", scale=voxel_size_list)
    level_chunks, level_subchunks = _resolve_zarr_layout(
        shape=data_array.shape,
        chunks=chunks,
        subchunks=subchunks,
        aligned_chunks=data_array.chunks if not rechunk_before_store else None,
    )
    logger.debug(
        "Resolved OME-Zarr level 0 layout: chunks=%s, subchunks=%s, rechunk_before_store=%s",
        level_chunks,
        level_subchunks,
        rechunk_before_store,
    )

    store_data_array = data_array
    if rechunk_before_store:
        logger.debug(
            "Rechunking input for OME-Zarr write: source_chunks=%s, write_shape=%s",
            data_array.chunks,
            level_chunks,
        )
        store_data_array = data_array.rechunk(level_chunks)  # type: ignore[no-untyped-call]
    else:
        logger.debug(
            "Skipping input rechunk for OME-Zarr write: source_chunks=%s",
            data_array.chunks,
        )

    level_specs: list[_OMELevelSpec] = [
        _OMELevelSpec(data_array.shape, level_chunks, level_subchunks)
    ]
    level_0_store_unit = level_subchunks or level_chunks
    level_0_store_unit_elements = prod(level_0_store_unit)
    if multiscale:
        current_array = data_array
        current_layout = (level_chunks, level_subchunks)
        while True:
            current_shape = current_array.shape
            if _array_fits_in_store_unit(current_shape, level_0_store_unit_elements):
                logger.debug(
                    "Stopping OME-Zarr pyramid at level %s because shape=%s fits "
                    "within one level-0 storage unit=%s.",
                    len(level_specs) - 1,
                    current_shape,
                    level_0_store_unit,
                )
                break
            if not _can_downsample_shape_by_two(current_shape):
                logger.debug(
                    "Stopping OME-Zarr pyramid at level %s because shape=%s cannot "
                    "be downsampled by two on every axis.",
                    len(level_specs) - 1,
                    current_shape,
                )
                break

            downsample_input = current_array
            if not _chunk_grid_preserves_downsample_alignment(current_array.chunks):
                rechunk_shape = _safe_downsample_rechunk_shape(
                    current_shape,
                    current_layout[0],
                )
                logger.debug(
                    "Rechunking OME-Zarr level %s before downsampling: "
                    "source_chunks=%s, rechunk_shape=%s",
                    len(level_specs) - 1,
                    current_array.chunks,
                    rechunk_shape,
                )
                if (
                    rechunk_before_store
                    and current_array is data_array
                    and rechunk_shape == level_chunks
                ):
                    downsample_input = store_data_array
                else:
                    downsample_input = current_array.rechunk(  # type: ignore[no-untyped-call]
                        rechunk_shape
                    )

            next_array = _downsample_array_by_two(downsample_input, downsample_method)
            next_shape = _downsampled_shape(current_shape)
            preferred_next_chunks = _downsampled_shape(current_layout[0])
            next_chunks, next_subchunks = _resolve_ome_level_layout(
                level_shape=next_shape,
                level_array=next_array,
                chunks=chunks,
                subchunks=subchunks,
                rechunk_before_store=rechunk_before_store,
                preferred_chunks=preferred_next_chunks,
            )

            level_specs.append(_OMELevelSpec(next_shape, next_chunks, next_subchunks))
            current_array = next_array
            current_layout = (next_chunks, next_subchunks)

    ome_datasets: list[OMEDataset] = []
    for level in range(len(level_specs)):
        scale = VectorScale(type="scale", scale=[float(2**level)] * ndim)
        transforms: tuple[VectorScale] | tuple[VectorScale, VectorTranslation]
        if level > 0:
            transforms = (
                scale,
                VectorTranslation(
                    type="translation",
                    translation=[float((2**level - 1) / 2)] * ndim,
                ),
            )
        else:
            transforms = (scale,)
        ome_datasets.append(
            OMEDataset(
                path=str(level),
                coordinateTransformations=transforms,
            )
        )

    ome_multiscale = Multiscale(
        name="",
        axes=axes,
        datasets=tuple(ome_datasets),
        coordinateTransformations=(scale_transform,),
    )

    # Set OME attributes on root group
    root.attrs["ome"] = {
        "version": ome_zarr_version.value,
        "multiscales": [ome_multiscale.model_dump(mode="json")],
    }

    if mango_attrs:
        root.attrs["mango"] = mango_attrs

    zarr_arrays: list[zarr.Array[Any]] = []
    for level, (level_shape, (level_chunks, level_subchunks)) in enumerate(
        (spec.shape, spec.layout) for spec in level_specs
    ):
        level_chunks, level_subchunks = _disable_single_subchunk_sharding(
            level_chunks, level_subchunks
        )
        level_create_array_kwargs = dict(create_array_kwargs)
        if dimension_separator_threshold is not None:
            level_create_array_kwargs["chunk_key_encoding"] = _chunk_key_encoding(
                level_shape, level_chunks, dimension_separator_threshold
            )

        array = root.create_array(
            str(level),
            shape=level_shape,
            chunks=level_subchunks or level_chunks,
            shards=level_chunks if level_subchunks is not None else None,
            dtype=data_array.dtype,
            dimension_names=list(dimension_names),
            overwrite=True,
            **level_create_array_kwargs,
        )
        zarr_arrays.append(array)
        logger.debug(
            "Created OME-Zarr array: path=%s/%s, shape=%s, dtype=%s, chunks=%s, "
            "shards=%s, dimension_names=%s",
            path,
            level,
            array.shape,
            array.dtype,
            array.chunks,
            array.shards,
            dimension_names,
        )

    result = _build_direct_ome_zarr_write(
        store_data_array,
        path,
        zarr_arrays,
        [spec.shape for spec in level_specs],
        [spec.layout for spec in level_specs],
        downsample_method,
    )
    if compute:
        result.compute()  # type: ignore[no-untyped-call]
        return None
    return result
