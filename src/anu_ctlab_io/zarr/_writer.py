"""Write data to the ANU CTLab zarr data format."""

import warnings
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import dask.array as da
import numpy as np
import zarr
from ome_zarr_models.v05.axes import Axis
from ome_zarr_models.v05.coordinate_transformations import VectorScale
from ome_zarr_models.v05.multiscales import Dataset as OMEDataset
from ome_zarr_models.v05.multiscales import Multiscale

from anu_ctlab_io._dataset import Dataset
from anu_ctlab_io._datatype import DataType
from anu_ctlab_io._parse_history import History

__all__ = ["OMEZarrVersion", "dataset_to_zarr"]

_DEFAULT_3D_ZARR_CHUNKS = (32, 32, 32)
_DEFAULT_2D_ZARR_CHUNKS = (1, 256, 256)
type ChunkSpec = tuple[int, ...] | Literal["auto"]
type ShardSpec = tuple[int, ...] | Literal["auto"] | None
type ChunkLayoutSpec = ChunkSpec | Literal["anu"]
type ShardLayoutSpec = ShardSpec | Literal["anu"]


class OMEZarrVersion(Enum):
    """OME-Zarr specification version to use when writing."""

    v05 = "0.5"


def dataset_to_zarr(
    dataset: Dataset,
    path: Path | str,
    datatype: DataType | str | None = None,
    dataset_id: str | None = None,
    ome_zarr_version: OMEZarrVersion | None = OMEZarrVersion.v05,
    max_shard_size_mb: float | None = None,
    history: History | None = None,
    chunk_size_mb: float | None = None,
    chunks: ChunkLayoutSpec = "anu",
    shards: ShardLayoutSpec = "anu",
    create_array_kwargs: dict[str, Any] | None = None,
    **extra_attrs: Any,
) -> None:
    """Write a :any:`Dataset` to Zarr format.

    :param dataset: The :any:`Dataset` to write.
    :param path: Path to write the Zarr store.
    :param datatype: The data type identifier. If None, attempts to infer from dataset.
    :param dataset_id: Unique identifier for the dataset. Auto-generated if not provided.
    :param ome_zarr_version: OME-Zarr specification version to use.
        Set to :any:`OMEZarrVersion.v05` (default) to write OME-Zarr V0.5 group format.
        Set to ``None`` to write a simple Zarr V3 array with mango metadata.
    :param max_shard_size_mb: Maximum shard size in MB for optional size-based Zarr v3 sharding.
        Deprecated and ignored. Passing this emits a warning and leaves layout selection
        to ``chunks``/``shards``.
    :param history: Dictionary of history entries to add. Keys should be identifiers,
        values are history strings.
    :param chunk_size_mb: Target chunk size in MB for optional size-based chunk calculation.
        Deprecated and ignored. Passing this emits a warning and leaves layout selection
        to ``chunks``/``shards``.
    :param chunks: Explicit chunk shape as a tuple (e.g., ``(10, 512, 512)``).
        Can be provided on its own to write a non-sharded Zarr array, or together with
        ``shards`` to use the sharding codec. ``'auto'`` delegates chunk selection to
        zarr-python. ``'anu'`` uses the library's default chunk heuristic. To write
        without sharding, pass ``shards=None`` explicitly.
        A value of ``0`` means "span this axis" - the full array axis for unsharded writes,
        or the full shard axis when ``shards`` is also provided.
    :param shards: Explicit shard shape as a tuple (e.g., ``(100, 512, 512)``).
        May be provided together with explicit ``chunks``, with ``chunks='anu'`` to use
        the library's default chunk heuristic, with ``'auto'`` to delegate shard selection
        to zarr-python, ``None`` to disable sharding, or ``'anu'`` to use the library's
        default shard heuristic. Cannot be used with ``0`` sentinels when
        ``chunks='auto'`` because zarr resolves the sizes itself.
        A value of ``0`` means "span the full array axis".
        When provided, the user is responsible for ensuring shard shapes are evenly divisible by chunk shapes.
    :param create_array_kwargs: Additional keyword arguments to pass to zarr.create_array().
        For example, to set compression: ``create_array_kwargs={'compressors': [ZstdCodec(level=5)]}``.
    :param extra_attrs: Additional attributes to include in mango metadata.
    """
    if isinstance(path, str):
        path = Path(path)

    if datatype is None:
        if dataset._datatype is not None:
            datatype = dataset._datatype
        else:
            datatype = None
    elif isinstance(datatype, str):
        datatype = DataType.from_basename(datatype)

    # Generate dataset_id if not provided
    if dataset_id is None and datatype is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_id = f"{timestamp}_{datatype}"

    data_array = dataset.data

    ignored_size_args = ", ".join(
        name
        for name, value in (
            ("chunk_size_mb", chunk_size_mb),
            ("max_shard_size_mb", max_shard_size_mb),
        )
        if value is not None
    )
    if ignored_size_args:
        warnings.warn(
            f"{ignored_size_args} is ignored when writing Zarr. Use chunks/shards or 'auto' instead.",
            UserWarning,
            stacklevel=2,
        )

    inner_chunks, outer_shards = _resolve_zarr_layout(
        shape=data_array.shape,
        chunks=chunks,
        shards=shards,
    )

    mango_attrs: dict[str, Any] | None = None
    if datatype is not None:
        mango_attrs = _build_mango_attrs(
            dataset, datatype, dataset_id, history, extra_attrs
        )

    if create_array_kwargs is None:
        create_array_kwargs = {}

    if ome_zarr_version is not None:
        _write_ome_zarr_group(
            data_array,
            path,
            dataset,
            inner_chunks,
            outer_shards,
            create_array_kwargs,
            mango_attrs,
            ome_zarr_version,
        )
    else:
        _write_zarr_array(
            data_array,
            path,
            dataset,
            inner_chunks,
            outer_shards,
            create_array_kwargs,
            mango_attrs,
        )


def _round_up_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _resolve_zarr_layout(
    *,
    shape: tuple[int, ...],
    chunks: ChunkLayoutSpec,
    shards: ShardLayoutSpec,
) -> tuple[ChunkSpec, ShardSpec]:
    """Resolve the final Zarr chunk/shard layout from the supported writer inputs.

    The resolution order is:
    - explicit ``chunks``/``shards`` shapes when provided
    - ``'anu'`` sentinels, which use the library default layout of ``(32, 32, 32)``
      subchunks with X/Y-spanning shards
      or, for ``z == 1`` data, ``(1, 256, 256)``-style in-plane chunks constrained by the array shape

    Explicit shapes also support ``0`` as a sentinel. In ``chunks``, ``0`` means span
    the full array axis for unsharded writes, or the full shard axis when ``shards`` is
    provided. In ``shards``, ``0`` means span the array axis, rounded up to a multiple
    of the resolved chunk size so the sharding codec remains valid.
    """
    anu_chunks, anu_shards = _default_chunks_and_shards(shape)

    if chunks is None:
        chunks = "anu"
    if shards == "anu":
        shards = anu_shards
    if chunks == "anu":
        chunks = anu_chunks

    if shards is not None and chunks is None and shards != "auto":
        raise ValueError(
            "shards requires chunks to also be provided. "
            f"Got chunks={chunks}, shards={shards}."
        )

    if chunks == "auto":
        return "auto", shards

    if shards == "auto":
        return chunks, "auto"

    return _normalize_explicit_shapes(shape, chunks, shards)


def _normalize_explicit_shapes(
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    shards: tuple[int, ...] | None,
) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
    if len(chunks) != len(shape):
        raise ValueError(
            f"chunks must have the same number of dimensions as the array shape. Got chunks={chunks}, shape={shape}."
        )
    if shards is not None and len(shards) != len(shape):
        raise ValueError(
            f"shards must have the same number of dimensions as the array shape. Got shards={shards}, shape={shape}."
        )

    chunk_span = (
        tuple(
            axis_size if shard_size == 0 else shard_size
            for shard_size, axis_size in zip(shards, shape, strict=True)
        )
        if shards is not None
        else shape
    )
    resolved_chunks = tuple(
        axis_size if chunk_size == 0 else chunk_size
        for chunk_size, axis_size in zip(chunks, chunk_span, strict=True)
    )

    resolved_shards: tuple[int, ...] | None = None
    if shards is not None:
        resolved_shards = tuple(
            _round_up_to_multiple(axis_size, chunk_size)
            if shard_size == 0
            else shard_size
            for shard_size, axis_size, chunk_size in zip(
                shards, shape, resolved_chunks, strict=True
            )
        )

    return resolved_chunks, resolved_shards


def _default_chunks_and_shards(
    shape: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    z_chunk, y_chunk, x_chunk = (
        _DEFAULT_2D_ZARR_CHUNKS if shape[0] == 1 else _DEFAULT_3D_ZARR_CHUNKS
    )
    zdim, ydim, xdim = shape
    inner_chunks = (min(z_chunk, zdim), min(y_chunk, ydim), min(x_chunk, xdim))
    outer_shards = (
        inner_chunks[0],
        inner_chunks[1] if ydim < y_chunk else _round_up_to_multiple(ydim, y_chunk),
        inner_chunks[2] if xdim < x_chunk else _round_up_to_multiple(xdim, x_chunk),
    )
    return inner_chunks, outer_shards


def _calculate_chunks_and_shards_by_size(
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    chunk_size_mb: float,
    max_shard_size_mb: float,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Calculate inner chunks and outer shards for Zarr v3 sharding.

    In Zarr v3:
    - inner_chunks (chunks param) = subdivisions within each shard file
    - outer_shards (shards param) = how data is split into shard files

    This algorithm:
    - Keeps xy planes intact (critical for CT data access patterns)
    - Makes outer shards multiples of inner chunks for alignment

    :param shape: Array shape (z, y, x).
    :param dtype: Array data type.
    :param chunk_size_mb: Target size for inner chunks in MB.
    :param max_shard_size_mb: Maximum shard size in MB.
    :return: Tuple of (inner_chunks, outer_shards).
    """
    zdim, ydim, xdim = shape
    bytes_per_element = np.dtype(dtype).itemsize
    bytes_per_slice = ydim * xdim * bytes_per_element

    # Calculate inner chunk size (subdivisions within shards)
    inner_target_bytes = chunk_size_mb * 1024 * 1024
    z_inner = max(1, int(inner_target_bytes / bytes_per_slice))
    z_inner = min(z_inner, zdim)

    inner_chunks = (z_inner, ydim, xdim)

    # Calculate outer shard size (how data is split into files)
    shard_target_bytes = max_shard_size_mb * 1024 * 1024
    z_outer = max(1, int(shard_target_bytes / bytes_per_slice))
    z_outer = min(z_outer, zdim)

    # Ensure outer shards are at least as large as inner chunks
    # and are a multiple of inner chunks for proper alignment
    if z_outer < z_inner:
        z_outer = z_inner
    else:
        # Make outer a multiple of inner
        multiple = max(1, z_outer // z_inner)
        z_outer = z_inner * multiple
        z_outer = min(z_outer, zdim)

    outer_shards = (z_outer, ydim, xdim)

    return inner_chunks, outer_shards


def _build_mango_attrs(
    dataset: Dataset,
    datatype: DataType,
    dataset_id: str | None,
    history: History | None,
    extra_attrs: dict[str, Any],
) -> dict[str, Any]:
    """Build mango metadata attributes."""
    mango_attrs: dict[str, Any] = {
        "metadata_version_major": 1,
        "metadata_version_minor": 0,
        "basename": str(datatype),
        "voxel_size_xyz": [float(v) for v in dataset.voxel_size],
        "voxel_unit": str(dataset.voxel_unit),
        "coord_transform": "",
        "intensity_f2i_offset_scale": [0.0, 1.0],
        "offset_to_coordinate_origin_xyz": [0.0, 0.0, 0.0],
        "coordinate_origin_xyz": [0, 0, 0],
    }

    if dataset_id is not None:
        mango_attrs["dataset_id"] = dataset_id

    if history is not None:
        mango_attrs["history"] = history
    else:
        mango_attrs["history"] = dataset.history

    mango_attrs.update(extra_attrs)

    return mango_attrs


def _write_ome_zarr_group(
    data_array: da.Array,
    path: Path,
    dataset: Dataset,
    inner_chunks: tuple[int, ...],
    outer_shards: tuple[int, ...] | None,
    create_array_kwargs: dict[str, Any],
    mango_attrs: dict[str, Any] | None,
    ome_zarr_version: OMEZarrVersion,
) -> None:
    """Write data as an OME-Zarr group with Zarr v3 sharding.

    In Zarr v3 sharding:
    - inner_chunks (chunks param) = subdivisions within each shard file
    - outer_shards (shards param) = how data is split into shard files

    If ``outer_shards`` is ``None``, the array is written without the sharding codec.
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
    identity_transform = VectorScale(type="scale", scale=[1.0] * ndim)

    multiscale = Multiscale(
        name="",
        axes=axes,
        datasets=(
            OMEDataset(
                path="0",
                coordinateTransformations=(identity_transform,),
            ),
        ),
        coordinateTransformations=(scale_transform,),
    )

    # Set OME attributes on root group
    root.attrs["ome"] = {
        "version": ome_zarr_version.value,
        "multiscales": [multiscale.model_dump(mode="json")],
    }

    if mango_attrs:
        root.attrs["mango"] = mango_attrs

    array = root.create_array(
        "0",
        shape=data_array.shape,
        chunks=inner_chunks,
        shards=outer_shards,
        dtype=data_array.dtype,
        dimension_names=list(dimension_names),
        overwrite=True,
        **create_array_kwargs,
    )

    # Always rechunk to the write granularity of the Zarr array before writing.
    # If using sharding, the write granularity is the shard shape (`.shards`).
    # If not using sharding, `.shards` is None and `chunks` is used as the write granularity.
    # Note that the rechunk is lazy if the array is already in the desired chunk shape.
    # The straddling chunks/shards w.r.t the array shape need no special handling.
    write_shape = array.shards or array.chunks
    data_array = data_array.rechunk(write_shape)  # type: ignore[no-untyped-call]

    # dask's to_zarr internally calls normalize_chunks("auto", ...) which can produce
    # chunk sizes that are not multiples of the shard shape, causing misaligned writes
    # that manifest as large regions of zeros in the output. Using da.store directly
    # bypasses that internal rechunk entirely, writing each dask chunk straight into
    # its corresponding region in the zarr array.
    da.store(data_array, array, lock=False, compute=True)  # type: ignore[arg-type]


def _write_zarr_array(
    data_array: da.Array,
    path: Path,
    dataset: Dataset,
    inner_chunks: tuple[int, ...],
    outer_shards: tuple[int, ...] | None,
    create_array_kwargs: dict[str, Any],
    mango_attrs: dict[str, Any] | None,
) -> None:
    """Write data as a simple Zarr V3 array with mango metadata.

    When ``outer_shards`` is provided, Zarr v3 sharding is used:
    - inner_chunks (chunks param) = subdivisions within each shard file
    - outer_shards (shards param) = how data is split into shard files
    """
    if not str(path).endswith(".zarr"):
        path = Path(str(path) + ".zarr")

    # Only use dimension names that match the actual data dimensions
    ndim = data_array.ndim
    dimension_names = (
        dataset.dimension_names[:ndim]
        if len(dataset.dimension_names) >= ndim
        else dataset.dimension_names
    )

    array = zarr.create_array(
        path,
        shape=data_array.shape,
        chunks=inner_chunks,
        shards=outer_shards,
        dtype=data_array.dtype,
        dimension_names=list(dimension_names),
        overwrite=True,
        **create_array_kwargs,
    )

    if mango_attrs:
        array.attrs["mango"] = mango_attrs

    # Always rechunk to the write granularity of the Zarr array before writing.
    # See comment in _write_ome_zarr_group for explanation.
    write_shape = array.shards or array.chunks
    data_array = data_array.rechunk(write_shape)  # type: ignore[no-untyped-call]

    da.store(data_array, array, lock=False, compute=True)  # type: ignore[arg-type]
