"""Write data to the ANU CTLab zarr data format."""

import warnings
from datetime import datetime
from enum import Enum
from math import prod
from pathlib import Path
from typing import Any, Literal

import dask.array as da
import zarr
from ome_zarr_models.v05.axes import Axis
from ome_zarr_models.v05.coordinate_transformations import VectorScale
from ome_zarr_models.v05.multiscales import Dataset as OMEDataset
from ome_zarr_models.v05.multiscales import Multiscale

from anu_ctlab_io._dataset import Dataset
from anu_ctlab_io._datatype import DataType
from anu_ctlab_io._parse_history import History

__all__ = ["OMEZarrVersion", "dataset_to_zarr"]

_DEFAULT_CHUNK_ELEMENTS = max(256**2, 32**3)
_DEFAULT_SHARD_ELEMENTS = max(8192**2, 512**3)

type ChunkShape = tuple[int, ...]
type ChunkSpec = ChunkShape | int | Literal["auto"]
type ShardSpec = ChunkShape | int | Literal["auto"] | None


class OMEZarrVersion(Enum):
    """OME-Zarr specification version to use when writing."""

    v05 = "0.5"


# TODO: 2.0.0 Force kwargs-only for optional parameters
def dataset_to_zarr(
    dataset: Dataset,
    path: Path | str,
    datatype: DataType | str | None = None,
    dataset_id: str | None = None,
    ome_zarr_version: OMEZarrVersion | None = OMEZarrVersion.v05,
    max_shard_size_mb: float
    | None = None,  # TODO: 2.0.0 Remove this deprecated parameter
    history: History | None = None,
    chunk_size_mb: float | None = None,  # TODO: 2.0.0 Remove this deprecated parameter
    chunks: ChunkSpec = "auto",
    shards: ShardSpec = "auto",
    create_array_kwargs: dict[str, Any] | None = None,
    dimension_separator_threshold: int | None = 64,
    **extra_attrs: Any,
) -> None:
    """Write a :any:`Dataset` to Zarr format.

    By default, chunks and shards use power-of-two square or cubic shapes targeting
    ``32**3`` and ``512**3`` elements respectively. Shapes are trimmed to the array
    dimensions, and shards are rounded up to chunk multiples.

    :param dataset: The :any:`Dataset` to write.
    :param path: Path to write the Zarr store.
    :param datatype: The data type identifier. If None, attempts to infer from dataset.
    :param dataset_id: Unique identifier for the dataset. Auto-generated if not provided.
    :param ome_zarr_version: OME-Zarr specification version to use.
        Set to :any:`OMEZarrVersion.v05` (default) to write OME-Zarr V0.5 group format.
        Set to ``None`` to write a simple Zarr V3 array with mango metadata.
    :param max_shard_size_mb: Maximum shard size in MB for Zarr v3 sharding.
        Deprecated and ignored.
        Passing this emits a warning and leaves layout selection to ``chunks``/``shards``.
    :param history: Dictionary of history entries to add.
        Keys should be identifiers, values are history strings.
    :param chunk_size_mb: Target chunk size in MB for automatic chunking.
        Deprecated and ignored.
        Passing this emits a warning and leaves layout selection to ``chunks``/``shards``.
    :param chunks: Explicit chunk shape as a tuple shape (e.g., ``(10, 512, 512)``), int (target # of elements), or ``'auto'``.
        Can be provided on its own to write a non-sharded Zarr array, or together with ``shards`` to use the sharding codec.
        An integer specifies the target number of elements for an automatically derived layout.
        ``'auto'`` uses a default target corresponding to ``32**3`` or ``256**2`` elements.
        To write without sharding, pass ``shards=None`` explicitly.
        A value of ``0`` in a shape tuple means "span this axis" — the full array axis for unsharded writes, or the full shard axis when ``shards`` is also provided.
    :param shards: Explicit shard shape as a tuple shape (e.g., ``(100, 512, 512)``), int (target # of elements), or ``'auto'``.
        May be provided together with an explicit ``chunks`` tuple.
        Providing an explicit shard shape with ``chunks='auto'`` is an error.
        An integer specifies the target number of elements for an automatically derived layout.
        Use ``None`` to disable sharding, or ``'auto'`` to use the default target of ``512**3`` or ``8192**2`` elements.
        A value of ``0`` in a shape tuple means "span the full array axis".
        When provided, the user is responsible for ensuring shard shapes are evenly divisible by chunk shapes.
    :param create_array_kwargs: Additional keyword arguments to pass to zarr.create_array().
        For example, to set compression: ``create_array_kwargs={'compressors': [ZstdCodec(level=5)]}``.
    :param dimension_separator_threshold: Use ``'/'`` as the chunk key dimension
        separator when the number of chunks exceeds this threshold; otherwise use
        ``'.'``. ``None`` uses the Zarr default of ``'/'``.
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
        verb = "are" if "," in ignored_size_args else "is"
        warnings.warn(
            f"{ignored_size_args} {verb} ignored when writing Zarr. Use chunks/shards or 'auto' instead.",
            UserWarning,
            stacklevel=2,
        )

    if isinstance(shards, tuple) and chunks == "auto":
        raise ValueError("shards cannot be provided without explicit chunks")

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

    create_array_kwargs = dict(create_array_kwargs or {})
    if dimension_separator_threshold is not None:
        # Use outer_shards for chunk key encoding calculation when sharding is
        # enabled, since Zarr's chunk_key_encoding addresses shards, not inner chunks.
        _key_chunks = outer_shards if outer_shards is not None else inner_chunks
        create_array_kwargs["chunk_key_encoding"] = _chunk_key_encoding(
            data_array.shape, _key_chunks, dimension_separator_threshold
        )

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
    if multiple == 0:
        raise ValueError(
            f"Cannot round up to a multiple of zero (value={value}). "
            "The array likely has a zero-length axis, which is not supported."
        )
    return ((value + multiple - 1) // multiple) * multiple


def _chunk_key_encoding(
    shape: ChunkShape, chunks: ChunkShape, dimension_separator_threshold: int
) -> dict[str, str]:
    """Use flat chunk keys for small arrays and nested keys for larger arrays."""
    num_chunks = prod(
        (axis_size + chunk_size - 1) // chunk_size
        for axis_size, chunk_size in zip(shape, chunks, strict=True)
    )
    separator = "/" if num_chunks > dimension_separator_threshold else "."
    return {"name": "default", "separator": separator}


def _resolve_zarr_layout(
    *,
    shape: ChunkShape,
    chunks: ChunkSpec,
    shards: ShardSpec,
) -> tuple[ChunkShape, ChunkShape | None]:
    """Resolve the final Zarr chunk/shard layout from the supported writer inputs.

    The resolution order is:
    - explicit ``chunks``/``shards`` shapes when provided
    - integer element targets and ``'auto'`` sentinels, which derive power-of-two
      square or cubic layouts

    Explicit shapes also support ``0`` as a sentinel. In ``chunks``, ``0`` means span
    the full array axis for unsharded writes, or the full shard axis when ``shards`` is
    provided. In ``shards``, ``0`` means span the array axis, rounded up to a multiple
    of the resolved chunk size so the sharding codec remains valid.
    """
    if chunks == "auto":
        chunks = _DEFAULT_CHUNK_ELEMENTS
    if isinstance(chunks, int):
        chunks = _auto_shape(shape, chunks)
    if shards == "auto":
        shards = _DEFAULT_SHARD_ELEMENTS
    if isinstance(shards, int):
        shards = _auto_shards(shape, chunks, shards)

    return _normalize_explicit_shapes(shape, chunks, shards)


def _normalize_explicit_shapes(
    shape: ChunkShape,
    chunks: ChunkShape,
    shards: ChunkShape | None,
) -> tuple[ChunkShape, ChunkShape | None]:
    """Validate explicit layout shapes and expand any zero-valued span sentinels.

    A zero chunk size spans the corresponding shard axis when sharding is enabled,
    otherwise it spans the full array axis. A zero shard size spans the array axis,
    rounded up to a chunk multiple so zarr's sharding codec accepts the layout.
    """
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

    resolved_shards: ChunkShape | None = None
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


def _power_of_two_edge(elements: int, dimensions: int) -> int:
    """Return the power-of-two edge whose volume is closest to the target."""
    if isinstance(elements, bool) or elements <= 0:
        raise ValueError(f"elements must be a positive integer. Got {elements}.")

    edge = 1
    while (edge * 2) ** dimensions <= elements:
        edge *= 2
    larger_edge = edge * 2
    if larger_edge**dimensions - elements < elements - edge**dimensions:
        return larger_edge
    return edge


def _auto_shape(shape: ChunkShape, elements: int) -> ChunkShape:
    """Return a power-of-two square or cubic shape, trimmed to the array."""
    is_2d = len(shape) == 3 and shape[0] == 1
    dimensions = 2 if is_2d else len(shape)
    edge = _power_of_two_edge(elements, dimensions)
    return tuple(
        axis_size if is_2d and axis == 0 else min(edge, axis_size)
        for axis, axis_size in enumerate(shape)
    )


def _auto_shards(
    shape: ChunkShape,
    chunks: ChunkShape,
    elements: int,
) -> ChunkShape:
    """Return auto shard dimensions trimmed to the array and aligned to chunks."""
    target = _auto_shape(shape, elements)
    return tuple(
        shard_size
        if chunk_size == 0
        else _round_up_to_multiple(max(chunk_size, shard_size), chunk_size)
        for shard_size, chunk_size in zip(target, chunks, strict=True)
    )


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
    inner_chunks: ChunkShape,
    outer_shards: ChunkShape | None,
    create_array_kwargs: dict[str, Any],
    mango_attrs: dict[str, Any] | None,
    ome_zarr_version: OMEZarrVersion,
) -> None:
    """Write data as an OME-Zarr group, optionally using Zarr v3 sharding.

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
    inner_chunks: ChunkShape,
    outer_shards: ChunkShape | None,
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
