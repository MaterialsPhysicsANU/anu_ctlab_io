"""Write data to the ANU CTLab zarr data format."""

import logging
import warnings
from datetime import datetime
from enum import Enum
from math import prod
from pathlib import Path
from typing import Any, Literal

import dask.array as da
import zarr
from dask.delayed import Delayed
from ome_zarr_models.v05.axes import Axis
from ome_zarr_models.v05.coordinate_transformations import VectorScale
from ome_zarr_models.v05.multiscales import Dataset as OMEDataset
from ome_zarr_models.v05.multiscales import Multiscale

from anu_ctlab_io._dataset import Dataset
from anu_ctlab_io._datatype import DataType
from anu_ctlab_io._parse_history import History

__all__ = ["OMEZarrVersion", "dataset_to_zarr"]

logger = logging.getLogger(__name__)

_DEFAULT_CHUNK_ELEMENTS = max(8192**2, 512**3)
_DEFAULT_SUBCHUNK_ELEMENTS = max(256**2, 32**3)

type ChunkShape = tuple[int, ...]
type ChunkSpec = ChunkShape | int | Literal["auto"]


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
    shards: ChunkSpec | None = "auto",
    create_array_kwargs: dict[str, Any] | None = None,
    dimension_separator_threshold: int | None = 64,
    input_aligned_chunks: bool = False,
    compute: bool = True,
    **extra_attrs: Any,
) -> Delayed | None:
    """Write a :any:`Dataset` to Zarr format.

    By default, sharded writes use power-of-two square or cubic shapes targeting
    ``32**3`` elements for Zarr chunks and ``512**3`` elements for shards. Shapes
    are trimmed to the array dimensions, and shards are rounded up to chunk
    multiples. With ``shards=None``, ``chunks='auto'`` uses the larger unsharded
    write target of ``512**3`` elements.

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
        With sharding enabled, ``'auto'`` uses a default target corresponding to ``32**3`` or ``256**2`` elements.
        With ``shards=None``, ``'auto'`` uses the larger unsharded target corresponding to ``512**3`` or ``8192**2`` elements.
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
    :param input_aligned_chunks: Choose Zarr chunks/shards aligned to the input dask
        chunk grid and skip rechunking before storing. With sharding is enabled, shards
        are aligned to the dask chunks and inner chunks evenly divide each shard.
        With ``shards=None``, chunks are aligned to the dask chunks.
    :param compute: If ``True`` (default), compute immediately. If ``False``, return
        a :any:`dask.delayed.Delayed` for deferred execution.
    :param extra_attrs: Additional attributes to include in mango metadata.
    """
    if isinstance(path, str):
        path = Path(path)

    logger.debug(
        "Writing dataset to Zarr: path=%s, datatype=%s, dataset_id=%s, "
        "ome_zarr_version=%s, chunks=%s, shards=%s, input_aligned_chunks=%s, "
        "dimension_separator_threshold=%s, compute=%s, create_array_kwargs=%s, "
        "extra_attrs=%s",
        path,
        datatype,
        dataset_id,
        ome_zarr_version,
        chunks,
        shards,
        input_aligned_chunks,
        dimension_separator_threshold,
        compute,
        create_array_kwargs,
        sorted(extra_attrs),
    )

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
    logger.debug(
        "Input dask array: shape=%s, dtype=%s, chunks=%s, chunksize=%s, npartitions=%s",
        data_array.shape,
        data_array.dtype,
        data_array.chunks,
        data_array.chunksize,
        data_array.npartitions,
    )

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

    if isinstance(shards, tuple) and chunks == "auto" and not input_aligned_chunks:
        raise ValueError("shards cannot be provided without explicit chunks")

    chunks, subchunks = _resolve_zarr_layout(
        shape=data_array.shape,
        chunks=shards if shards is not None else chunks,
        subchunks=chunks if shards is not None else None,
        aligned_chunks=data_array.chunks if input_aligned_chunks else None,
    )
    logger.debug(
        "Resolved Zarr layout: chunks=%s, subchunks=%s, rechunk_before_store=%s",
        chunks,
        subchunks,
        not input_aligned_chunks,
    )

    mango_attrs: dict[str, Any] | None = None
    if datatype is not None:
        mango_attrs = _build_mango_attrs(
            dataset, datatype, dataset_id, history, extra_attrs
        )

    create_array_kwargs = dict(create_array_kwargs or {})
    if dimension_separator_threshold is not None:
        create_array_kwargs["chunk_key_encoding"] = _chunk_key_encoding(
            data_array.shape, chunks, dimension_separator_threshold
        )
        logger.debug(
            "Selected chunk key encoding: key_chunks=%s, encoding=%s",
            chunks,
            create_array_kwargs["chunk_key_encoding"],
        )

    if ome_zarr_version is not None:
        logger.debug("Writing OME-Zarr group with version %s", ome_zarr_version)
        return _write_ome_zarr_group(
            data_array,
            path,
            dataset,
            chunks=chunks,
            subchunks=subchunks,
            create_array_kwargs=create_array_kwargs,
            mango_attrs=mango_attrs,
            ome_zarr_version=ome_zarr_version,
            rechunk_before_store=not input_aligned_chunks,
            compute=compute,
        )
    else:
        logger.debug("Writing plain Zarr array")
        return _write_zarr_array(
            data_array,
            path,
            dataset,
            chunks=chunks,
            subchunks=subchunks,
            create_array_kwargs=create_array_kwargs,
            mango_attrs=mango_attrs,
            rechunk_before_store=not input_aligned_chunks,
            compute=compute,
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
    subchunks: ChunkSpec | None,
    aligned_chunks: tuple[tuple[int, ...], ...] | None = None,
) -> tuple[ChunkShape, ChunkShape | None]:
    """Resolve the internal Zarr chunks/subchunks layout.

    This helper uses the writer's internal naming, not the public
    ``dataset_to_zarr`` parameter names. Before this function is called,
    public arguments are adapted as follows:

    - ``shards is None``: ``chunks`` is the public ``chunks`` value and
      ``subchunks`` is ``None``.
    - ``shards is not None``: ``chunks`` is the public ``shards`` value and
      ``subchunks`` is the public ``chunks`` value.

    The return value follows the same internal naming. ``chunks`` is the outer
    write shape. ``subchunks`` is ``None`` for unsharded writes, or the inner
    Zarr chunk shape stored inside each sharded chunk.

    Without ``aligned_chunks``:

    - Unsharded ``chunks='auto'`` uses ``_DEFAULT_CHUNK_ELEMENTS`` and is
      trimmed to the array shape.
    - Sharded ``chunks='auto'`` uses ``_DEFAULT_CHUNK_ELEMENTS`` for the outer
      sharded chunk; ``subchunks='auto'`` uses ``_DEFAULT_SUBCHUNK_ELEMENTS``
      for the inner Zarr chunks.
    - Integer specs are element targets. Inner subchunks are derived as a
      power-of-two square/cubic shape trimmed to the array shape. Outer chunks
      are derived the same way, then rounded up to a multiple of the resolved
      subchunk shape so the sharding codec can store them.
    - Tuple specs are explicit shapes. A zero in unsharded ``chunks`` spans the
      full array axis. A zero in sharded ``chunks`` spans the full array axis,
      rounded up to a subchunk multiple. A zero in ``subchunks`` spans the
      corresponding resolved outer chunk axis.

    With ``aligned_chunks``, no writer rechunking is required. The aligned grid
    must be regular except for smaller final edge chunks. Unsharded chunks must
    evenly divide the aligned chunk shape. Sharded outer chunks must evenly
    divide the aligned chunk shape, and subchunks must evenly divide the outer
    chunk shape. Tuple zeros span the aligned chunk shape for ``chunks`` and the
    resolved outer chunk shape for ``subchunks``. Integer and ``'auto'`` specs
    choose the largest divisor of the containing shape that does not exceed the
    corresponding automatic target.
    """
    if aligned_chunks is not None:
        return _resolve_input_aligned_zarr_layout(
            shape=shape,
            aligned_chunks=aligned_chunks,
            chunks=chunks,
            subchunks=subchunks,
        )

    if subchunks is None:
        if chunks == "auto":
            chunks = _DEFAULT_CHUNK_ELEMENTS
        if isinstance(chunks, int):
            chunks = _auto_shape(shape, chunks)

        resolved_chunks, _ = _normalize_explicit_shapes(
            shape, chunks=chunks, subchunks=None
        )
        return resolved_chunks, None

    if chunks == "auto":
        chunks = _DEFAULT_CHUNK_ELEMENTS
    if subchunks == "auto":
        subchunks = _DEFAULT_SUBCHUNK_ELEMENTS
    if isinstance(subchunks, int):
        subchunks = _auto_shape(shape, subchunks)
    if isinstance(chunks, int):
        chunks = _auto_shards(shape, subchunks, chunks)

    resolved_chunks, resolved_subchunks = _normalize_explicit_shapes(
        shape, chunks=chunks, subchunks=subchunks
    )
    return resolved_chunks, resolved_subchunks


def _resolve_input_aligned_zarr_layout(
    *,
    shape: ChunkShape,
    aligned_chunks: tuple[tuple[int, ...], ...],
    chunks: ChunkSpec,
    subchunks: ChunkSpec | None,
) -> tuple[ChunkShape, ChunkShape | None]:
    aligned_chunk_shape = _regular_aligned_chunk_shape(aligned_chunks)
    axis_spans_array = _aligned_chunk_axes_span_array(shape, aligned_chunks)

    if subchunks is None:
        if chunks == "auto":
            resolved_chunks = aligned_chunk_shape
        elif isinstance(chunks, tuple):
            resolved_chunks = _normalize_input_aligned_tuple(
                chunks, aligned_chunk_shape
            )
        else:
            resolved_chunks = _input_aligned_subchunk_shape(
                shape, aligned_chunk_shape, chunks
            )
        _validate_aligned_write_shape(aligned_chunk_shape, resolved_chunks)
        return resolved_chunks, None

    if chunks == "auto":
        resolved_chunks = _input_aligned_chunk_shape(
            shape=shape,
            aligned_chunk_shape=aligned_chunk_shape,
            axis_spans_array=axis_spans_array,
            subchunks=subchunks,
        )
    elif isinstance(chunks, tuple):
        resolved_chunks = _normalize_input_aligned_tuple(chunks, aligned_chunk_shape)
    else:
        resolved_chunks = _input_aligned_subchunk_shape(
            shape, aligned_chunk_shape, chunks
        )
    _validate_aligned_write_shape(
        aligned_chunk_shape, resolved_chunks, axis_spans_array=axis_spans_array
    )

    resolved_subchunks = (
        _normalize_input_aligned_tuple(subchunks, resolved_chunks)
        if isinstance(subchunks, tuple)
        else _input_aligned_subchunk_shape(shape, resolved_chunks, subchunks)
    )
    _validate_subchunks_divide_write_shape(resolved_subchunks, resolved_chunks)
    return resolved_chunks, resolved_subchunks


def _aligned_chunk_axes_span_array(
    shape: ChunkShape,
    aligned_chunks: tuple[tuple[int, ...], ...],
) -> tuple[bool, ...]:
    return tuple(
        len(axis_chunks) == 1 and int(axis_chunks[0]) == axis_size
        for axis_size, axis_chunks in zip(shape, aligned_chunks, strict=True)
    )


def _regular_aligned_chunk_shape(
    aligned_chunks: tuple[tuple[int, ...], ...],
) -> ChunkShape:
    regular_chunks: list[int] = []
    for axis, axis_chunks in enumerate(aligned_chunks):
        if len(axis_chunks) == 0:
            raise ValueError("input_aligned_chunks requires non-empty aligned chunks")

        regular_chunk = int(axis_chunks[0])
        if regular_chunk <= 0:
            raise ValueError("input_aligned_chunks requires positive aligned chunks")

        internal_chunks = axis_chunks[:-1] if len(axis_chunks) > 1 else axis_chunks
        if any(int(chunk) != regular_chunk for chunk in internal_chunks):
            raise ValueError(
                "input_aligned_chunks requires regular aligned chunks; "
                f"axis {axis} has chunks {axis_chunks}."
            )
        if int(axis_chunks[-1]) > regular_chunk:
            raise ValueError(
                "input_aligned_chunks requires regular aligned chunks; "
                f"axis {axis} has chunks {axis_chunks}."
            )
        regular_chunks.append(regular_chunk)

    return tuple(regular_chunks)


def _normalize_input_aligned_tuple(
    spec: ChunkShape,
    span: ChunkShape,
) -> ChunkShape:
    if len(spec) != len(span):
        raise ValueError(
            f"input-aligned layout must have {len(span)} dimensions. Got {spec}."
        )
    resolved = tuple(
        span_size if size == 0 else size
        for size, span_size in zip(spec, span, strict=True)
    )
    if any(size <= 0 for size in resolved):
        raise ValueError(f"input-aligned layout sizes must be positive. Got {spec}.")
    return resolved


def _input_aligned_chunk_shape(
    *,
    shape: ChunkShape,
    aligned_chunk_shape: ChunkShape,
    axis_spans_array: tuple[bool, ...],
    subchunks: ChunkSpec,
) -> ChunkShape:
    subchunk_shape = _input_aligned_chunk_expansion_subchunk_shape(shape, subchunks)
    auto_chunk_edge = _input_aligned_auto_chunk_edge(shape)
    return tuple(
        _round_up_to_multiple(axis_size, subchunk_size)
        if axis_spans and axis_size >= auto_chunk_edge
        else aligned_size
        for axis_size, aligned_size, axis_spans, subchunk_size in zip(
            shape, aligned_chunk_shape, axis_spans_array, subchunk_shape, strict=True
        )
    )


def _input_aligned_auto_chunk_edge(shape: ChunkShape) -> int:
    is_2d = len(shape) == 3 and shape[0] == 1
    dimensions = 2 if is_2d else len(shape)
    return _power_of_two_edge(_DEFAULT_CHUNK_ELEMENTS, dimensions)


def _input_aligned_chunk_expansion_subchunk_shape(
    shape: ChunkShape,
    subchunks: ChunkSpec,
) -> ChunkShape:
    if subchunks == "auto":
        subchunks = _DEFAULT_SUBCHUNK_ELEMENTS
    if isinstance(subchunks, int):
        return _auto_shape(shape, subchunks)
    assert isinstance(subchunks, tuple)
    return tuple(
        axis_size if size == 0 else size
        for axis_size, size in zip(shape, subchunks, strict=True)
    )


def _input_aligned_subchunk_shape(
    shape: ChunkShape,
    write_shape: ChunkShape,
    chunks: ChunkSpec,
) -> ChunkShape:
    if chunks == "auto":
        chunks = _DEFAULT_SUBCHUNK_ELEMENTS
    if isinstance(chunks, int):
        target = _auto_shape(shape, chunks)
        return tuple(
            _largest_divisor_at_most(write_size, target_size)
            for write_size, target_size in zip(write_shape, target, strict=True)
        )
    assert isinstance(chunks, tuple)
    return chunks


def _largest_divisor_at_most(value: int, limit: int) -> int:
    limit = min(value, limit)
    for candidate in range(limit, 0, -1):
        if value % candidate == 0:
            return candidate
    raise ValueError(f"Could not find a positive divisor for {value}.")


def _validate_aligned_write_shape(
    aligned_chunk_shape: ChunkShape,
    write_shape: ChunkShape,
    *,
    axis_spans_array: tuple[bool, ...] | None = None,
) -> None:
    if axis_spans_array is None:
        axis_spans_array = (False,) * len(aligned_chunk_shape)

    if len(write_shape) != len(aligned_chunk_shape):
        raise ValueError(
            "write shape must have the same dimensions as aligned chunks. "
            f"Got {write_shape} and {aligned_chunk_shape}."
        )
    for axis, (write_size, aligned_size, axis_spans) in enumerate(
        zip(write_shape, aligned_chunk_shape, axis_spans_array, strict=True)
    ):
        if write_size <= 0:
            raise ValueError(f"write shape sizes must be positive. Got {write_shape}.")
        if axis_spans:
            if write_size < aligned_size:
                raise ValueError(
                    "input_aligned_chunks requires expanded span axes to cover the "
                    f"aligned chunk. Axis {axis}: {write_size} is smaller than {aligned_size}."
                )
            continue
        if aligned_size % write_size != 0:
            raise ValueError(
                "input_aligned_chunks requires write shape to evenly divide the "
                f"aligned chunk shape. Axis {axis}: {write_size} does not divide {aligned_size}."
            )


def _validate_subchunks_divide_write_shape(
    subchunks: ChunkShape,
    write_shape: ChunkShape,
) -> None:
    if len(subchunks) != len(write_shape):
        raise ValueError(
            "subchunks must have the same dimensions as the write shape. "
            f"Got {subchunks} and {write_shape}."
        )
    for axis, (subchunk, write_size) in enumerate(
        zip(subchunks, write_shape, strict=True)
    ):
        if subchunk <= 0:
            raise ValueError(f"subchunk sizes must be positive. Got {subchunks}.")
        if write_size % subchunk != 0:
            raise ValueError(
                "input_aligned_chunks requires subchunks to evenly divide the "
                f"write shape. Axis {axis}: {subchunk} does not divide {write_size}."
            )


def _normalize_explicit_shapes(
    shape: ChunkShape,
    chunks: ChunkShape,
    subchunks: ChunkShape | None,
) -> tuple[ChunkShape, ChunkShape | None]:
    """Validate explicit layout shapes and expand any zero-valued span sentinels.

    A zero in unsharded ``chunks`` spans the full array axis. A zero in sharded
    ``chunks`` spans the full array axis, rounded up to a subchunk multiple. A
    zero in ``subchunks`` spans the corresponding resolved chunk axis.
    """
    if len(chunks) != len(shape):
        raise ValueError(
            f"chunks must have the same number of dimensions as the array shape. Got chunks={chunks}, shape={shape}."
        )
    if subchunks is not None and len(subchunks) != len(shape):
        raise ValueError(
            f"subchunks must have the same number of dimensions as the array shape. Got subchunks={subchunks}, shape={shape}."
        )

    resolved_chunks_list: list[int] = []
    for axis_size, chunk_size, subchunk_size in zip(
        shape, chunks, subchunks or (0,) * len(shape), strict=True
    ):
        if chunk_size != 0:
            resolved_chunks_list.append(chunk_size)
        elif subchunks is None or subchunk_size == 0:
            resolved_chunks_list.append(axis_size)
        else:
            resolved_chunks_list.append(_round_up_to_multiple(axis_size, subchunk_size))
    resolved_chunks = tuple(resolved_chunks_list)

    resolved_subchunks: ChunkShape | None = None
    if subchunks is not None:
        resolved_subchunks = tuple(
            chunk_size if subchunk_size == 0 else subchunk_size
            for subchunk_size, chunk_size in zip(
                subchunks, resolved_chunks, strict=True
            )
        )

    return resolved_chunks, resolved_subchunks


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
    """Return auto outer dimensions trimmed to the array and aligned to chunks."""
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
    *,
    chunks: ChunkShape,
    subchunks: ChunkShape | None,
    create_array_kwargs: dict[str, Any],
    mango_attrs: dict[str, Any] | None,
    ome_zarr_version: OMEZarrVersion,
    rechunk_before_store: bool,
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
        chunks=subchunks or chunks,
        shards=chunks if subchunks is not None else None,
        dtype=data_array.dtype,
        dimension_names=list(dimension_names),
        overwrite=True,
        **create_array_kwargs,
    )
    logger.debug(
        "Created OME-Zarr array: path=%s, shape=%s, dtype=%s, chunks=%s, shards=%s, "
        "dimension_names=%s",
        path,
        array.shape,
        array.dtype,
        array.chunks,
        array.shards,
        dimension_names,
    )

    # Always rechunk to the write granularity of the Zarr array before writing.
    # If using sharding, the write granularity is the shard shape (`.shards`).
    # If not using sharding, `.shards` is None and `chunks` is used as the write granularity.
    # Note that the rechunk is lazy if the array is already in the desired chunk shape.
    # The straddling chunks/shards w.r.t the array shape need no special handling.
    if rechunk_before_store:
        write_shape = array.shards or array.chunks
        logger.debug(
            "Rechunking input for OME-Zarr write: source_chunks=%s, write_shape=%s",
            data_array.chunks,
            write_shape,
        )
        data_array = data_array.rechunk(write_shape)  # type: ignore[no-untyped-call]
    else:
        logger.debug(
            "Skipping input rechunk for OME-Zarr write: source_chunks=%s",
            data_array.chunks,
        )

    # dask's to_zarr internally calls normalize_chunks("auto", ...) which can produce
    # chunk sizes that are not multiples of the shard shape, causing misaligned writes
    # that manifest as large regions of zeros in the output. Using da.store directly
    # bypasses that internal rechunk entirely, writing each dask chunk straight into
    # its corresponding region in the zarr array.
    result: Delayed | None = da.store(data_array, array, lock=False, compute=compute)  # type: ignore[arg-type]
    return result


def _write_zarr_array(
    data_array: da.Array,
    path: Path,
    dataset: Dataset,
    *,
    chunks: ChunkShape,
    subchunks: ChunkShape | None,
    create_array_kwargs: dict[str, Any],
    mango_attrs: dict[str, Any] | None,
    rechunk_before_store: bool,
    compute: bool = True,
) -> Delayed | None:
    """Write data as a simple Zarr V3 array with mango metadata.

    When ``subchunks`` is provided, Zarr v3 sharding is used:
    - chunks = primary write chunks
    - subchunks = subdivisions within each sharded chunk
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
        chunks=subchunks or chunks,
        shards=chunks if subchunks is not None else None,
        dtype=data_array.dtype,
        dimension_names=list(dimension_names),
        overwrite=True,
        **create_array_kwargs,
    )
    logger.debug(
        "Created Zarr array: path=%s, shape=%s, dtype=%s, chunks=%s, shards=%s, "
        "dimension_names=%s",
        path,
        array.shape,
        array.dtype,
        array.chunks,
        array.shards,
        dimension_names,
    )

    if mango_attrs:
        array.attrs["mango"] = mango_attrs

    # Always rechunk to the write granularity of the Zarr array before writing.
    # See comment in _write_ome_zarr_group for explanation.
    if rechunk_before_store:
        write_shape = array.shards or array.chunks
        logger.debug(
            "Rechunking input for Zarr write: source_chunks=%s, write_shape=%s",
            data_array.chunks,
            write_shape,
        )
        data_array = data_array.rechunk(write_shape)  # type: ignore[no-untyped-call]
    else:
        logger.debug(
            "Skipping input rechunk for Zarr write: source_chunks=%s",
            data_array.chunks,
        )

    result: Delayed | None = da.store(data_array, array, lock=False, compute=compute)  # type: ignore[arg-type]
    return result
