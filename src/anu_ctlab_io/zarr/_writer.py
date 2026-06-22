"""Write data to the ANU CTLab zarr data format."""

import logging
import warnings
from datetime import datetime
from enum import Enum
from math import prod
from pathlib import Path
from typing import Any, Literal, cast

import dask.array as da
import numpy as np
import zarr
from dask.base import compute as dask_compute
from dask.delayed import Delayed, delayed
from ome_zarr_models.v05.axes import Axis
from ome_zarr_models.v05.coordinate_transformations import (
    VectorScale,
    VectorTranslation,
)
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
type DownsampleMethod = Literal["strided", "mean", "mode"]


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
    multiscale: bool = True,
    downsample_method: DownsampleMethod = "strided",
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
    :param multiscale: For OME-Zarr output, write a multiscale pyramid where possible.
        Plain Zarr output remains single-scale.
    :param downsample_method: Downsampling method for multiscale OME-Zarr output.
        ``"strided"`` (default) takes every second voxel without averaging.
        ``"mean"`` averages each 2x block.
        ``"mode"`` chooses the smallest value when a block has tied modes.
    :param compute: If ``True`` (default), compute immediately. If ``False``, return
        a :any:`dask.delayed.Delayed` for deferred execution.
    :param extra_attrs: Additional attributes to include in mango metadata.
    """
    if isinstance(path, str):
        path = Path(path)

    logger.debug(
        "Writing dataset to Zarr: path=%s, datatype=%s, dataset_id=%s, "
        "ome_zarr_version=%s, chunks=%s, shards=%s, input_aligned_chunks=%s, "
        "dimension_separator_threshold=%s, multiscale=%s, downsample_method=%s, "
        "compute=%s, create_array_kwargs=%s, extra_attrs=%s",
        path,
        datatype,
        dataset_id,
        ome_zarr_version,
        chunks,
        shards,
        input_aligned_chunks,
        dimension_separator_threshold,
        multiscale,
        downsample_method,
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

    if downsample_method not in ("strided", "mean", "mode"):
        raise ValueError(
            "downsample_method must be one of 'strided', 'mean', or 'mode'. "
            f"Got {downsample_method!r}."
        )

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

    subchunks = chunks if shards is not None else None
    chunks = chunks if shards is None else shards

    mango_attrs: dict[str, Any] | None = None
    if datatype is not None:
        mango_attrs = _build_mango_attrs(
            dataset, datatype, dataset_id, history, extra_attrs
        )

    create_array_kwargs = dict(create_array_kwargs or {})

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
            rechunk_before_store=not (input_aligned_chunks or multiscale),
            multiscale=multiscale,
            downsample_method=downsample_method,
            dimension_separator_threshold=dimension_separator_threshold,
            compute=compute,
        )
    else:
        chunks, subchunks = _resolve_zarr_layout(
            shape=data_array.shape,
            chunks=chunks,
            subchunks=subchunks,
            aligned_chunks=data_array.chunks if input_aligned_chunks else None,
        )
        logger.debug(
            "Resolved Zarr layout: chunks=%s, subchunks=%s, rechunk_before_store=%s",
            chunks,
            subchunks,
            not input_aligned_chunks,
        )
        if dimension_separator_threshold is not None:
            create_array_kwargs["chunk_key_encoding"] = _chunk_key_encoding(
                data_array.shape, chunks, dimension_separator_threshold
            )
            logger.debug(
                "Selected chunk key encoding: key_chunks=%s, encoding=%s",
                chunks,
                create_array_kwargs["chunk_key_encoding"],
            )

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


def _regular_chunk_shape(chunks: tuple[tuple[int, ...], ...]) -> ChunkShape | None:
    regular_chunks: list[int] = []
    for axis_chunks in chunks:
        if not axis_chunks:
            return None
        regular_chunk = int(axis_chunks[0])
        internal_chunks = axis_chunks[:-1] if len(axis_chunks) > 1 else axis_chunks
        if any(int(chunk) != regular_chunk for chunk in internal_chunks):
            return None
        if int(axis_chunks[-1]) > regular_chunk:
            return None
        regular_chunks.append(regular_chunk)
    return tuple(regular_chunks)


def _can_downsample_chunks_by_two(chunks: tuple[tuple[int, ...], ...]) -> bool:
    regular_chunks = _regular_chunk_shape(chunks)
    if regular_chunks is None:
        return False
    return all(int(chunk) >= 2 for axis_chunks in chunks for chunk in axis_chunks)


def _downsampled_chunks(
    chunks: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple((int(chunk) + 1) // 2 for chunk in axis_chunks) for axis_chunks in chunks
    )


def _shape_from_chunks(chunks: tuple[tuple[int, ...], ...]) -> ChunkShape:
    return tuple(sum(axis_chunks) for axis_chunks in chunks)


def _array_fits_in_store_unit(
    shape: ChunkShape,
    store_unit_elements: int,
) -> bool:
    return prod(shape) <= store_unit_elements


def _cast_mean_downsampled_block(block: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return cast(
            np.ndarray, np.clip(np.rint(block), info.min, info.max).astype(dtype)
        )
    return block.astype(dtype)


def _mode_smallest_tie(
    block: np.ndarray, axis: int | tuple[int, ...] | None = None
) -> np.ndarray:
    axes = (
        tuple(range(block.ndim))
        if axis is None
        else ((axis,) if isinstance(axis, int) else tuple(axis))
    )
    if not axes:
        return block

    positive_axes = tuple(ax if ax >= 0 else block.ndim + ax for ax in axes)
    remaining_axes = tuple(ax for ax in range(block.ndim) if ax not in positive_axes)
    moved = np.moveaxis(block, remaining_axes + positive_axes, range(block.ndim))
    remaining_shape = moved.shape[: len(remaining_axes)]
    reduction_size = prod(moved.shape[len(remaining_axes) :])
    flattened = moved.reshape(*remaining_shape, reduction_size)

    if not remaining_shape:
        values, counts = np.unique(flattened, return_counts=True)
        return np.asarray(values[counts == counts.max()].min(), dtype=block.dtype)

    result = np.empty(remaining_shape, dtype=block.dtype)
    for index in np.ndindex(remaining_shape):
        values, counts = np.unique(flattened[index], return_counts=True)
        result[index] = values[counts == counts.max()].min()
    return result


def _downsample_block_by_two(
    block: np.ndarray,
    method: DownsampleMethod,
) -> np.ndarray:
    if method == "strided":
        return block[(slice(None, None, 2),) * block.ndim]

    if method == "mode":
        return _downsample_block_mode(block)

    downsampled = _downsample_block_mean(block)
    return _cast_mean_downsampled_block(downsampled, np.dtype(block.dtype))


def _downsampled_block_shape(block: np.ndarray) -> ChunkShape:
    return tuple((axis_size + 1) // 2 for axis_size in block.shape)


def _downsample_offset_parts(
    block: np.ndarray,
) -> list[tuple[np.ndarray, tuple[slice, ...]]]:
    parts = []
    for offsets in np.ndindex((2,) * block.ndim):
        part_index = tuple(slice(int(offset), None, 2) for offset in offsets)
        part = block[part_index]
        parts.append((part, tuple(slice(0, axis_size) for axis_size in part.shape)))
    return parts


def _downsample_block_mean(block: np.ndarray) -> np.ndarray:
    output_shape = _downsampled_block_shape(block)
    total = np.zeros(output_shape, dtype=np.float64)
    count = np.zeros(output_shape, dtype=np.uint8)
    for part, part_slice in _downsample_offset_parts(block):
        total[part_slice] += part
        count[part_slice] += 1
    return cast(np.ndarray, total / count)


def _downsample_block_mode(block: np.ndarray) -> np.ndarray:
    output_shape = _downsampled_block_shape(block)
    parts = _downsample_offset_parts(block)
    values = np.empty((len(parts), *output_shape), dtype=block.dtype)
    valid = np.zeros((len(parts), *output_shape), dtype=bool)
    for index, (part, part_slice) in enumerate(parts):
        destination: tuple[int | slice, ...] = (index, *part_slice)
        values[destination] = part
        valid[destination] = True

    order = np.argsort(values, axis=0, kind="stable")
    sorted_values = np.take_along_axis(values, order, axis=0)
    sorted_valid = np.take_along_axis(valid, order, axis=0)

    counts = np.zeros_like(values, dtype=np.uint8)
    for index in range(values.shape[0]):
        counts[index] = np.sum((values == sorted_values[index]) & valid, axis=0)
    counts = np.take_along_axis(counts, order, axis=0)
    counts[~sorted_valid] = 0

    best_count = np.max(counts, axis=0, keepdims=True)
    winners = sorted_valid & (counts == best_count)
    winner_indices = np.argmax(winners, axis=0, keepdims=True)
    return cast(
        np.ndarray, np.take_along_axis(sorted_values, winner_indices, axis=0)[0]
    )


def _chunk_starts(chunks: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    starts: list[tuple[int, ...]] = []
    for axis_chunks in chunks:
        axis_starts: list[int] = []
        offset = 0
        for chunk in axis_chunks:
            axis_starts.append(offset)
            offset += int(chunk)
        starts.append(tuple(axis_starts))
    return tuple(starts)


def _level_block_slices(
    block_index: tuple[int, ...],
    level_chunks: tuple[tuple[int, ...], ...],
    level_starts: tuple[tuple[int, ...], ...],
) -> tuple[slice, ...]:
    return tuple(
        slice(
            level_starts[axis][chunk_index],
            level_starts[axis][chunk_index] + level_chunks[axis][chunk_index],
        )
        for axis, chunk_index in enumerate(block_index)
    )


def _write_multiscale_chunk(
    block: np.ndarray,
    *,
    zarr_arrays: list[zarr.Array[Any]],
    block_index: tuple[int, ...],
    level_chunks: list[tuple[tuple[int, ...], ...]],
    level_starts: list[tuple[tuple[int, ...], ...]],
    downsample_method: DownsampleMethod,
) -> None:
    level_block = block
    for level, array in enumerate(zarr_arrays):
        array[
            _level_block_slices(block_index, level_chunks[level], level_starts[level])
        ] = level_block
        if level < len(zarr_arrays) - 1:
            level_block = _downsample_block_by_two(level_block, downsample_method)


def _combine_delayed(tasks: list[Delayed]) -> Delayed:
    return cast(Delayed, delayed(lambda *results: None)(*tasks))


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

    level_shapes: list[ChunkShape] = [data_array.shape]
    level_aligned_chunks: list[tuple[tuple[int, ...], ...]] = [data_array.chunks]
    level_layouts: list[tuple[ChunkShape, ChunkShape | None]] = [
        (level_chunks, level_subchunks)
    ]
    level_0_store_unit = level_subchunks or level_chunks
    level_0_store_unit_elements = prod(level_0_store_unit)
    if multiscale:
        while _can_downsample_chunks_by_two(level_aligned_chunks[-1]):
            if _array_fits_in_store_unit(level_shapes[-1], level_0_store_unit_elements):
                logger.debug(
                    "Stopping OME-Zarr pyramid at level %s because shape=%s fits "
                    "within one level-0 storage unit=%s.",
                    len(level_shapes) - 1,
                    level_shapes[-1],
                    level_0_store_unit,
                )
                break

            next_aligned_chunks = _downsampled_chunks(level_aligned_chunks[-1])
            next_shape = _shape_from_chunks(next_aligned_chunks)
            try:
                next_chunks, next_subchunks = _resolve_zarr_layout(
                    shape=next_shape,
                    chunks=chunks,
                    subchunks=subchunks,
                    aligned_chunks=next_aligned_chunks,
                )
            except ValueError:
                logger.debug(
                    "Stopping OME-Zarr pyramid before level %s because the layout is "
                    "incompatible with the downsampled chunk grid.",
                    len(level_shapes),
                )
                break
            level_shapes.append(next_shape)
            level_aligned_chunks.append(next_aligned_chunks)
            level_layouts.append((next_chunks, next_subchunks))

    ome_datasets: list[OMEDataset] = []
    for level in range(len(level_shapes)):
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
        zip(level_shapes, level_layouts, strict=True)
    ):
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
        zarr_arrays.append(array)

    if multiscale:
        if rechunk_before_store:
            raise RuntimeError("Multiscale OME-Zarr writes must not rechunk.")

        level_starts = [_chunk_starts(chunks) for chunks in level_aligned_chunks]
        blocks = data_array.to_delayed()  # type: ignore[no-untyped-call]
        tasks: list[Delayed] = [
            cast(
                Delayed,
                delayed(_write_multiscale_chunk, pure=False)(
                    blocks[block_index],
                    zarr_arrays=zarr_arrays,
                    block_index=block_index,
                    level_chunks=level_aligned_chunks,
                    level_starts=level_starts,
                    downsample_method=downsample_method,
                ),
            )
            for block_index in np.ndindex(blocks.shape)
        ]
        if compute:
            dask_compute(*tasks)  # type: ignore[no-untyped-call]
            return None
        return _combine_delayed(tasks)

    if rechunk_before_store:
        rechunked_arrays: list[da.Array] = []
        level_arrays = [data_array]
        for level_array, zarr_array in zip(level_arrays, zarr_arrays, strict=True):
            write_shape = zarr_array.shards or zarr_array.chunks
            logger.debug(
                "Rechunking input for OME-Zarr write: source_chunks=%s, write_shape=%s",
                level_array.chunks,
                write_shape,
            )
            rechunked_arrays.append(level_array.rechunk(write_shape))  # type: ignore[no-untyped-call]
        level_arrays = rechunked_arrays
    else:
        level_arrays = [data_array]
        logger.debug(
            "Skipping input rechunk for OME-Zarr write: source_chunks=%s",
            [level_array.chunks for level_array in level_arrays],
        )

    # dask's to_zarr internally calls normalize_chunks("auto", ...) which can produce
    # chunk sizes that are not multiples of the shard shape, causing misaligned writes
    # that manifest as large regions of zeros in the output. Using da.store directly
    # bypasses that internal rechunk entirely, writing each dask chunk straight into
    # its corresponding region in the zarr array.
    result = da.store(level_arrays, zarr_arrays, lock=False, compute=compute)  # type: ignore[arg-type]
    if compute:
        return None
    if isinstance(result, Delayed):
        return result
    return _combine_delayed(list(result))


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
