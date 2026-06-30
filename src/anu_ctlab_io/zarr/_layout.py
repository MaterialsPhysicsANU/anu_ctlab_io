"""Resolve Zarr chunk and shard layouts."""

from math import log2, prod
from typing import Literal

_DEFAULT_CHUNK_ELEMENTS = max(8192**2, 512**3)
_DEFAULT_SUBCHUNK_ELEMENTS = max(256**2, 32**3)

type ChunkShape = tuple[int, ...]
type ChunkSpec = ChunkShape | int | Literal["auto"]


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
    resolved outer chunk shape for ``subchunks``. Integer and ``'auto'`` subchunk
    specs choose a divisor of the containing shard by scoring target element
    closeness and shape balance.
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
    dimensions = len(_balanced_shape_axes(shape))
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
        return _best_divisor_subchunk_shape(shape, write_shape, chunks)
    assert isinstance(chunks, tuple)
    return chunks


def _best_divisor_subchunk_shape(
    shape: ChunkShape,
    write_shape: ChunkShape,
    target_elements: int,
) -> ChunkShape:
    if isinstance(target_elements, bool) or target_elements <= 0:
        raise ValueError(f"elements must be a positive integer. Got {target_elements}.")

    active_axes = _balanced_shape_axes(shape)
    target_elements = min(
        _effective_target_elements(shape, target_elements), prod(write_shape)
    )
    best_score: tuple[float, float, float, int, ChunkShape] | None = None
    for candidate in _divisor_shape_candidates(write_shape):
        candidate_elements = prod(candidate)
        target_score = abs(log2(candidate_elements / target_elements))
        active_sizes = [candidate[axis] for axis in active_axes]
        balance_score = log2(max(active_sizes) / min(active_sizes))
        score = (
            target_score + balance_score,
            balance_score,
            target_score,
            -candidate_elements,
            candidate,
        )
        if best_score is None or score < best_score:
            best_score = score

    assert best_score is not None
    return best_score[-1]


def _balanced_shape_axes(shape: ChunkShape) -> tuple[int, ...]:
    active_axes = tuple(axis for axis, axis_size in enumerate(shape) if axis_size > 1)
    return active_axes or tuple(range(len(shape)))


def _effective_target_elements(shape: ChunkShape, target_elements: int) -> int:
    dimensions = len(_balanced_shape_axes(shape))
    edge = _power_of_two_edge(target_elements, dimensions)
    return int(edge**dimensions)


def _divisor_shape_candidates(write_shape: ChunkShape) -> list[ChunkShape]:
    candidates: list[ChunkShape] = [()]
    for axis_size in write_shape:
        candidates = [
            (*candidate, divisor)
            for candidate in candidates
            for divisor in _divisors(axis_size)
        ]
    return candidates


def _divisors(value: int) -> tuple[int, ...]:
    if value <= 0:
        raise ValueError(f"Cannot find divisors for non-positive value {value}.")

    lower: list[int] = []
    upper: list[int] = []
    candidate = 1
    while candidate * candidate <= value:
        if value % candidate == 0:
            lower.append(candidate)
            if candidate * candidate != value:
                upper.append(value // candidate)
        candidate += 1
    return (*lower, *reversed(upper))


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
    active_axes = _balanced_shape_axes(shape)
    dimensions = len(active_axes)
    edge = _power_of_two_edge(elements, dimensions)
    return tuple(
        min(edge, axis_size) if axis in active_axes else axis_size
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
