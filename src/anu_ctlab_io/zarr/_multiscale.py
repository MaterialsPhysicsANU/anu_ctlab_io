"""Build and write OME-Zarr multiscale pyramids."""

from collections.abc import Sequence
from math import prod
from pathlib import Path
from typing import Any, Literal, cast

import dask.array as da
import numpy as np
import zarr
from dask.base import tokenize
from dask.delayed import Delayed, delayed

from anu_ctlab_io.zarr._layout import ChunkShape

type DownsampleMethod = Literal["strided", "mean", "mode"]


def _can_downsample_shape_by_two(shape: ChunkShape) -> bool:
    return all(axis_size >= 2 for axis_size in shape)


def _chunk_grid_preserves_downsample_alignment(
    chunks: tuple[tuple[int, ...], ...],
) -> bool:
    """Return whether each chunk starts on a global 2x downsampling boundary."""
    for axis_chunks in chunks:
        if not axis_chunks:
            return False
        if any(int(chunk) <= 0 for chunk in axis_chunks):
            return False
        if any(int(chunk) % 2 for chunk in axis_chunks[:-1]):
            return False
    return True


def _downsampled_chunks(
    chunks: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple((int(chunk) + 1) // 2 for chunk in axis_chunks) for axis_chunks in chunks
    )


def _downsampled_shape(shape: ChunkShape) -> ChunkShape:
    return tuple((axis_size + 1) // 2 for axis_size in shape)


def _safe_downsample_rechunk_shape(
    shape: ChunkShape,
    preferred_chunks: ChunkShape,
) -> ChunkShape:
    return tuple(
        axis_size
        if preferred_chunk >= axis_size
        else max(2, preferred_chunk - preferred_chunk % 2)
        for axis_size, preferred_chunk in zip(shape, preferred_chunks, strict=True)
    )


def _array_fits_in_store_unit(
    shape: ChunkShape,
    store_unit_elements: int,
) -> bool:
    return prod(shape) <= store_unit_elements


def _disable_single_subchunk_sharding(
    chunks: ChunkShape,
    subchunks: ChunkShape | None,
) -> tuple[ChunkShape, ChunkShape | None]:
    if subchunks == chunks:
        return chunks, None
    return chunks, subchunks


def _cast_mean_downsampled_block(block: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return cast(
            np.ndarray, np.clip(np.rint(block), info.min, info.max).astype(dtype)
        )
    return block.astype(dtype)


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


def _downsample_array_by_two(
    data_array: da.Array,
    method: DownsampleMethod,
) -> da.Array:
    return cast(
        da.Array,
        da.map_blocks(  # type: ignore[no-untyped-call]
            _downsample_block_by_two,
            data_array,
            method,
            dtype=data_array.dtype,
            chunks=_downsampled_chunks(data_array.chunks),
        ),
    )


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


def _complete_delayed_writes(*_results: Any) -> None:
    """Barrier task used to make write dependencies explicit in Dask graphs."""
    return None


def _combine_delayed(tasks: Sequence[Delayed], *, dask_key_name: str) -> Delayed:
    return cast(
        Delayed,
        delayed(_complete_delayed_writes)(*tasks, dask_key_name=dask_key_name),
    )


def _chunk_offsets(chunks: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    offsets: list[tuple[int, ...]] = []
    for axis_chunks in chunks:
        axis_offsets = [0]
        for chunk in axis_chunks[:-1]:
            axis_offsets.append(axis_offsets[-1] + int(chunk))
        offsets.append(tuple(axis_offsets))
    return tuple(offsets)


def _chunk_region(
    chunks: tuple[tuple[int, ...], ...],
    offsets: tuple[tuple[int, ...], ...],
    index: tuple[int, ...],
) -> tuple[slice, ...]:
    return tuple(
        slice(
            axis_offsets[axis_index],
            axis_offsets[axis_index] + axis_chunks[axis_index],
        )
        for axis_chunks, axis_offsets, axis_index in zip(
            chunks, offsets, index, strict=True
        )
    )


def _regular_chunks_for_shape(
    shape: ChunkShape,
    chunk_shape: ChunkShape,
) -> tuple[tuple[int, ...], ...]:
    axis_chunks: list[tuple[int, ...]] = []
    for axis_size, chunk_size in zip(shape, chunk_shape, strict=True):
        chunks = [chunk_size] * (axis_size // chunk_size)
        remainder = axis_size % chunk_size
        if remainder:
            chunks.append(remainder)
        axis_chunks.append(tuple(chunks) or (0,))
    return tuple(axis_chunks)


def _chunk_grid_aligns_with_storage(
    shape: ChunkShape,
    chunks: tuple[tuple[int, ...], ...],
    storage_unit: ChunkShape,
) -> bool:
    """Return whether chunk boundaries cannot split a storage chunk or shard."""
    for axis_size, axis_chunks, storage_size in zip(
        shape, chunks, storage_unit, strict=True
    ):
        offset = 0
        for chunk in axis_chunks[:-1]:
            offset += int(chunk)
            if offset != axis_size and offset % storage_size != 0:
                return False
    return True


def _downsample_region_by_two(region: tuple[slice, ...]) -> tuple[slice, ...]:
    return tuple(
        slice(cast(int, item.start) // 2, (cast(int, item.stop) + 1) // 2)
        for item in region
    )


def _upsampled_source_region(
    region: tuple[slice, ...],
    source_shape: ChunkShape,
) -> tuple[slice, ...]:
    return tuple(
        slice(cast(int, item.start) * 2, min(cast(int, item.stop) * 2, axis_size))
        for item, axis_size in zip(region, source_shape, strict=True)
    )


def _write_multiscale_prefix_block(
    block: np.ndarray,
    arrays: Sequence[zarr.Array[Any]],
    region: tuple[slice, ...],
    method: DownsampleMethod,
) -> None:
    """Write one source block to level 0 and aligned in-memory pyramid levels."""
    current_block = block
    current_region = region
    for level, array in enumerate(arrays):
        array[current_region] = current_block
        if level < len(arrays) - 1:
            current_block = _downsample_block_by_two(current_block, method)
            current_region = _downsample_region_by_two(current_region)


def _write_downsampled_zarr_region(
    source_path: Path,
    destination_path: Path,
    source_shape: ChunkShape,
    destination_region: tuple[slice, ...],
    method: DownsampleMethod,
    _dependency: Any,
) -> None:
    """Read a source Zarr region, downsample it, and write one destination region."""
    source = zarr.open_array(source_path, mode="r")
    destination = zarr.open_array(destination_path, mode="r+")
    source_region = _upsampled_source_region(destination_region, source_shape)
    source_block = np.asarray(source[source_region])
    destination[destination_region] = _downsample_block_by_two(source_block, method)


def _direct_multiscale_prefix_level_count(
    level_shapes: Sequence[ChunkShape],
    level_layouts: Sequence[tuple[ChunkShape, ChunkShape | None]],
    source_chunks: tuple[tuple[int, ...], ...],
) -> int:
    count = 1
    current_chunks = source_chunks
    while count < len(level_shapes):
        if not _chunk_grid_preserves_downsample_alignment(current_chunks):
            break

        next_chunks = _downsampled_chunks(current_chunks)
        storage_unit = level_layouts[count][0]
        if not _chunk_grid_aligns_with_storage(
            level_shapes[count],
            next_chunks,
            storage_unit,
        ):
            break

        count += 1
        current_chunks = next_chunks
    return count


def _build_direct_ome_zarr_write(
    data_array: da.Array,
    zarr_path: Path,
    zarr_arrays: Sequence[zarr.Array[Any]],
    level_shapes: Sequence[ChunkShape],
    level_layouts: Sequence[tuple[ChunkShape, ChunkShape | None]],
    downsample_method: DownsampleMethod,
) -> Delayed:
    """Build explicit delayed tasks for OME-Zarr level writes."""
    write_token = tokenize(str(zarr_path))
    prefix_level_count = _direct_multiscale_prefix_level_count(
        level_shapes,
        level_layouts,
        data_array.chunks,
    )

    block_offsets = _chunk_offsets(data_array.chunks)
    delayed_blocks = data_array.to_delayed()  # type: ignore[no-untyped-call]
    prefix_arrays = zarr_arrays[:prefix_level_count]
    prefix_tasks: list[Delayed] = []
    for index in np.ndindex(delayed_blocks.shape):
        region = _chunk_region(data_array.chunks, block_offsets, index)
        prefix_tasks.append(
            cast(
                Delayed,
                delayed(_write_multiscale_prefix_block)(
                    delayed_blocks[index],
                    prefix_arrays,
                    region,
                    downsample_method,
                ),
            )
        )

    barrier = _combine_delayed(
        prefix_tasks,
        dask_key_name=f"complete-ome-zarr-prefix-writes-{write_token}",
    )
    for level in range(prefix_level_count, len(level_shapes)):
        destination_path = zarr_path / str(level)
        source_path = zarr_path / str(level - 1)
        source_shape = level_shapes[level - 1]
        storage_unit = level_layouts[level][0]
        destination_chunks = _regular_chunks_for_shape(
            level_shapes[level],
            storage_unit,
        )
        destination_offsets = _chunk_offsets(destination_chunks)
        level_tasks: list[Delayed] = []
        for index in np.ndindex(tuple(len(axis) for axis in destination_chunks)):
            destination_region = _chunk_region(
                destination_chunks,
                destination_offsets,
                index,
            )
            level_tasks.append(
                cast(
                    Delayed,
                    delayed(_write_downsampled_zarr_region)(
                        source_path,
                        destination_path,
                        source_shape,
                        destination_region,
                        downsample_method,
                        barrier,
                    ),
                )
            )
        barrier = _combine_delayed(
            level_tasks,
            dask_key_name=f"complete-ome-zarr-level-{level}-writes-{write_token}",
        )

    return barrier
