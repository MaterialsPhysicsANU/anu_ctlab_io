import tempfile
from pathlib import Path

import dask.array as da
import numpy as np

import anu_ctlab_io
from anu_ctlab_io.raw import dataset_to_raw


def _make_dataset(shape, dtype=np.uint16, chunks=None):
    if chunks is None:
        chunks = shape
    data = da.from_array(
        np.arange(np.prod(shape), dtype=dtype).reshape(shape), chunks=chunks
    )
    return anu_ctlab_io.Dataset(
        data=data,
        dimension_names=("z", "y", "x"),
        voxel_unit=anu_ctlab_io.VoxelUnit.MM,
        voxel_size=(0.05, 0.05, 0.05),
    )


def test_write_raw_basic():
    """File exists, has the right size, and bytes decode back to the original array."""
    shape = (10, 20, 30)
    dataset = _make_dataset(shape)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "output.raw"
        dataset_to_raw(dataset, path)

        assert path.exists()
        expected_bytes = np.prod(shape) * np.dtype(np.uint16).itemsize
        assert path.stat().st_size == expected_bytes

        recovered = np.frombuffer(
            path.read_bytes(), dtype=np.dtype(np.uint16).newbyteorder("<")
        ).reshape(shape)
        assert np.array_equal(recovered, dataset.data.compute())


def test_write_raw_dtype_preserved():
    """Float32 data is written as float32 (little-endian), not coerced."""
    shape = (5, 8, 8)
    dataset = _make_dataset(shape, dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "output.raw"
        dataset_to_raw(dataset, path)

        expected_bytes = np.prod(shape) * np.dtype(np.float32).itemsize
        assert path.stat().st_size == expected_bytes

        recovered = np.frombuffer(
            path.read_bytes(), dtype=np.dtype(np.float32).newbyteorder("<")
        ).reshape(shape)
        assert np.array_equal(recovered, dataset.data.compute())


def test_write_raw_z_chunks():
    """Multiple Z-chunks are streamed correctly and produce intact data."""
    shape = (30, 20, 10)
    # 3 chunks along Z
    dataset = _make_dataset(shape, chunks=(10, 20, 10))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "output.raw"
        dataset_to_raw(dataset, path)

        recovered = np.frombuffer(
            path.read_bytes(), dtype=np.dtype(np.uint16).newbyteorder("<")
        ).reshape(shape)
        assert np.array_equal(recovered, dataset.data.compute())


def test_write_raw_arbitrary_chunks():
    """Data with multiple chunks on all three axes is written correctly."""
    shape = (24, 20, 16)
    # 3 chunks on Z, 2 on Y, 4 on X
    dataset = _make_dataset(shape, chunks=(8, 10, 4))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "output.raw"
        dataset_to_raw(dataset, path)

        recovered = np.frombuffer(
            path.read_bytes(), dtype=np.dtype(np.uint16).newbyteorder("<")
        ).reshape(shape)
        assert np.array_equal(recovered, dataset.data.compute())


def test_write_via_dataset_to_path():
    """High-level Dataset.to_path with filetype='raw' writes a valid raw file."""
    shape = (8, 12, 16)
    dataset = _make_dataset(shape, chunks=(4, 6, 4))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "output.raw"
        dataset.to_path(path, filetype="raw")

        assert path.exists()
        recovered = np.frombuffer(
            path.read_bytes(), dtype=np.dtype(np.uint16).newbyteorder("<")
        ).reshape(shape)
        assert np.array_equal(recovered, dataset.data.compute())
