import tempfile
from pathlib import Path

import numpy as np

from anu_ctlab_io.raw import dataset_to_raw


def test_write_raw_basic(_make_dataset):
    """File exists, has the right size, and bytes decode back to the original array."""
    shape = (10, 20, 30)
    dataset, _ = _make_dataset(shape, datatype=None)

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


def test_write_raw_dtype_preserved(_make_dataset):
    """Float32 data is written as float32 (little-endian), not coerced."""
    shape = (5, 8, 8)
    dataset, _ = _make_dataset(shape, dtype=np.float32, datatype=None)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "output.raw"
        dataset_to_raw(dataset, path)

        expected_bytes = np.prod(shape) * np.dtype(np.float32).itemsize
        assert path.stat().st_size == expected_bytes

        recovered = np.frombuffer(
            path.read_bytes(), dtype=np.dtype(np.float32).newbyteorder("<")
        ).reshape(shape)
        assert np.array_equal(recovered, dataset.data.compute())


def test_write_raw_z_chunks(_make_dataset):
    """Multiple Z-chunks are streamed correctly and produce intact data."""
    shape = (30, 20, 10)
    # 3 chunks along Z
    dataset, _ = _make_dataset(shape, chunks=(10, 20, 10), datatype=None)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "output.raw"
        dataset_to_raw(dataset, path)

        recovered = np.frombuffer(
            path.read_bytes(), dtype=np.dtype(np.uint16).newbyteorder("<")
        ).reshape(shape)
        assert np.array_equal(recovered, dataset.data.compute())


def test_write_raw_arbitrary_chunks(_make_dataset):
    """Data with multiple chunks on all three axes is written correctly."""
    shape = (24, 20, 16)
    # 3 chunks on Z, 2 on Y, 4 on X
    dataset, _ = _make_dataset(shape, chunks=(8, 10, 4), datatype=None)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "output.raw"
        dataset_to_raw(dataset, path)

        recovered = np.frombuffer(
            path.read_bytes(), dtype=np.dtype(np.uint16).newbyteorder("<")
        ).reshape(shape)
        assert np.array_equal(recovered, dataset.data.compute())


def test_write_via_dataset_to_path(_make_dataset):
    """High-level Dataset.to_path with filetype='raw' writes a valid raw file."""
    shape = (8, 12, 16)
    dataset, _ = _make_dataset(shape, chunks=(4, 6, 4), datatype=None)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "output.raw"
        dataset.to_path(path, filetype="raw")

        assert path.exists()
        recovered = np.frombuffer(
            path.read_bytes(), dtype=np.dtype(np.uint16).newbyteorder("<")
        ).reshape(shape)
        assert np.array_equal(recovered, dataset.data.compute())
