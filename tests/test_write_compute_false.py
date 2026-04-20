"""Tests for writing with compute=False, demonstrating concurrent writes via dask.compute."""

import tempfile
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import pytest

try:
    import anu_ctlab_io.netcdf

    _HAS_NETCDF = True
except ImportError:
    _HAS_NETCDF = False

try:
    import anu_ctlab_io.zarr

    _HAS_ZARR = True
except ImportError:
    _HAS_ZARR = False

import anu_ctlab_io
import anu_ctlab_io.raw


def _make_dataset(data: da.Array) -> anu_ctlab_io.Dataset:
    return anu_ctlab_io.Dataset(
        data=data,
        dimension_names=("z", "y", "x"),
        voxel_unit=anu_ctlab_io.VoxelUnit.MM,
        voxel_size=(0.05, 0.05, 0.05),
        datatype=anu_ctlab_io.DataType.TOMO,
    )


@pytest.mark.skipif(not _HAS_ZARR, reason="Requires 'zarr' extra")
def test_zarr_compute_false_concurrent():
    """Split an array into two halves, write both concurrently with compute=False."""
    shape = (20, 16, 16)
    arr = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    data = da.from_array(arr, chunks=(10, 16, 16))

    lo = _make_dataset(data[:10])
    hi = _make_dataset(data[10:])

    with tempfile.TemporaryDirectory() as tmpdir:
        path_lo = Path(tmpdir) / "lo.zarr"
        path_hi = Path(tmpdir) / "hi.zarr"

        task_lo = anu_ctlab_io.zarr.dataset_to_zarr(lo, path_lo, dataset_id="lo", compute=False)
        task_hi = anu_ctlab_io.zarr.dataset_to_zarr(hi, path_hi, dataset_id="hi", compute=False)

        assert task_lo is not None
        assert task_hi is not None

        dask.compute(task_lo, task_hi)  # type: ignore[attr-defined]

        read_lo = anu_ctlab_io.Dataset.from_path(path_lo)
        read_hi = anu_ctlab_io.Dataset.from_path(path_hi)

        assert np.array_equal(read_lo.data.compute(), arr[:10])
        assert np.array_equal(read_hi.data.compute(), arr[10:])


@pytest.mark.skipif(not _HAS_NETCDF, reason="Requires 'netcdf' extra")
def test_netcdf_compute_false_concurrent():
    """Split an array into two halves, write both concurrently with compute=False."""
    shape = (20, 16, 16)
    arr = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    data = da.from_array(arr, chunks=(10, 16, 16))

    lo = _make_dataset(data[:10])
    hi = _make_dataset(data[10:])

    with tempfile.TemporaryDirectory() as tmpdir:
        path_lo = Path(tmpdir) / "tomo_lo.nc"
        path_hi = Path(tmpdir) / "tomo_hi.nc"

        task_lo = anu_ctlab_io.netcdf.dataset_to_netcdf(lo, path_lo, dataset_id="lo", compute=False)
        task_hi = anu_ctlab_io.netcdf.dataset_to_netcdf(hi, path_hi, dataset_id="hi", compute=False)

        assert task_lo is not None
        assert task_hi is not None

        dask.compute(task_lo, task_hi)  # type: ignore[attr-defined]

        read_lo = anu_ctlab_io.Dataset.from_path(path_lo, parse_history=False)
        read_hi = anu_ctlab_io.Dataset.from_path(path_hi, parse_history=False)

        assert np.array_equal(read_lo.data.compute(), arr[:10])
        assert np.array_equal(read_hi.data.compute(), arr[10:])


@pytest.mark.skipif(not _HAS_NETCDF, reason="Requires 'netcdf' extra")
def test_netcdf_split_compute_false_concurrent():
    """Split an array into two halves, write both as split NetCDF concurrently."""
    shape = (40, 16, 16)
    arr = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    data = da.from_array(arr, chunks=(10, 16, 16))

    lo = _make_dataset(data[:20])
    hi = _make_dataset(data[20:])

    with tempfile.TemporaryDirectory() as tmpdir:
        # Each z-slice is 16*16*2 = 512 bytes; max_file_size_mb=0.005 => ~10 slices/file
        path_lo = Path(tmpdir) / "tomo_lo"
        path_hi = Path(tmpdir) / "tomo_hi"

        task_lo = anu_ctlab_io.netcdf.dataset_to_netcdf(
            lo, path_lo, dataset_id="lo", max_file_size_mb=0.005, compute=False
        )
        task_hi = anu_ctlab_io.netcdf.dataset_to_netcdf(
            hi, path_hi, dataset_id="hi", max_file_size_mb=0.005, compute=False
        )

        assert task_lo is not None
        assert task_hi is not None

        dask.compute(task_lo, task_hi)  # type: ignore[attr-defined]

        dir_lo = Path(str(path_lo) + "_nc")
        dir_hi = Path(str(path_hi) + "_nc")

        assert dir_lo.is_dir()
        assert dir_hi.is_dir()

        read_lo = anu_ctlab_io.Dataset.from_path(dir_lo, parse_history=False)
        read_hi = anu_ctlab_io.Dataset.from_path(dir_hi, parse_history=False)

        assert np.array_equal(read_lo.data.compute(), arr[:20])
        assert np.array_equal(read_hi.data.compute(), arr[20:])


def test_raw_compute_false_concurrent():
    """Split an array into two halves, write both as raw binary concurrently."""
    shape = (20, 16, 16)
    arr = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    data = da.from_array(arr, chunks=(10, 16, 16))

    lo = _make_dataset(data[:10])
    hi = _make_dataset(data[10:])

    with tempfile.TemporaryDirectory() as tmpdir:
        path_lo = Path(tmpdir) / "lo.raw"
        path_hi = Path(tmpdir) / "hi.raw"

        task_lo = anu_ctlab_io.raw.dataset_to_raw(lo, path_lo, compute=False)
        task_hi = anu_ctlab_io.raw.dataset_to_raw(hi, path_hi, compute=False)

        assert task_lo is not None
        assert task_hi is not None

        dask.compute(task_lo, task_hi)  # type: ignore[attr-defined]

        half_shape = (10, 16, 16)
        read_lo = np.frombuffer(path_lo.read_bytes(), dtype="<u2").reshape(half_shape)
        read_hi = np.frombuffer(path_hi.read_bytes(), dtype="<u2").reshape(half_shape)

        assert np.array_equal(read_lo, arr[:10])
        assert np.array_equal(read_hi, arr[10:])
