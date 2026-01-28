import tempfile
from pathlib import Path

import dask.array as da
import numpy as np
import pytest

try:
    import anu_ctlab_io.netcdf

    _HAS_NETCDF = True
except ImportError:
    _HAS_NETCDF = False

import anu_ctlab_io


@pytest.mark.skipif(not _HAS_NETCDF, reason="Requires 'netcdf' extra")
def test_write_single_netcdf():
    """Test writing a single NetCDF file."""
    # Create test dataset
    shape = (10, 20, 30)
    data = da.from_array(
        np.arange(np.prod(shape), dtype=np.uint16).reshape(shape), chunks=shape
    )

    dataset = anu_ctlab_io.Dataset(
        data=data,
        dimension_names=("z", "y", "x"),
        voxel_unit=anu_ctlab_io.VoxelUnit.MM,
        voxel_size=(0.03374304, 0.03374304, 0.03374304),
        datatype=anu_ctlab_io.DataType.TOMO,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Use proper filename with datatype prefix for reading
        output_path = Path(tmpdir) / "tomo_test_output.nc"

        # Write the dataset
        anu_ctlab_io.netcdf.dataset_to_netcdf(
            dataset,
            output_path,
            dataset_id="test_dataset",
        )

        # Read it back (without parsing history since we didn't write Mango-format history)
        read_dataset = anu_ctlab_io.Dataset.from_path(output_path, parse_history=False)

        # Verify
        assert read_dataset.data.shape == shape
        assert read_dataset.voxel_unit == anu_ctlab_io.VoxelUnit.MM
        assert np.allclose(read_dataset.voxel_size, dataset.voxel_size)
        assert np.array_equal(read_dataset.data.compute(), data.compute())


@pytest.mark.skipif(not _HAS_NETCDF, reason="Requires 'netcdf' extra")
def test_write_split_netcdf():
    """Test writing split NetCDF files."""
    # Create test dataset (100 z-slices)
    shape = (100, 20, 30)
    data = da.from_array(
        np.arange(np.prod(shape), dtype=np.uint16).reshape(shape), chunks=(10, 20, 30)
    )

    dataset = anu_ctlab_io.Dataset(
        data=data,
        dimension_names=("z", "y", "x"),
        voxel_unit=anu_ctlab_io.VoxelUnit.MM,
        voxel_size=(0.03374304, 0.03374304, 0.03374304),
        datatype=anu_ctlab_io.DataType.TOMO,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Use proper filename with datatype prefix
        output_path = Path(tmpdir) / "tomo_test_split"

        # Write with max file size to force splitting
        # Each z-slice is 20*30*2 bytes = 1200 bytes
        # Set max to ~0.02 MB to get ~18 slices per file
        anu_ctlab_io.netcdf.dataset_to_netcdf(
            dataset,
            output_path,
            dataset_id="test_split_dataset",
            max_file_size_mb=0.02,
        )

        # Check directory was created
        dir_path = Path(str(output_path) + "_nc")
        assert dir_path.exists()
        assert dir_path.is_dir()

        # Check multiple block files exist
        block_files = sorted(list(dir_path.glob("block*.nc")))
        assert len(block_files) > 1

        # Read it back (without parsing history)
        read_dataset = anu_ctlab_io.Dataset.from_path(dir_path, parse_history=False)

        # Verify
        assert read_dataset.data.shape == shape
        assert read_dataset.voxel_unit == anu_ctlab_io.VoxelUnit.MM
        assert np.allclose(read_dataset.voxel_size, dataset.voxel_size)
        assert np.array_equal(read_dataset.data.compute(), data.compute())


@pytest.mark.skipif(not _HAS_NETCDF, reason="Requires 'netcdf' extra")
def test_write_without_histogram():
    """Test writing without computing histogram."""
    shape = (5, 10, 15)
    data = da.from_array(
        np.arange(np.prod(shape), dtype=np.uint16).reshape(shape), chunks=shape
    )

    dataset = anu_ctlab_io.Dataset(
        data=data,
        dimension_names=("z", "y", "x"),
        voxel_unit=anu_ctlab_io.VoxelUnit.UM,
        voxel_size=(1.0, 1.0, 1.0),
        datatype=anu_ctlab_io.DataType.TOMO,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "tomo_test_no_hist.nc"

        # Write without histogram
        anu_ctlab_io.netcdf.dataset_to_netcdf(
            dataset, output_path, compute_histogram=False
        )

        # Read it back (without parsing history)
        read_dataset = anu_ctlab_io.Dataset.from_path(output_path, parse_history=False)

        # Verify basic properties
        assert read_dataset.data.shape == shape
        assert np.array_equal(read_dataset.data.compute(), data.compute())


@pytest.mark.skipif(not _HAS_NETCDF, reason="Requires 'netcdf' extra")
def test_roundtrip_against_reference():
    """Test roundtrip by comparing with reference files."""
    # Read existing test file
    original_dataset = anu_ctlab_io.Dataset.from_path("tests/data/tomoLoRes_SS.nc")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "tomo_roundtrip.nc"

        # Write it
        anu_ctlab_io.netcdf.dataset_to_netcdf(
            original_dataset,
            output_path,
            datatype=anu_ctlab_io.DataType.TOMO,
            dataset_id="roundtrip_test",
        )

        # Read it back (without parsing history)
        read_dataset = anu_ctlab_io.Dataset.from_path(output_path, parse_history=False)

        # Verify data matches
        assert read_dataset.data.shape == original_dataset.data.shape
        assert np.array_equal(
            read_dataset.data.compute(), original_dataset.data.compute()
        )
        assert np.allclose(read_dataset.voxel_size, original_dataset.voxel_size)
        assert read_dataset.voxel_unit == original_dataset.voxel_unit


@pytest.mark.skipif(not _HAS_NETCDF, reason="Requires 'netcdf' extra")
def test_write_split_replaces_nc_extension():
    """Test that split writing replaces .nc with _nc in directory name."""
    shape = (50, 20, 30)
    data = da.from_array(
        np.arange(np.prod(shape), dtype=np.uint16).reshape(shape), chunks=(10, 20, 30)
    )

    dataset = anu_ctlab_io.Dataset(
        data=data,
        dimension_names=("z", "y", "x"),
        voxel_unit=anu_ctlab_io.VoxelUnit.MM,
        voxel_size=(0.05, 0.05, 0.05),
        datatype=anu_ctlab_io.DataType.TOMO,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Path ends with .nc
        output_path = Path(tmpdir) / "tomo_split.nc"

        # Write with splitting via dataset_to_netcdf
        anu_ctlab_io.netcdf.dataset_to_netcdf(
            dataset,
            output_path,
            dataset_id="test_nc_replacement",
            max_file_size_mb=0.05,
        )

        # Should have created directory with .nc replaced by _nc
        expected_dir = Path(tmpdir) / "tomo_split_nc"
        assert expected_dir.exists(), f"Expected directory {expected_dir} not found"
        assert expected_dir.is_dir()

        # Should NOT have created tomo_split.nc_nc
        wrong_dir = Path(tmpdir) / "tomo_split.nc_nc"
        assert not wrong_dir.exists(), f"Incorrect directory {wrong_dir} was created"

        # Verify blocks exist
        block_files = list(expected_dir.glob("block*.nc"))
        assert len(block_files) > 1

        # Read back and verify
        read_dataset = anu_ctlab_io.Dataset.from_path(expected_dir, parse_history=False)
        assert read_dataset.data.shape == shape
        assert np.array_equal(read_dataset.data.compute(), data.compute())
