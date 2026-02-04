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
def test_to_path_auto_nc_extension():
    """Test Dataset.to_path with automatic NetCDF detection via .nc extension."""
    shape = (10, 20, 30)
    data = da.from_array(
        np.arange(np.prod(shape), dtype=np.uint16).reshape(shape), chunks=shape
    )

    dataset = anu_ctlab_io.Dataset(
        data=data,
        dimension_names=("z", "y", "x"),
        voxel_unit=anu_ctlab_io.VoxelUnit.MM,
        voxel_size=(0.05, 0.05, 0.05),
        datatype=anu_ctlab_io.DataType.TOMO,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "tomo_output.nc"

        # Use the new to_path method with auto-detection
        dataset.to_path(output_path, dataset_id="test_auto_nc")

        # Read it back
        read_dataset = anu_ctlab_io.Dataset.from_path(output_path, parse_history=False)

        # Verify
        assert read_dataset.data.shape == shape
        assert np.array_equal(read_dataset.data.compute(), data.compute())


@pytest.mark.skipif(not _HAS_NETCDF, reason="Requires 'netcdf' extra")
def test_to_path_explicit_netcdf():
    """Test Dataset.to_path with explicit NetCDF filetype."""
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
        output_path = Path(tmpdir) / "tomo_output"

        # Use explicit filetype
        dataset.to_path(output_path, filetype="NetCDF", dataset_id="test_explicit")

        # Should have created .nc file
        expected_path = Path(str(output_path) + ".nc")
        assert expected_path.exists()

        # Read it back
        read_dataset = anu_ctlab_io.Dataset.from_path(
            expected_path, parse_history=False
        )
        assert read_dataset.data.shape == shape


@pytest.mark.skipif(not _HAS_NETCDF, reason="Requires 'netcdf' extra")
def test_to_path_with_split():
    """Test Dataset.to_path with file splitting."""
    shape = (100, 20, 30)
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
        output_path = Path(tmpdir) / "tomo_split"

        # Use to_path with splitting
        dataset.to_path(output_path, dataset_id="test_split", max_file_size_mb=0.05)

        # Should have created directory
        dir_path = Path(str(output_path) + "_nc")
        assert dir_path.exists()
        assert dir_path.is_dir()

        # Should have multiple blocks
        block_files = list(dir_path.glob("block*.nc"))
        assert len(block_files) > 1

        # Read it back
        read_dataset = anu_ctlab_io.Dataset.from_path(dir_path, parse_history=False)
        assert read_dataset.data.shape == shape
        assert np.array_equal(read_dataset.data.compute(), data.compute())


@pytest.mark.skipif(not _HAS_NETCDF, reason="Requires 'netcdf' extra")
def test_to_path_roundtrip():
    """Test complete roundtrip: read -> to_path -> from_path."""
    # Read original
    original = anu_ctlab_io.Dataset.from_path("tests/data/tomoLoRes_SS.nc")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "tomo_roundtrip.nc"

        # Write using to_path
        original.to_path(output_path, dataset_id="roundtrip_test")

        # Read back
        roundtrip = anu_ctlab_io.Dataset.from_path(output_path, parse_history=False)

        # Verify
        assert roundtrip.data.shape == original.data.shape
        assert np.array_equal(roundtrip.data.compute(), original.data.compute())
        assert roundtrip.voxel_unit == original.voxel_unit
        assert np.allclose(roundtrip.voxel_size, original.voxel_size)


@pytest.mark.skipif(not _HAS_NETCDF, reason="Requires 'netcdf' extra")
def test_to_path_error_on_unknown_extension():
    """Test that to_path raises error for unknown file extensions."""
    shape = (5, 10, 15)
    data = da.from_array(
        np.arange(np.prod(shape), dtype=np.uint16).reshape(shape), chunks=shape
    )

    dataset = anu_ctlab_io.Dataset(
        data=data,
        dimension_names=("z", "y", "x"),
        voxel_unit=anu_ctlab_io.VoxelUnit.MM,
        voxel_size=(1.0, 1.0, 1.0),
        datatype=anu_ctlab_io.DataType.TOMO,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "output.unknown"

        # Should raise ValueError for unknown extension
        with pytest.raises(ValueError, match="Unable to determine output format"):
            dataset.to_path(output_path)


@pytest.mark.skipif(not _HAS_NETCDF, reason="Requires 'netcdf' extra")
def test_to_path_split_with_nc_extension():
    """Test that split files replace .nc with _nc in directory name."""
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
        output_path = Path(tmpdir) / "tomo_output.nc"

        # Write with splitting
        dataset.to_path(output_path, dataset_id="test_nc_split", max_file_size_mb=0.05)

        # Should have created directory with .nc replaced by _nc
        expected_dir = Path(tmpdir) / "tomo_output_nc"
        assert expected_dir.exists(), f"Expected directory {expected_dir} not found"
        assert expected_dir.is_dir()

        # Should NOT have created tomo_output.nc_nc
        wrong_dir = Path(tmpdir) / "tomo_output.nc_nc"
        assert not wrong_dir.exists(), f"Incorrect directory {wrong_dir} was created"

        # Should have multiple blocks
        block_files = list(expected_dir.glob("block*.nc"))
        assert len(block_files) > 1

        # Read it back
        read_dataset = anu_ctlab_io.Dataset.from_path(expected_dir, parse_history=False)
        assert read_dataset.data.shape == shape
        assert np.array_equal(read_dataset.data.compute(), data.compute())
