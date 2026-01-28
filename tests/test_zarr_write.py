import tempfile
from pathlib import Path

import dask.array as da
import numpy as np
import pytest

try:
    import anu_ctlab_io.zarr

    _HAS_ZARR = True
except ImportError:
    _HAS_ZARR = False

import anu_ctlab_io


@pytest.mark.skipif(not _HAS_ZARR, reason="Requires 'zarr' extra")
def test_write_single_ome_zarr():
    """Test writing a single OME-Zarr group."""
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
        output_path = Path(tmpdir) / "test_output.zarr"

        # Write the dataset as OME-Zarr
        anu_ctlab_io.zarr.dataset_to_zarr(
            dataset,
            output_path,
            dataset_id="test_dataset",
            use_ome_zarr=True,
        )

        # Verify file exists
        assert output_path.exists()
        assert output_path.is_dir()

        # Read it back
        read_dataset = anu_ctlab_io.Dataset.from_path(output_path)

        # Verify
        assert read_dataset.data.shape == shape
        assert read_dataset.voxel_unit == anu_ctlab_io.VoxelUnit.MM
        assert np.allclose(read_dataset.voxel_size, dataset.voxel_size)
        assert np.array_equal(read_dataset.data.compute(), data.compute())
        assert read_dataset.dimension_names == dataset.dimension_names


@pytest.mark.skipif(not _HAS_ZARR, reason="Requires 'zarr' extra")
def test_write_single_zarr_array():
    """Test writing a simple Zarr V3 array with mango metadata."""
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
        output_path = Path(tmpdir) / "test_output.zarr"

        # Write as simple Zarr array
        anu_ctlab_io.zarr.dataset_to_zarr(
            dataset,
            output_path,
            dataset_id="test_dataset",
            use_ome_zarr=False,
        )

        # Verify file exists
        assert output_path.exists()
        assert output_path.is_dir()

        # Read it back
        read_dataset = anu_ctlab_io.Dataset.from_path(output_path)

        # Verify
        assert read_dataset.data.shape == shape
        assert read_dataset.voxel_unit == anu_ctlab_io.VoxelUnit.MM
        assert np.allclose(read_dataset.voxel_size, dataset.voxel_size)
        assert np.array_equal(read_dataset.data.compute(), data.compute())
        assert read_dataset.dimension_names == dataset.dimension_names


@pytest.mark.skipif(not _HAS_ZARR, reason="Requires 'zarr' extra")
def test_write_without_datatype():
    """Test writing OME-Zarr without mango metadata (no datatype)."""
    shape = (5, 10, 15)
    data = da.from_array(
        np.arange(np.prod(shape), dtype=np.float32).reshape(shape), chunks=shape
    )

    dataset = anu_ctlab_io.Dataset(
        data=data,
        dimension_names=("z", "y", "x"),
        voxel_unit=anu_ctlab_io.VoxelUnit.UM,
        voxel_size=(1.0, 1.0, 1.0),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_no_mango.zarr"

        # Write without mango metadata (no datatype)
        anu_ctlab_io.zarr.dataset_to_zarr(dataset, output_path)

        # Read it back
        read_dataset = anu_ctlab_io.Dataset.from_path(output_path)

        # Verify basic properties
        assert read_dataset.data.shape == shape
        assert np.array_equal(read_dataset.data.compute(), data.compute())
        assert read_dataset.voxel_unit == anu_ctlab_io.VoxelUnit.UM
        assert np.allclose(read_dataset.voxel_size, (1.0, 1.0, 1.0))


@pytest.mark.skipif(not _HAS_ZARR, reason="Requires 'zarr' extra")
def test_write_split_zarr():
    """Test writing split Zarr stores."""
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
        output_path = Path(tmpdir) / "test_split"

        # Write with max store size to force splitting
        # Each z-slice is 20*30*2 bytes = 1200 bytes
        # Set max to ~0.02 MB to get ~18 slices per store
        anu_ctlab_io.zarr.dataset_to_zarr(
            dataset,
            output_path,
            dataset_id="test_split_dataset",
            max_store_size_mb=0.02,
        )

        # Check directory was created
        dir_path = Path(str(output_path) + "_zarr")
        assert dir_path.exists()
        assert dir_path.is_dir()

        # Check multiple store directories exist
        store_dirs = sorted(list(dir_path.glob("store*.zarr")))
        assert len(store_dirs) > 1

        # Verify each store can be read
        for store_path in store_dirs:
            store_dataset = anu_ctlab_io.Dataset.from_path(store_path)
            assert store_dataset.data.shape[1:] == (20, 30)
            assert store_dataset.voxel_unit == anu_ctlab_io.VoxelUnit.MM


@pytest.mark.skipif(not _HAS_ZARR, reason="Requires 'zarr' extra")
def test_roundtrip_against_reference():
    """Test roundtrip by comparing with reference Zarr files."""
    # Read existing test file
    original_dataset = anu_ctlab_io.Dataset.from_path("tests/data/tomoLoRes.zarr")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "roundtrip.zarr"

        # Write it as simple array (to match source format)
        anu_ctlab_io.zarr.dataset_to_zarr(
            original_dataset,
            output_path,
            datatype=anu_ctlab_io.DataType.TOMO,
            dataset_id="roundtrip_test",
            use_ome_zarr=False,
        )

        # Read it back
        read_dataset = anu_ctlab_io.Dataset.from_path(output_path)

        # Verify data matches
        assert read_dataset.data.shape == original_dataset.data.shape
        assert np.array_equal(
            read_dataset.data.compute(), original_dataset.data.compute()
        )
        assert np.allclose(read_dataset.voxel_size, original_dataset.voxel_size)
        assert read_dataset.voxel_unit == original_dataset.voxel_unit
        assert read_dataset.dimension_names == original_dataset.dimension_names


@pytest.mark.skipif(not _HAS_ZARR, reason="Requires 'zarr' extra")
def test_roundtrip_ome_zarr():
    """Test roundtrip with OME-Zarr format."""
    # Read existing OME-Zarr test file
    original_dataset = anu_ctlab_io.Dataset.from_path("tests/data/tomoLoRes_SS_AM.zarr")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "ome_roundtrip.zarr"

        # Write it as OME-Zarr
        anu_ctlab_io.zarr.dataset_to_zarr(
            original_dataset,
            output_path,
            datatype=anu_ctlab_io.DataType.TOMO,
            dataset_id="ome_roundtrip_test",
            use_ome_zarr=True,
        )

        # Read it back
        read_dataset = anu_ctlab_io.Dataset.from_path(output_path)

        # Verify data matches
        assert read_dataset.data.shape == original_dataset.data.shape
        assert np.array_equal(
            read_dataset.data.compute(), original_dataset.data.compute()
        )
        assert np.allclose(read_dataset.voxel_size, original_dataset.voxel_size)
        assert read_dataset.voxel_unit == original_dataset.voxel_unit


@pytest.mark.skipif(not _HAS_ZARR, reason="Requires 'zarr' extra")
def test_to_path_auto_detection():
    """Test that Dataset.to_path() correctly auto-detects .zarr extension."""
    shape = (5, 10, 15)
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
        output_path = Path(tmpdir) / "auto_detect.zarr"

        # Use to_path with auto-detection
        dataset.to_path(output_path)

        # Verify it was written as Zarr
        assert output_path.exists()
        assert output_path.is_dir()

        # Read back and verify
        read_dataset = anu_ctlab_io.Dataset.from_path(output_path)
        assert read_dataset.data.shape == shape
        assert np.array_equal(read_dataset.data.compute(), data.compute())


@pytest.mark.skipif(not _HAS_ZARR, reason="Requires 'zarr' extra")
def test_write_split_replaces_zarr_extension():
    """Test that split writing replaces .zarr with _zarr in directory name."""
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
        # Path ends with .zarr
        output_path = Path(tmpdir) / "split.zarr"

        # Write with splitting
        anu_ctlab_io.zarr.dataset_to_zarr(
            dataset,
            output_path,
            dataset_id="test_zarr_replacement",
            max_store_size_mb=0.05,
        )

        # Should have created directory with .zarr replaced by _zarr
        expected_dir = Path(tmpdir) / "split_zarr"
        assert expected_dir.exists(), f"Expected directory {expected_dir} not found"
        assert expected_dir.is_dir()

        # Should NOT have created split.zarr_zarr
        wrong_dir = Path(tmpdir) / "split.zarr_zarr"
        assert not wrong_dir.exists(), f"Incorrect directory {wrong_dir} was created"

        # Verify stores exist
        store_dirs = list(expected_dir.glob("store*.zarr"))
        assert len(store_dirs) > 1


@pytest.mark.skipif(not _HAS_ZARR, reason="Requires 'zarr' extra")
def test_write_with_history():
    """Test writing with custom history metadata."""
    shape = (5, 10, 15)
    data = da.from_array(
        np.arange(np.prod(shape), dtype=np.uint16).reshape(shape), chunks=shape
    )

    custom_history = {
        "step1": {"operation": "reconstruction", "timestamp": "2024-01-01"},
        "step2": {"operation": "filtering", "timestamp": "2024-01-02"},
    }

    dataset = anu_ctlab_io.Dataset(
        data=data,
        dimension_names=("z", "y", "x"),
        voxel_unit=anu_ctlab_io.VoxelUnit.MM,
        voxel_size=(0.05, 0.05, 0.05),
        datatype=anu_ctlab_io.DataType.TOMO,
        history=custom_history,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "history_test.zarr"

        anu_ctlab_io.zarr.dataset_to_zarr(dataset, output_path, use_ome_zarr=False)

        # Read back and verify history is preserved
        read_dataset = anu_ctlab_io.Dataset.from_path(output_path)
        assert isinstance(read_dataset.history, dict)
        assert "step1" in read_dataset.history
        assert "step2" in read_dataset.history


@pytest.mark.skipif(not _HAS_ZARR, reason="Requires 'zarr' extra")
def test_write_different_dtypes():
    """Test writing with different numpy dtypes."""
    dtypes = [np.uint8, np.uint16, np.int16, np.float32]

    for dtype in dtypes:
        shape = (5, 10, 15)
        data = da.from_array(
            np.arange(np.prod(shape), dtype=dtype).reshape(shape), chunks=shape
        )

        dataset = anu_ctlab_io.Dataset(
            data=data,
            dimension_names=("z", "y", "x"),
            voxel_unit=anu_ctlab_io.VoxelUnit.MM,
            voxel_size=(0.05, 0.05, 0.05),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / f"dtype_{dtype.__name__}.zarr"
            anu_ctlab_io.zarr.dataset_to_zarr(dataset, output_path)

            # Read back and verify dtype preserved
            read_dataset = anu_ctlab_io.Dataset.from_path(output_path)
            assert read_dataset.data.dtype == dtype
            assert np.array_equal(read_dataset.data.compute(), data.compute())
