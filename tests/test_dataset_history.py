"""Tests for Dataset history modification methods."""

import dask.array as da
import numpy as np

import anu_ctlab_io


def create_test_dataset():
    """Create a simple test dataset."""
    shape = (10, 20, 30)
    data = da.from_array(
        np.arange(np.prod(shape), dtype=np.uint16).reshape(shape), chunks=shape
    )
    return anu_ctlab_io.Dataset(
        data=data,
        dimension_names=("z", "y", "x"),
        voxel_unit=anu_ctlab_io.VoxelUnit.MM,
        voxel_size=(0.1, 0.1, 0.1),
        datatype=anu_ctlab_io.DataType.TOMO,
        history={"initial": {"created": "test"}},
    )


def test_add_to_history_dict():
    """Test adding a dict entry to history."""
    ds = create_test_dataset()

    # Add history entry
    ds.add_to_history("20260128_crop", {"operation": "crop", "z_range": [10, 50]})

    # Verify entry was added
    assert "20260128_crop" in ds.history
    assert ds.history["20260128_crop"]["operation"] == "crop"
    assert ds.history["20260128_crop"]["z_range"] == [10, 50]

    # Original history should still exist
    assert "initial" in ds.history


def test_add_to_history_string():
    """Test adding a string entry to history."""
    ds = create_test_dataset()

    # Add string history entry
    ds.add_to_history("20260128_note", "Cropped to region of interest")

    # Verify entry was added
    assert "20260128_note" in ds.history
    assert ds.history["20260128_note"] == "Cropped to region of interest"


def test_add_to_history_empty_dataset():
    """Test adding history to a dataset with no existing history."""
    shape = (5, 10, 15)
    data = da.zeros(shape, dtype=np.uint16, chunks=shape)

    # Create dataset with no history (defaults to {})
    ds = anu_ctlab_io.Dataset(
        data=data,
        dimension_names=("z", "y", "x"),
        voxel_unit=anu_ctlab_io.VoxelUnit.MM,
        voxel_size=(0.1, 0.1, 0.1),
    )

    # Add history to empty history
    ds.add_to_history("first_entry", {"operation": "test"})

    assert isinstance(ds.history, dict)
    assert "first_entry" in ds.history


def test_update_history():
    """Test bulk updating history."""
    ds = create_test_dataset()

    # Add multiple entries at once
    ds.update_history(
        {
            "20260128_crop": {"operation": "crop", "z_range": [10, 50]},
            "20260128_filter": {"operation": "gaussian", "sigma": 2.0},
            "20260128_note": "Processing pipeline v1.0",
        }
    )

    # Verify all entries were added
    assert "20260128_crop" in ds.history
    assert "20260128_filter" in ds.history
    assert "20260128_note" in ds.history

    assert ds.history["20260128_crop"]["z_range"] == [10, 50]
    assert ds.history["20260128_filter"]["sigma"] == 2.0

    # Original history should still exist
    assert "initial" in ds.history


def test_update_history_overwrites():
    """Test that update_history overwrites existing keys."""
    ds = create_test_dataset()

    # Overwrite the initial entry
    ds.update_history({"initial": {"updated": "yes"}})

    assert ds.history["initial"]["updated"] == "yes"
    assert "created" not in ds.history["initial"]


def test_from_modified_data_only():
    """Test from_modified with just data change."""
    ds = create_test_dataset()

    # Modify data
    new_data = ds.data[5:, :, :]
    modified = anu_ctlab_io.Dataset.from_modified(
        ds,
        data=new_data,
        history_entry={"operation": "crop", "z_start": 5},
        history_key="20260128_crop",
    )

    # Verify new dataset has modified data
    assert modified.data.shape == (5, 20, 30)
    assert modified.data is new_data

    # Verify metadata preserved
    assert modified.voxel_size == ds.voxel_size
    assert modified.voxel_unit == ds.voxel_unit
    assert modified.dimension_names == ds.dimension_names

    # Verify history updated
    assert "20260128_crop" in modified.history
    assert "initial" in modified.history  # Original preserved

    # Verify source unchanged
    assert ds.data.shape == (10, 20, 30)
    assert "20260128_crop" not in ds.history


def test_from_modified_voxel_size():
    """Test from_modified with voxel size change."""
    ds = create_test_dataset()

    new_voxel_size = (np.float32(0.2), np.float32(0.2), np.float32(0.2))
    modified = anu_ctlab_io.Dataset.from_modified(
        ds,
        voxel_size=new_voxel_size,
        history_entry={"operation": "downsample", "factor": 2},
    )

    assert modified.voxel_size == new_voxel_size
    assert modified.data.shape == ds.data.shape  # Data unchanged

    # Check history was added with auto-generated key
    history_keys = list(modified.history.keys())
    assert len(history_keys) == 2  # initial + new entry
    assert any("modification" in k for k in history_keys)


def test_from_modified_auto_timestamp():
    """Test that from_modified auto-generates timestamp-based keys."""
    ds = create_test_dataset()

    modified = anu_ctlab_io.Dataset.from_modified(
        ds,
        data=ds.data * 2,
        history_entry={"operation": "scale", "factor": 2},
        # No history_key provided - should auto-generate
    )

    # Should have auto-generated key with format YYYYMMDD_HHMMSS_modification
    new_keys = [k for k in modified.history.keys() if k != "initial"]
    assert len(new_keys) == 1

    key = new_keys[0]
    assert "_modification" in key
    assert len(key.split("_")[0]) == 8  # YYYYMMDD


def test_from_modified_no_history():
    """Test from_modified without adding history entry."""
    ds = create_test_dataset()

    # Modify without adding history
    modified = anu_ctlab_io.Dataset.from_modified(ds, data=ds.data * 2)

    # Should only have original history
    assert modified.history == ds.history
    assert len(modified.history) == 1  # Just "initial"


def test_from_modified_multiple_attributes():
    """Test from_modified changing multiple attributes."""
    ds = create_test_dataset()

    modified = anu_ctlab_io.Dataset.from_modified(
        ds,
        data=ds.data[:5, :, :],
        voxel_size=(0.05, 0.05, 0.05),
        voxel_unit=anu_ctlab_io.VoxelUnit.UM,
        history_entry={
            "operation": "crop_and_rescale",
            "crop_z": 5,
            "new_voxel_size_um": [50, 50, 50],
        },
        history_key="20260128_process",
    )

    assert modified.data.shape == (5, 20, 30)
    assert modified.voxel_size == (0.05, 0.05, 0.05)
    assert modified.voxel_unit == anu_ctlab_io.VoxelUnit.UM
    assert "20260128_process" in modified.history


def test_from_modified_chaining():
    """Test chaining multiple from_modified calls."""
    ds = create_test_dataset()

    # Chain of operations
    cropped = anu_ctlab_io.Dataset.from_modified(
        ds,
        data=ds.data[2:8, :, :],
        history_entry={"operation": "crop", "z_range": [2, 8]},
        history_key="step1_crop",
    )

    scaled = anu_ctlab_io.Dataset.from_modified(
        cropped,
        voxel_size=(0.2, 0.2, 0.2),
        history_entry={"operation": "downsample", "factor": 2},
        history_key="step2_downsample",
    )

    # Final dataset should have all history entries
    assert "initial" in scaled.history
    assert "step1_crop" in scaled.history
    assert "step2_downsample" in scaled.history

    # Verify history accumulates correctly
    assert scaled.history["step1_crop"]["z_range"] == [2, 8]
    assert scaled.history["step2_downsample"]["factor"] == 2

    # Verify shape reflects cropping
    assert scaled.data.shape == (6, 20, 30)


def test_from_modified_dimension_names():
    """Test from_modified with dimension name changes."""
    ds = create_test_dataset()

    # Change dimension names (e.g., after transpose)
    modified = anu_ctlab_io.Dataset.from_modified(
        ds,
        data=ds.data.transpose(2, 1, 0),
        dimension_names=("x", "y", "z"),
        history_entry={"operation": "transpose", "axes": [2, 1, 0]},
    )

    assert modified.dimension_names == ("x", "y", "z")
    assert modified.data.shape == (30, 20, 10)


def test_history_preserved_in_copy():
    """Test that from_modified creates independent history copy."""
    ds = create_test_dataset()

    modified = anu_ctlab_io.Dataset.from_modified(
        ds, data=ds.data * 2, history_entry={"operation": "scale"}
    )

    # Modify source history after creating modified version
    ds.add_to_history("post_modification", {"test": "value"})

    # Modified dataset should not see the new entry
    assert "post_modification" in ds.history
    assert "post_modification" not in modified.history
