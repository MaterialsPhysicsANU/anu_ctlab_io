"""Tests for dataset_id tracking."""

from pathlib import Path

import numpy as np
import pytest

from anu_ctlab_io import Dataset

try:
    import anu_ctlab_io.netcdf

    _HAS_NETCDF = True
except ImportError:
    _HAS_NETCDF = False

try:
    import anu_ctlab_io.zarr  # noqa: F401

    _HAS_ZARR = True
except ImportError:
    _HAS_ZARR = False


@pytest.mark.skipif(not _HAS_NETCDF, reason="Requires 'netcdf' extra")
class TestDatasetIdNetCDF:
    """Tests for dataset_id tracking with NetCDF format."""

    def test_dataset_id_extracted_on_read(self, tmp_path):
        """Test that dataset_id is extracted when reading NetCDF."""
        # Use existing test file
        test_file = Path(__file__).parent / "data" / "tomoLoRes_SS.nc"
        if not test_file.exists():
            pytest.skip("Test file not found")

        ds = Dataset.from_path(test_file)

        # Check that dataset_id was extracted
        assert ds.dataset_id is not None
        assert isinstance(ds.dataset_id, str)
        # Known dataset_id from test file
        assert ds.dataset_id == "20250314_012913_tomoLoRes_SS"

    def test_dataset_id_preserved_in_write(self, tmp_path):
        """Test that dataset_id is preserved when writing."""
        test_file = Path(__file__).parent / "data" / "tomoLoRes_SS.nc"
        if not test_file.exists():
            pytest.skip("Test file not found")

        ds = Dataset.from_path(test_file)
        original_id = ds.dataset_id

        # Write and read back with proper filename (includes datatype)
        output_file = tmp_path / "tomoLoRes_output.nc"
        ds.to_path(output_file)

        ds2 = Dataset.from_path(output_file)
        assert ds2.dataset_id == original_id

    def test_dataset_id_preserved_in_from_modified(self, tmp_path):
        """Test that dataset_id is preserved in from_modified() without suffix."""
        test_file = Path(__file__).parent / "data" / "tomoLoRes_SS.nc"
        if not test_file.exists():
            pytest.skip("Test file not found")

        ds = Dataset.from_path(test_file)
        original_id = ds.dataset_id

        # Create modified dataset without suffix
        modified = Dataset.from_modified(
            ds, data=ds.data[:5, :, :], history_entry={"operation": "crop"}
        )

        assert modified.dataset_id == original_id

    def test_dataset_id_modified_with_suffix(self, tmp_path):
        """Test that dataset_id is modified when dataset_id_suffix is provided."""
        test_file = Path(__file__).parent / "data" / "tomoLoRes_SS.nc"
        if not test_file.exists():
            pytest.skip("Test file not found")

        ds = Dataset.from_path(test_file)
        original_id = ds.dataset_id

        # Create modified dataset with suffix
        modified = Dataset.from_modified(
            ds,
            data=ds.data[:5, :, :],
            history_entry={"operation": "crop"},
            dataset_id_suffix="cropped",
        )

        assert modified.dataset_id == f"{original_id}_cropped"
        assert modified.dataset_id == "20250314_012913_tomoLoRes_SS_cropped"

    def test_dataset_id_suffix_chaining(self, tmp_path):
        """Test that dataset_id suffixes can be chained."""
        test_file = Path(__file__).parent / "data" / "tomoLoRes_SS.nc"
        if not test_file.exists():
            pytest.skip("Test file not found")

        ds = Dataset.from_path(test_file)

        # Chain modifications with suffixes
        cropped = Dataset.from_modified(
            ds, data=ds.data[:5, :, :], dataset_id_suffix="cropped"
        )

        filtered = Dataset.from_modified(
            cropped, data=cropped.data, dataset_id_suffix="filtered"
        )

        assert filtered.dataset_id == "20250314_012913_tomoLoRes_SS_cropped_filtered"

    def test_dataset_id_suffix_without_source_id(self, tmp_path, _make_dataset):
        """Test that dataset_id_suffix is ignored if source has no dataset_id."""
        # Create dataset without dataset_id
        ds, _ = _make_dataset(
            (10, 20, 30),
            datatype=None,
            data=np.ones((10, 20, 30), dtype=np.float32),
        )

        # Try to add suffix - should be ignored
        modified = Dataset.from_modified(
            ds, data=ds.data[:5, :, :], dataset_id_suffix="cropped"
        )

        assert modified.dataset_id is None

    def test_explicit_dataset_id_override(self, tmp_path):
        """Test that explicit dataset_id parameter overrides stored value."""
        test_file = Path(__file__).parent / "data" / "tomoLoRes_SS.nc"
        if not test_file.exists():
            pytest.skip("Test file not found")

        ds = Dataset.from_path(test_file)

        # Write with explicit dataset_id
        output_file = tmp_path / "tomoLoRes_custom.nc"
        custom_id = "custom_dataset_id_12345"
        ds.to_path(output_file, dataset_id=custom_id)

        # Read back and verify
        ds2 = Dataset.from_path(output_file)
        assert ds2.dataset_id == custom_id

    def test_dataset_id_none_generates_new(self, tmp_path):
        """Test that dataset_id=None generates a new ID (legacy behavior)."""
        test_file = Path(__file__).parent / "data" / "tomoLoRes_SS.nc"
        if not test_file.exists():
            pytest.skip("Test file not found")

        ds = Dataset.from_path(test_file)
        original_id = ds.dataset_id

        # Write with dataset_id=None (should generate new)
        output_file = tmp_path / "tomoLoRes_new.nc"
        ds.to_path(output_file, dataset_id=None)

        # Read back - should have a different ID
        ds2 = Dataset.from_path(output_file)
        assert ds2.dataset_id != original_id
        assert ds2.dataset_id is not None


@pytest.mark.skipif(not _HAS_ZARR, reason="Requires 'zarr' extra")
class TestDatasetIdZarr:
    """Tests for dataset_id tracking with Zarr format."""

    def test_dataset_id_extracted_on_read_zarr(self, tmp_path):
        """Test that dataset_id is extracted when reading Zarr."""
        test_file = Path(__file__).parent / "data" / "tomoLoRes.zarr"
        if not test_file.exists():
            pytest.skip("Test file not found")

        ds = Dataset.from_path(test_file)
        assert ds.dataset_id is not None

    def test_dataset_id_roundtrip_zarr(self, tmp_path):
        """Test dataset_id roundtrip with Zarr format."""
        test_file = Path(__file__).parent / "data" / "tomoLoRes.zarr"
        if not test_file.exists():
            pytest.skip("Test file not found")

        ds = Dataset.from_path(test_file)
        original_id = ds.dataset_id

        # Write and read back
        output_file = tmp_path / "tomoLoRes_output.zarr"
        ds.to_path(output_file)

        ds2 = Dataset.from_path(output_file)
        assert ds2.dataset_id == original_id
