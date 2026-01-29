"""Tests for dataset_id tracking and save() functionality."""

from pathlib import Path

import numpy as np
import pytest

from anu_ctlab_io import Dataset
from anu_ctlab_io._dataset import _extract_base_name_from_dataset_id


class TestExtractBaseName:
    """Tests for the dataset_id timestamp stripping helper."""

    def test_old_format_with_timestamp(self):
        """Test stripping timestamp from old format dataset_id."""
        dataset_id = "20250314_012913_tomoLoRes_SS"
        result = _extract_base_name_from_dataset_id(dataset_id)
        assert result == "tomoLoRes_SS"

    def test_new_format_no_timestamp(self):
        """Test new format passes through unchanged."""
        dataset_id = "0-00000_gb1"
        result = _extract_base_name_from_dataset_id(dataset_id)
        assert result == "0-00000_gb1"

    def test_partial_timestamp_pattern(self):
        """Test that partial patterns don't match."""
        dataset_id = "20250314_tomoLoRes"  # Only date, no time
        result = _extract_base_name_from_dataset_id(dataset_id)
        assert result == "20250314_tomoLoRes"  # No match, return as-is

    def test_timestamp_like_but_not_at_start(self):
        """Test timestamp pattern in middle doesn't match."""
        dataset_id = "prefix_20250314_012913_suffix"
        result = _extract_base_name_from_dataset_id(dataset_id)
        assert result == "prefix_20250314_012913_suffix"  # No match at start

    def test_multiple_underscores_after_timestamp(self):
        """Test old format with multiple underscores in basename."""
        dataset_id = "20250314_012913_tomo_LoRes_SS_v2"
        result = _extract_base_name_from_dataset_id(dataset_id)
        assert result == "tomo_LoRes_SS_v2"


@pytest.mark.skipif(
    not pytest.importorskip("netCDF4", reason="Requires 'netcdf' extra"), reason=""
)
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

    def test_source_format_set_on_read(self, tmp_path):
        """Test that source_format is set when reading NetCDF."""
        test_file = Path(__file__).parent / "data" / "tomoLoRes_SS.nc"
        if not test_file.exists():
            pytest.skip("Test file not found")

        ds = Dataset.from_path(test_file)
        assert ds.source_format == "netcdf"

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
        assert modified.source_format == ds.source_format

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

    def test_dataset_id_suffix_without_source_id(self, tmp_path):
        """Test that dataset_id_suffix is ignored if source has no dataset_id."""
        import dask.array as da

        from anu_ctlab_io._voxel_properties import VoxelUnit

        # Create dataset without dataset_id
        ds = Dataset(
            data=da.from_array(np.ones((10, 20, 30), dtype=np.float32)),
            dimension_names=("z", "y", "x"),
            voxel_unit=VoxelUnit.MM,
            voxel_size=(1.0, 1.0, 1.0),
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


@pytest.mark.skipif(
    not pytest.importorskip("zarr", reason="Requires 'zarr' extra"), reason=""
)
class TestDatasetIdZarr:
    """Tests for dataset_id tracking with Zarr format."""

    def test_dataset_id_extracted_on_read_zarr(self, tmp_path):
        """Test that dataset_id is extracted when reading Zarr."""
        test_file = Path(__file__).parent / "data" / "tomoLoRes.zarr"
        if not test_file.exists():
            pytest.skip("Test file not found")

        ds = Dataset.from_path(test_file)
        assert ds.dataset_id is not None
        assert ds.source_format == "zarr"

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


@pytest.mark.skipif(
    not pytest.importorskip("netCDF4", reason="Requires 'netcdf' extra"), reason=""
)
class TestSaveMethod:
    """Tests for the save() method with auto-path generation."""

    def test_save_strips_timestamp_from_filename(self, tmp_path):
        """Test that save() strips timestamp from old format dataset_id."""
        test_file = Path(__file__).parent / "data" / "tomoLoRes_SS.nc"
        if not test_file.exists():
            pytest.skip("Test file not found")

        ds = Dataset.from_path(test_file)

        # Save with default suffix
        output_path = ds.save(directory=tmp_path)

        # Check filename doesn't contain timestamp
        assert output_path.name == "tomoLoRes_SS_CTLAB_IO.nc"
        assert "20250314_012913" not in output_path.name

    def test_save_preserves_timestamp_in_metadata(self, tmp_path):
        """Test that save() preserves timestamp in file metadata."""
        test_file = Path(__file__).parent / "data" / "tomoLoRes_SS.nc"
        if not test_file.exists():
            pytest.skip("Test file not found")

        ds = Dataset.from_path(test_file)
        original_id = ds.dataset_id

        # Save and read back
        output_path = ds.save(directory=tmp_path)
        ds2 = Dataset.from_path(output_path)

        # Metadata should preserve full dataset_id with timestamp
        assert ds2.dataset_id == original_id
        assert ds2.dataset_id == "20250314_012913_tomoLoRes_SS"

    def test_save_custom_suffix(self, tmp_path):
        """Test save() with custom suffix."""
        test_file = Path(__file__).parent / "data" / "tomoLoRes_SS.nc"
        if not test_file.exists():
            pytest.skip("Test file not found")

        ds = Dataset.from_path(test_file)

        output_path = ds.save(suffix="_processed", directory=tmp_path)
        assert output_path.name == "tomoLoRes_SS_processed.nc"

    @pytest.mark.skipif(
        not pytest.importorskip("zarr", reason="Requires 'zarr' extra"), reason=""
    )
    def test_save_format_zarr(self, tmp_path):
        """Test save() with zarr format."""
        test_file = Path(__file__).parent / "data" / "tomoLoRes_SS.nc"
        if not test_file.exists():
            pytest.skip("Test file not found")

        ds = Dataset.from_path(test_file)

        output_path = ds.save(format="zarr", directory=tmp_path)
        assert output_path.name == "tomoLoRes_SS_CTLAB_IO.zarr"
        assert output_path.exists()

    def test_save_uses_source_format_by_default(self, tmp_path):
        """Test that save() defaults to source_format."""
        test_file = Path(__file__).parent / "data" / "tomoLoRes_SS.nc"
        if not test_file.exists():
            pytest.skip("Test file not found")

        ds = Dataset.from_path(test_file)

        # Don't specify format - should use source format (netcdf)
        output_path = ds.save(directory=tmp_path)
        assert output_path.suffix == ".nc"

    def test_save_raises_without_dataset_id(self, tmp_path):
        """Test that save() raises error when dataset_id is missing."""
        import dask.array as da

        from anu_ctlab_io._voxel_properties import VoxelUnit

        # Create dataset without dataset_id
        ds = Dataset(
            data=da.from_array(np.ones((10, 20, 30), dtype=np.float32)),
            dimension_names=("z", "y", "x"),
            voxel_unit=VoxelUnit.MM,
            voxel_size=(1.0, 1.0, 1.0),
        )

        with pytest.raises(ValueError, match="Cannot auto-generate filename"):
            ds.save(directory=tmp_path)

    def test_save_returns_path(self, tmp_path):
        """Test that save() returns the Path to the written file."""
        test_file = Path(__file__).parent / "data" / "tomoLoRes_SS.nc"
        if not test_file.exists():
            pytest.skip("Test file not found")

        ds = Dataset.from_path(test_file)

        output_path = ds.save(directory=tmp_path)
        assert isinstance(output_path, Path)
        assert output_path.exists()

    def test_save_new_format_dataset_id(self, tmp_path):
        """Test save() with new format dataset_id (no timestamp)."""
        # Use a real test file to ensure we can roundtrip
        test_file = Path(__file__).parent / "data" / "tomoLoRes_SS.nc"
        if not test_file.exists():
            pytest.skip("Test file not found")

        # Load a real dataset and replace its dataset_id with new format
        ds = Dataset.from_path(test_file)

        # Create a new dataset with new format ID (using from_modified to preserve all properties)
        new_ds = Dataset(
            data=ds.data,
            dimension_names=ds.dimension_names,
            voxel_unit=ds.voxel_unit,
            voxel_size=ds.voxel_size,
            dataset_id="0-00000_gb1",  # New format
            source_format="netcdf",
            datatype=ds._datatype,
            history=ds._history,
        )

        output_path = new_ds.save(directory=tmp_path)

        # New format should be used as-is (no timestamp to strip)
        assert output_path.name == "0-00000_gb1_CTLAB_IO.nc"

        # Can't verify by reading back because filename doesn't have datatype
        # But we can verify the file was created
        assert output_path.exists()
