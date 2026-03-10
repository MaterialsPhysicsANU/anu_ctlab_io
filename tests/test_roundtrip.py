import tempfile
from pathlib import Path

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


@pytest.mark.skipif(
    not (_HAS_NETCDF and _HAS_ZARR), reason="Requires both 'netcdf' and 'zarr' extras"
)
def test_roundtrip_netcdf_zarr_netcdf():
    """Roundtrip test: NetCDF → OME-Zarr → Simple Zarr → NetCDF.

    This test validates the complete chain of conversions to ensure
    data and metadata preservation across all formats.
    """
    # Load original NetCDF file from test data
    original_path = Path("tests/data/tomoLoRes_SS.nc")
    original_dataset = anu_ctlab_io.Dataset.from_path(original_path)
    original_data = original_dataset.data.compute()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Step 1: Write to OME-Zarr
        zarr_ome_path = tmpdir_path / "roundtrip_ome.zarr"
        anu_ctlab_io.zarr.dataset_to_zarr(
            original_dataset,
            zarr_ome_path,
            datatype=anu_ctlab_io.DataType.TOMO,
            dataset_id="roundtrip_test_ome",
            ome_zarr_version=anu_ctlab_io.zarr.OMEZarrVersion.v05,
        )

        # Step 2: Read back from OME-Zarr and verify
        zarr_ome_dataset = anu_ctlab_io.Dataset.from_path(zarr_ome_path)
        zarr_ome_data = zarr_ome_dataset.data.compute()

        assert np.array_equal(zarr_ome_data, original_data)
        assert zarr_ome_dataset.data.shape == original_dataset.data.shape
        assert zarr_ome_dataset.data.dtype == original_dataset.data.dtype

        # Step 3: Write to simple Zarr array
        zarr_array_path = tmpdir_path / "roundtrip_array.zarr"
        anu_ctlab_io.zarr.dataset_to_zarr(
            original_dataset,
            zarr_array_path,
            datatype=anu_ctlab_io.DataType.TOMO,
            dataset_id="roundtrip_test_array",
            ome_zarr_version=None,
        )

        # Step 4: Read back from simple Zarr and verify
        zarr_array_dataset = anu_ctlab_io.Dataset.from_path(zarr_array_path)
        zarr_array_data = zarr_array_dataset.data.compute()

        assert np.array_equal(zarr_array_data, original_data)
        assert zarr_array_dataset.data.shape == original_dataset.data.shape

        # Step 5: Write Zarr back to NetCDF
        netcdf_path = tmpdir_path / "tomo_roundtrip.nc"
        anu_ctlab_io.netcdf.dataset_to_netcdf(
            zarr_ome_dataset,
            netcdf_path,
            datatype=anu_ctlab_io.DataType.TOMO,
            dataset_id="roundtrip_test_final",
        )

        # Step 6: Read back final NetCDF and verify correctness
        final_dataset = anu_ctlab_io.Dataset.from_path(netcdf_path)
        final_data = final_dataset.data.compute()

        # Verify data matches exactly
        assert np.array_equal(final_data, original_data)

        # Verify all metadata preserved
        assert final_dataset.data.shape == original_dataset.data.shape
        assert final_dataset.data.dtype == original_dataset.data.dtype
        assert final_dataset.voxel_unit == original_dataset.voxel_unit
        assert np.allclose(final_dataset.voxel_size, original_dataset.voxel_size)
        assert final_dataset.dimension_names == original_dataset.dimension_names
