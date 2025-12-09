import numpy as np
import pytest

import anu_ctlab_io
from anu_ctlab_io import VoxelUnit

try:
    import anu_ctlab_io.zarr

    _HAS_ZARR = True
except ImportError:
    _HAS_ZARR = False


@pytest.mark.skipif(not _HAS_ZARR, reason="Requires 'zarr' extra")
def test_read_zarr():
    dataset = anu_ctlab_io.Dataset.from_path("tests/data/tomoHiRes.zarr")
    array = dataset.data

    assert array.dtype == np.uint16
    assert array.shape[0] == 100
    assert array.shape[1] == 200
    assert array.shape[2] == 300

    # See generate_test_data_zarr.py
    #   Each chunk of z shape 30 is just filled with a constant value of the chunk index
    #   The chunk z shape is independent of the netcdf block z shape.
    print(array.compute()[range(0, 100, 5), 0, 0])
    shape_z = array.shape[0]
    chunk_z = 30
    num_chunks_z = (shape_z + chunk_z - 1) // chunk_z
    for i in range(num_chunks_z):
        chunk = array[i * chunk_z : min((i + 1) * chunk_z, shape_z), ...]
        assert (chunk == i).all()


@pytest.mark.skipif(not _HAS_ZARR, reason="Requires 'zarr' extra")
def test_read_ome_zarr_mango():
    dataset = anu_ctlab_io.Dataset.from_path("tests/data/tomoLoRes_SS_AM.zarr")
    assert dataset.dimension_names == ("z", "y", "x")
    assert dataset.voxel_unit == "mm"
    assert str(dataset.voxel_unit) == "mm"
    assert np.isclose(
        dataset.voxel_size,
        (3.374303877353668e-2, 3.374303877353668e-2, 3.374303877353668e-2),
    ).all()
    voxel_size_um = dataset.voxel_size_with_unit(VoxelUnit.UM)
    assert np.isclose(
        voxel_size_um,
        (33.74303877353668, 33.74303877353668, 33.74303877353668),
    ).all()


@pytest.mark.skipif(not _HAS_ZARR, reason="Requires 'zarr' extra")
def test_read_ome_zarr_plain():
    dataset = anu_ctlab_io.Dataset.from_path("tests/data/generic.ome.zarr")
    assert dataset.dimension_names == ("z", "y", "x")
    assert dataset.voxel_unit == "mm"
    assert str(dataset.voxel_unit) == "mm"
    assert np.isclose(
        dataset.voxel_size,
        (3.374303877353668e-2, 3.374303877353668e-2, 3.374303877353668e-2),
    ).all()
    voxel_size_um = dataset.voxel_size_with_unit(VoxelUnit.UM)
    assert np.isclose(
        voxel_size_um,
        (33.74303877353668, 33.74303877353668, 33.74303877353668),
    ).all()
