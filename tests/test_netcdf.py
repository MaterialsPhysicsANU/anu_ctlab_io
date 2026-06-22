import h5netcdf.legacyapi as nc4
import numpy as np
import pytest

try:
    import anu_ctlab_io.netcdf

    _HAS_NETCDF = True
except ImportError:
    _HAS_NETCDF = False

import anu_ctlab_io

pytestmark = pytest.mark.skipif(not _HAS_NETCDF, reason="Requires 'netcdf' extra")


def test_read_netcdf_single():
    dataset = anu_ctlab_io.Dataset.from_path("tests/data/tomoLoRes_SS.nc")

    # check the type is right
    assert str(dataset._datatype) == "tomo"

    array = dataset.data
    assert array.dtype == np.uint16
    assert array.shape[0] == 10
    assert array.shape[1] == 20
    assert array.shape[2] == 30

    assert (array[:] == np.arange(np.prod(array.shape)).reshape(array.shape)).all()
    assert dataset.history and isinstance(dataset.history, dict)
    print(dataset.history)


def test_read_netcdf_multi():
    dataset = anu_ctlab_io.Dataset.from_path(
        "tests/data/tomoHiRes_SS_nc",
    )
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


def _write_chunked_netcdf_block(path, data):
    with nc4.Dataset(path, "w", format="NETCDF4") as ncfile:
        ncfile.createDimension("tomo_zdim", data.shape[0])
        ncfile.createDimension("tomo_ydim", data.shape[1])
        ncfile.createDimension("tomo_xdim", data.shape[2])
        ncfile.setncattr("voxel_size_xyz", np.array([1.0, 1.0, 1.0], dtype=np.float32))
        ncfile.setncattr("voxel_unit", np.bytes_(b"mm"))
        data_var = ncfile.createVariable(
            "tomo",
            "u2",
            ("tomo_zdim", "tomo_ydim", "tomo_xdim"),
            chunksizes=(2, 4, 5),
        )
        data_var[:] = data


def test_read_netcdf_respects_internal_chunks_by_default(tmp_path):
    path = tmp_path / "tomo_chunked.nc"
    data = np.arange(6 * 8 * 10, dtype=np.uint16).reshape(6, 8, 10)
    _write_chunked_netcdf_block(path, data)

    dataset = anu_ctlab_io.Dataset.from_path(path, parse_history=False)

    assert dataset.data.chunks == ((2, 2, 2), (4, 4), (5, 5))
    assert (dataset.data == data).all()


def test_read_netcdf_can_treat_each_block_as_one_chunk(tmp_path):
    path = tmp_path / "tomo_chunked_nc"
    path.mkdir()
    block0 = np.zeros((4, 8, 10), dtype=np.uint16)
    block1 = np.ones((4, 8, 10), dtype=np.uint16)
    _write_chunked_netcdf_block(path / "block00000000.nc", block0)
    _write_chunked_netcdf_block(path / "block00000001.nc", block1)

    dataset = anu_ctlab_io.Dataset.from_path(
        path, parse_history=False, ignore_block_chunks=True
    )

    assert dataset.data.chunks == ((4, 4), (8,), (10,))
    assert (dataset.data[:4] == 0).all()
    assert (dataset.data[4:] == 1).all()
