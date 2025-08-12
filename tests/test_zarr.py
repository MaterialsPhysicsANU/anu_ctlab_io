import numpy as np
import anu_ctlab_io as io


def test_read_zarr():
    dataset = io.Dataset.from_path("tests/data/tomoHiRes.zarr")
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


def test_read_ome_zarr():
    dataset = io.Dataset.from_path("tests/data/tomoLoRes_SS_AM.zarr")
    assert dataset.dimension_names == ("z", "y", "x")
    assert dataset.voxel_unit == "mm"
    assert str(dataset.voxel_unit) == "mm"
