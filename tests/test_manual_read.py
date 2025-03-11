import xarray as xr
import numpy as np


def test_manual_read_netcdf_single():
    dataset = xr.open_mfdataset("tests/data/tomoLoRes_SS.nc")

    # Load array variable and check shape
    array = dataset["tomo"]
    assert dataset.attrs["zdim_total"] == array.shape[0]
    assert (dataset.attrs["total_grid_size_xyz"][::-1] == array.shape).all()

    # Manually strip blocking attributes, these need to be recreated on write
    del dataset.attrs["number_of_files"]
    del dataset.attrs["zdim_total"]
    del dataset.attrs["total_grid_size_xyz"]

    # Manual dtype correction
    assert array.dtype == np.int16
    array = array.astype(np.uint16)

    assert array.dtype == np.uint16
    assert array.shape[0] == 10
    assert array.shape[1] == 20
    assert array.shape[2] == 30

    assert (array[:] == np.arange(np.prod(array.shape)).reshape(array.shape)).all()


def test_manual_read_netcdf_multi():
    dataset = xr.open_mfdataset(
        "tests/data/tomoHiRes_SS_nc/*",
        data_vars=["tomo"],
        combine="nested",
        concat_dim="tomo_zdim",
        combine_attrs="drop_conflicts",
        coords="minimal",
        compat="override",
    )
    print(dataset.attrs.keys())

    # Load array variable and check shape
    array = dataset["tomo"]
    assert dataset.attrs["zdim_total"] == array.shape[0]
    assert (dataset.attrs["total_grid_size_xyz"][::-1] == array.shape).all()

    # Manually strip blocking attributes, these need to be recreated on write
    del dataset.attrs["number_of_files"]
    del dataset.attrs["zdim_total"]
    del dataset.attrs["total_grid_size_xyz"]

    # Manual dtype correction
    assert array.dtype == np.int16
    array = array.astype(np.uint16)

    assert array.dtype == np.uint16
    assert array.shape[0] == 100
    assert array.shape[1] == 200
    assert array.shape[2] == 300

    # See generate_test_data_zarr.py
    #   Each chunk of z shape 30 is just filled with a constant value of the chunk index
    #   The chunk z shape is independent of the netcdf block z shape.
    print(array.to_numpy()[range(0,100,5), 0, 0])
    shape_z = array.shape[0]
    chunk_z = 30
    num_chunks_z = (shape_z + chunk_z - 1) // chunk_z
    for i in range(num_chunks_z):
        chunk = array[i * chunk_z : min((i + 1) * chunk_z, shape_z), ...]
        assert (chunk == i).all()
