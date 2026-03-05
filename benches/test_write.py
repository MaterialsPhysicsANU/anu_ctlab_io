import dask.array as da
import numpy as np
import pytest
from dask.distributed import Client, LocalCluster
from zarr.codecs import ZstdCodec

import anu_ctlab_io
import anu_ctlab_io.netcdf
import anu_ctlab_io.zarr

SHAPE = (1024, 1024, 1024)
MAX_FILE_SIZE_MB = 256


@pytest.fixture(scope="module")
def dataset():
    data = da.zeros(SHAPE, dtype=np.float32, chunks=SHAPE)
    return anu_ctlab_io.Dataset(
        data=data,
        dimension_names=("z", "y", "x"),
        voxel_unit=anu_ctlab_io.VoxelUnit.MM,
        voxel_size=(1.0, 1.0, 1.0),
        datatype=anu_ctlab_io.DataType.TOMO,
    )


def _assert_round_trip(dataset, read_back):
    expected = dataset.data.astype(dataset._datatype.dtype).compute()
    actual = read_back.data.astype(read_back._datatype.dtype).compute()
    assert np.array_equal(actual, expected)


def test_write_netcdf(benchmark, dataset, tmp_path):
    def write():
        anu_ctlab_io.netcdf.dataset_to_netcdf(
            dataset,
            tmp_path / "tomoOut_nc",
            compression_level=5,
            max_file_size_mb=MAX_FILE_SIZE_MB,
        )

    benchmark.pedantic(write, rounds=1, iterations=1)
    _assert_round_trip(
        dataset, anu_ctlab_io.netcdf.dataset_from_netcdf(tmp_path / "tomoOut_nc")
    )


def test_write_netcdf_distributed(benchmark, dataset, tmp_path):
    with LocalCluster() as cluster, Client(cluster):

        def write():
            anu_ctlab_io.netcdf.dataset_to_netcdf(
                dataset,
                tmp_path / "tomoOut_nc",
                compression_level=5,
                max_file_size_mb=MAX_FILE_SIZE_MB,
            )

        benchmark.pedantic(write, rounds=1, iterations=1)
    _assert_round_trip(
        dataset, anu_ctlab_io.netcdf.dataset_from_netcdf(tmp_path / "tomoOut_nc")
    )


def test_write_zarr(benchmark, dataset, tmp_path):
    """This uses the same chunking as netcdf

    Compressor is different (zstd vs zlib), but performance should be comparable
    """
    shape = SHAPE
    bytes_per_slice = shape[1] * shape[2] * np.dtype(np.float32).itemsize
    slices_per_block = max(1, int((MAX_FILE_SIZE_MB * 1024 * 1024) / bytes_per_slice))
    chunks = (slices_per_block, shape[1], shape[2])

    def write():
        anu_ctlab_io.zarr.dataset_to_zarr(
            dataset,
            tmp_path / "tomoOut.zarr",
            chunks=chunks,
            shards=chunks,
            create_array_kwargs={
                "compressors": [ZstdCodec(level=5)],
            },
        )

    benchmark.pedantic(write, rounds=1, iterations=1)
    _assert_round_trip(
        dataset, anu_ctlab_io.zarr.dataset_from_zarr(tmp_path / "tomoOut.zarr")
    )
