import dask.array as da
import numpy as np
import pytest

import anu_ctlab_io


@pytest.fixture
def _make_dataset():
    def make_dataset(
        shape,
        *,
        chunks=None,
        dtype=np.uint16,
        voxel_unit=anu_ctlab_io.VoxelUnit.MM,
        voxel_size=(0.03374304, 0.03374304, 0.03374304),
        datatype=anu_ctlab_io.DataType.TOMO,
        history=None,
        data=None,
    ):
        if data is None:
            data = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
        dask_data = da.from_array(data, chunks=chunks or shape)
        return (
            anu_ctlab_io.Dataset(
                data=dask_data,
                dimension_names=("z", "y", "x"),
                voxel_unit=voxel_unit,
                voxel_size=voxel_size,
                datatype=datatype,
                history=history,
            ),
            dask_data,
        )

    return make_dataset
