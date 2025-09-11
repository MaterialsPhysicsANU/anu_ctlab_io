from anu_ctlab_io._datatype import DataType


def from_basename():
    assert DataType.from_basename("tomo_float") == DataType.TOMO_FLOAT
    assert DataType.from_basename("tomo") == DataType.TOMO


def infer_from_path():
    assert (
        DataType.infer_from_path("/path/to/tomo_floatData.zarr") == DataType.TOMO_FLOAT
    )
    assert DataType.infer_from_path("/path/to/tomoData.zarr") == DataType.TOMO
