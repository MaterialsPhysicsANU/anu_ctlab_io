import numpy as np

import anu_ctlab_io


# Subclass to test the dataset's masking without needing to generate real data
class Dataset(anu_ctlab_io.Dataset):
    def __init__(self, array, mask_value):
        self._data = array
        self._datatype = anu_ctlab_io.DataType.TOMO
        self._datatype._mask_value = (
            lambda: mask_value
        )  # override the internal function _mask_value used by the property mask_value (evil!)


def test_mask_creation():
    dataset = Dataset(np.array([[1, 0, 2], [2, 1, 3], [1, 0, 4]]), 1)
    print(dataset._datatype)
    print(dataset._data)
    print(dataset.mask)

    assert np.all(dataset.mask == np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]]))
    assert np.all(dataset.mask == (dataset._data == dataset.mask_value))

    dataset = Dataset(np.array([[1, 0, 2], [2, 1, 3], [1, 0, 4]]), None)
    assert np.all(dataset.mask == np.zeros_like(dataset.data))
