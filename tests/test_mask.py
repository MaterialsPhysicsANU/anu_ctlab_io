import numpy as np
import pytest

import anu_ctlab_io


# Subclass to test the dataset's masking without needing to generate real data
class Dataset(anu_ctlab_io.Dataset):
    def __init__(self, array, mask_value):
        self._data = array
        self._datatype = anu_ctlab_io.DataType.TOMO
        self._datatype._mask_value = (
            lambda: mask_value
        )  # override the internal function _mask_value used by the property mask_value (evil!)


DATA = np.array([[1, 0, 2], [2, 1, 3], [1, 0, 4]])
MASK = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]])


def test_mask_creation():
    dataset = Dataset(DATA.copy(), 1)
    print(dataset._datatype)
    print(dataset._data)
    print(dataset.mask)

    assert np.all(dataset.mask == MASK.copy())
    assert np.all(dataset.mask == (dataset._data == dataset.mask_value))

    dataset = Dataset(DATA.copy(), None)
    assert np.all(dataset.mask == np.zeros_like(dataset.data))


def masked_array_eq(ma1, ma2):
    print(np.ma.getmask(ma1))
    return (
        np.ma.allequal(ma1, ma2)
        and np.all(np.ma.getmaskarray(ma1) == np.ma.getmaskarray(ma2))
        and not np.logical_xor(
            np.ma.getmask(ma1) is np.ma.nomask, np.ma.getmask(ma2) is np.ma.nomask
        )
    )


def test_masked_array():
    dataset = Dataset(DATA.copy(), 1)
    print(dataset.masked_data.compute())
    print(np.ma.masked_array(DATA.copy(), mask=MASK.copy()))

    # Check that testing with an invalid mask fails
    with pytest.raises(AssertionError):
        assert masked_array_eq(
            dataset.masked_data.compute(),
            np.ma.array(
                DATA.copy(),
                mask=False,
            ),
        )

    # Check our data is being masked correctly
    assert masked_array_eq(
        dataset.masked_data.compute(),
        np.ma.masked_array(
            DATA.copy(),
            mask=MASK.copy(),
        ),
    )

    dataset = Dataset(DATA.copy(), None)
    dataset._datatype = None

    print(dataset.masked_data.compute())
    print(np.ma.getmask(dataset.masked_data.compute()))

    # check we're creating nomask when possible
    assert np.ma.getmask(dataset.masked_data.compute()) is np.ma.nomask

    assert np.ma.allequal(
        dataset.masked_data,
        np.ma.masked_array(DATA.copy(), mask=False),
        fill_value=False,
    )
