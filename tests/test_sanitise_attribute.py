"""Tests for _sanitise_attribute_value function."""

import numpy as np
import pytest

try:
    from anu_ctlab_io.netcdf._writer import _sanitise_attribute_value

    _HAS_NETCDF = True
except ImportError:
    _HAS_NETCDF = False


@pytest.mark.skipif(not _HAS_NETCDF, reason="Requires 'netcdf' extra")
class TestSanitiseAttributeValue:
    """Test attribute value sanitization for HDF5 compatibility."""

    def test_numeric_scalars_passthrough(self):
        """Numeric scalars should pass through unchanged."""
        assert _sanitise_attribute_value(42) == 42
        assert _sanitise_attribute_value(3.14) == 3.14
        assert _sanitise_attribute_value(np.int32(10)) == np.int32(10)
        assert _sanitise_attribute_value(np.float64(2.71)) == np.float64(2.71)

    def test_bytes_passthrough(self):
        """Bytes should pass through unchanged."""
        byte_val = b"test"
        result = _sanitise_attribute_value(byte_val)
        assert result is byte_val

    def test_np_bytes_passthrough(self):
        """np.bytes_ should pass through unchanged."""
        np_byte_val = np.bytes_(b"test")
        result = _sanitise_attribute_value(np_byte_val)
        assert result is np_byte_val

    def test_numeric_arrays_passthrough(self):
        """Numeric arrays should pass through unchanged."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = _sanitise_attribute_value(arr)
        assert result is arr

    def test_object_dtype_array_to_string(self):
        """Object dtype arrays should be converted to ASCII-encoded bytes."""
        arr = np.array(["test"], dtype=object)
        result = _sanitise_attribute_value(arr)
        assert isinstance(result, np.bytes_)
        # Should contain the string representation
        assert b"test" in result

    def test_list_to_numeric_array(self):
        """Lists with numeric elements should be converted to numeric arrays."""
        lst = [1, 2, 3]
        result = _sanitise_attribute_value(lst)
        assert isinstance(result, np.ndarray)
        assert result.dtype != np.object_
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    def test_tuple_to_numeric_array(self):
        """Tuples with numeric elements should be converted to numeric arrays."""
        tup = (1.0, 2.0, 3.0)
        result = _sanitise_attribute_value(tup)
        assert isinstance(result, np.ndarray)
        assert result.dtype != np.object_
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_mixed_list_to_string(self):
        """Lists with mixed types (resulting in object dtype) should be strings."""
        lst = [1, "two", 3.0]
        result = _sanitise_attribute_value(lst)
        assert isinstance(result, np.bytes_)
        assert b"1" in result and b"two" in result

    def test_string_to_ascii_bytes(self):
        """ASCII strings should be encoded to bytes."""
        result = _sanitise_attribute_value("hello")
        assert result == np.bytes_(b"hello")

    def test_unicode_string_to_escaped_bytes(self):
        """Non-ASCII Unicode should be backslash-escaped."""
        result = _sanitise_attribute_value("café")
        assert isinstance(result, np.bytes_)
        # Should not raise UnicodeEncodeError
        # The é should be escaped as \\xe9
        assert b"caf\\xe9" in result  # spellchecker:disable-line

    def test_datetime_to_string(self):
        """Datetime values should be converted to strings."""
        dt = np.datetime64("2026-04-20")
        result = _sanitise_attribute_value(dt)
        assert isinstance(result, np.bytes_)
        assert b"2026" in result

    def test_complex_to_string(self):
        """Complex numbers should be converted to strings."""
        val = complex(1, 2)
        result = _sanitise_attribute_value(val)
        assert isinstance(result, np.bytes_)
        assert b"1" in result and b"2" in result
