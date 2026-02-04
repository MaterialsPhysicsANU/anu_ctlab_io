#!/usr/bin/env python
import re
from pprint import pp

import pytest

import anu_ctlab_io
from anu_ctlab_io._parse_history import parse_history

try:
    import anu_ctlab_io.netcdf

    _HAS_NETCDF = True
except ImportError:
    _HAS_NETCDF = False


@pytest.mark.skipif(not _HAS_NETCDF, reason="Requires 'netcdf' extra")
def test_parse_history():
    dataset = anu_ctlab_io.Dataset.from_path(
        "tests/data/tomoHiRes_SS_nc",
    )
    for k, v in dataset.history.items():
        assert isinstance(v, dict), "history item is not a dictionary"
        assert re.match(r"(19|20)\d{6}_\d{6}(_\S+)*", k), (
            "history key appears not to be valid mango filename"
        )
        pp(k)
        pp(v)


def test_parse_col_sep():
    # Test parsing of log-style key: value format with angle brackets
    # Angle brackets are stripped per design decision
    assert parse_history(
        """

        machine: <lapac71h>
        start time: <Fri Mar 14 12:29:19 2025>
        mango git repo: <git@github.com:MaterialsPhysicsANU/mango.git>
        """
    ) == {
        "machine": "lapac71h",
        "start time": "Fri Mar 14 12:29:19 2025",
        "mango git repo": "git@github.com:MaterialsPhysicsANU/mango.git",
    }


def test_parse_log_repeated_keys():
    # Test that repeated keys become lists
    result = parse_history(
        """
        input dataset ID: _file1
        input dataset ID: _file2
        input dataset ID: _file3
        """
    )
    assert result == {"input dataset ID": ["_file1", "_file2", "_file3"]}


def test_parse_log_with_unstructured():
    # Test log format with unstructured text lines
    result = parse_history(
        """
        input dataset ID: _file1
        Out-of-core: Reconstruct_Multigrid: Recon volume origin = [-5520  -656  -656]
        Re-scaling data linearly to unsigned short [us(f)=55530.0*f + 10000.0]
        input dataset ID: _file2
        """
    )
    assert "input dataset ID" in result
    assert result["input dataset ID"] == ["_file1", "_file2"]
    assert "_log_text" in result
    assert len(result["_log_text"]) == 2


def test_parse_boolean_conversion():
    # Test that boolean values are converted
    result = parse_history(
        """
        verbosity_level         high
        BeginSection Test
            enabled                 True
            disabled                False
        EndSection
        """
    )
    assert result["Test"]["enabled"] is True
    assert result["Test"]["disabled"] is False


def test_parse_angle_bracket_array():
    # Test that multiple angle brackets become lists with numeric conversion
    result = parse_history(
        """
        BeginSection Geometry
            field_of_view           <20.130377><20.130377><128.712411>
            num_voxels              <100><200><300>
        EndSection
        """
    )
    assert result["Geometry"]["field_of_view"] == [20.130377, 20.130377, 128.712411]
    assert result["Geometry"]["num_voxels"] == [100, 200, 300]


def test_parse_empty_values():
    # Test that empty values are handled (requires at least one space after key)
    # Note: Using explicit string construction to preserve trailing spaces
    # (pre-commit trailing-whitespace hook would strip them from string literals)
    history_str = "\n".join(
        [
            "",
            "        verbosity_level         high",
            "        empty_value             ",  # Has trailing spaces for empty value test
            "        BeginSection Test",
            "            run_identifier      ",  # Has trailing spaces for empty value test
            "        EndSection",
            "        ",
        ]
    )
    result = parse_history(history_str)
    assert result["empty_value"] == ""
    assert result["Test"]["run_identifier"] == ""


def test_parse_always_returns_dict():
    # Test that parser always returns dict, never string
    # Even for malformed input
    result = parse_history("some random text without structure")
    assert isinstance(result, dict)
    assert "_raw_history" in result


def test_serialize_history_structured():
    # Test serialization of structured format
    from anu_ctlab_io._parse_history import serialize_history

    structured = """
    verbosity_level         high
    BeginSection MPI
        num_processors_x    2
        enabled             True
    EndSection
    BeginSection Geometry
        field_of_view       <20.1><30.2><40.3>
    EndSection
    """
    parsed = parse_history(structured)
    serialized = serialize_history(parsed)
    reparsed = parse_history(serialized)

    assert parsed == reparsed, "Structured format should roundtrip perfectly"


def test_serialize_history_log_repeated():
    # Test serialization of log format with repeated keys
    from anu_ctlab_io._parse_history import serialize_history

    log_format = """input dataset ID: value1
input dataset ID: value2
input dataset ID: value3"""

    parsed = parse_history(log_format)
    assert isinstance(parsed["input dataset ID"], list)
    assert len(parsed["input dataset ID"]) == 3

    serialized = serialize_history(parsed)
    reparsed = parse_history(serialized)

    assert parsed == reparsed, "Log format with repeated keys should roundtrip"
    assert reparsed["input dataset ID"] == ["value1", "value2", "value3"]


def test_serialize_history_log_unstructured():
    # Test serialization of log format with unstructured text
    from anu_ctlab_io._parse_history import serialize_history

    log_with_text = """key1: value1
Some unstructured log line
key2: value2
Another log line without colon"""

    parsed = parse_history(log_with_text)
    assert "_log_text" in parsed
    assert len(parsed["_log_text"]) == 2

    serialized = serialize_history(parsed)
    reparsed = parse_history(serialized)

    assert parsed == reparsed, "Log format with unstructured text should roundtrip"
    assert reparsed["_log_text"] == parsed["_log_text"]


def test_serialize_history_angle_brackets():
    # Test that angle brackets are preserved during roundtrip
    from anu_ctlab_io._parse_history import serialize_history

    with_brackets = """executable: </path/to/executable>
machine: <hostname>"""

    parsed = parse_history(with_brackets)
    # Angle brackets should be stripped during parsing
    assert parsed["executable"] == "/path/to/executable"
    assert parsed["machine"] == "hostname"

    serialized = serialize_history(parsed)
    # Should NOT have angle brackets in serialized form (they were stripped)
    assert "<" not in serialized

    reparsed = parse_history(serialized)
    assert parsed == reparsed


def test_serialize_history_angle_bracket_arrays():
    # Test that angle bracket arrays roundtrip correctly
    from anu_ctlab_io._parse_history import serialize_history

    structured = """
    BeginSection Geometry
        field_of_view       <20.1><30.2><40.3>
        num_voxels          <100><200><300>
    EndSection
    """

    parsed = parse_history(structured)
    assert parsed["Geometry"]["field_of_view"] == [20.1, 30.2, 40.3]
    assert parsed["Geometry"]["num_voxels"] == [100, 200, 300]

    serialized = serialize_history(parsed)
    # Should reconstruct angle bracket format
    assert "<20.1><30.2><40.3>" in serialized
    assert "<100><200><300>" in serialized

    reparsed = parse_history(serialized)
    assert parsed == reparsed


def test_parse_recursive():
    assert parse_history(
        """
        verbosity_level         high
        BeginSection MPI
            num_bytes_in_chunk      10000000
        EndSection
        BeginSection Output_Data_File
            compression             NONE
            BeginSection netcdf
                operator_name           Ben Young
                internal_compression    on
                multi_string            __start_multi_string__
        a b c
        d e f
        __end_multi_string__
            EndSection
        EndSection
        """
    ) == {
        "verbosity_level": "high",
        "MPI": {"num_bytes_in_chunk": "10000000"},
        "Output_Data_File": {
            "compression": "NONE",
            "netcdf": {
                "operator_name": "Ben Young",
                "internal_compression": "on",
                "multi_string": "a b c\n        d e f\n        ",
            },
        },
    }
