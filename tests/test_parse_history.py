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
    assert parse_history(
        """

        machine: <lapac71h>
        start time: <Fri Mar 14 12:29:19 2025>
        mango git repo: <git@github.com:MaterialsPhysicsANU/mango.git>
        """
    ) == {
        "machine": "<lapac71h>",
        "start time": "<Fri Mar 14 12:29:19 2025>",
        "mango git repo": "<git@github.com:MaterialsPhysicsANU/mango.git>",
    }


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
