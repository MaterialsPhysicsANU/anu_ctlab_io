#!/usr/bin/env python
from pprint import pp
import re
import anu_ctlab_io.netcdf as nc
import anu_ctlab_io.netcdf.parse_history as parse_history


def test_parse_history():
    ctlab_dataset = nc.NetCDFDataset.from_path(
        "tests/data/tomoHiRes_SS_nc",
    )
    dataset = ctlab_dataset._dataset
    for k, v in dataset.attrs["history"].items():
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
                internal_compression    on
            EndSection
        EndSection
        """
    ) == {
        "verbosity_level": "high",
        "MPI": {"num_bytes_in_chunk": "10000000"},
        "Output_Data_File": {
            "compression": "NONE",
            "netcdf": {"internal_compression": "on"},
        },
    }
