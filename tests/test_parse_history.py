#!/usr/bin/env python
from pprint import pp
import re
import anu_ctlab_io.netcdf as nc


def test_parse_history():
    ctlab_dataset = nc.CTLabDataset.from_netcdf(
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
