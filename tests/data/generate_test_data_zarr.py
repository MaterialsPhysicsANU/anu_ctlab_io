#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "zarr==3.0.5",
#     "numpy==2.2.3",
# ]
# ///

import json

import numpy as np
import zarr

chunk_key_encoding = {"name": "default", "separator": "."}
schema_example = "../schema/anu_ctlab_zarr_1_0_example.json"

### Little array
arr = zarr.create_array(
    "./tomoLoRes.zarr",
    dimension_names=["z", "y", "x"],
    shape=(10, 20, 30),
    chunks=(3, 20, 30),
    dtype=np.uint16,
    chunk_key_encoding=chunk_key_encoding,
    overwrite=True,
)
arr[:] = np.arange(np.prod(arr.shape)).reshape(arr.shape)
with open(schema_example) as f:
    arr.attrs["mango"] = json.load(f)

### Big array
arr = zarr.create_array(
    "./tomoHiRes.zarr",
    dimension_names=["z", "y", "x"],
    shape=(100, 200, 300),
    chunks=(30, 200, 300),
    dtype=np.uint16,
    chunk_key_encoding=chunk_key_encoding,
    overwrite=True,
)

shape_z = arr.shape[0]
chunk_z = arr.chunks[0]
num_chunks_z = (shape_z + chunk_z - 1) // chunk_z
for i in range(num_chunks_z):
    arr[i * chunk_z : min((i + 1) * chunk_z, shape_z), ...] = i
with open(schema_example) as f:
    arr.attrs["mango"] = json.load(f)
