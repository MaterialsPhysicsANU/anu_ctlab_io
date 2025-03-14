# Generate Zarr data

This generates Zarr V3 data with `"mango"` attributes that are recognised by the `mango` tool used internally at the ANU CTLab.

```bash
./generate_test_data_zarr.py
```

# Convert Zarr to ANU NetCDF4

`mango` can convert to and from Zarr V3 arrays with `"mango"` attributes.

For `tomo{Lo,Hi}Res.zarr`:

- Open input file
- Set `internal_compression` to `on`
- Set `num_mbytes_per_file` to 2
- Apply `Subset` filter with default parameters
