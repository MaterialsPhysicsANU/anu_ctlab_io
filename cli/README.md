# anu-ctlab-io-cli

Command-line tool for converting between [ANU CTLab](https://anu-ctlab-io.readthedocs.io) array storage formats (NetCDF, Zarr, Raw).

> [!NOTE]
> Future versions will expose more control over output formats, e.g. chunking and compression options for Zarr.

## Installation

```bash
pip install anu-ctlab-io-cli
```

## Usage

```bash
anu-ctlab-io <input> <output> [--input-format FORMAT] [--output-format FORMAT]
```

The format is auto-detected from the file extension by default. The scheduler defaults
to `auto`, which uses `distributed-mpi` when launched under an MPI runtime such as
`mpirun`/`mpiexec`, and `threads` otherwise.

### Examples

```bash
# NetCDF to Zarr
anu-ctlab-io tomoHiRes_nc tomoHiRes.zarr

# NetCDF to Raw binary
anu-ctlab-io tomoHiRes_nc tomoHiRes.raw
```

### Options

| Option            | Values                                                                          | Default |
| ----------------- | ------------------------------------------------------------------------------- | ------- |
| `--input-format`  | `auto`, `NetCDF`, `zarr`                                                        | `auto`  |
| `--output-format` | `auto`, `NetCDF`, `zarr`, `raw`                                                 | `auto`  |
| `--scheduler`     | `auto`, `synchronous`, `threads`, `processes`, `distributed`, `distributed-mpi` | `auto`  |

## License

MIT — see [LICENSE.md](LICENSE.md).
