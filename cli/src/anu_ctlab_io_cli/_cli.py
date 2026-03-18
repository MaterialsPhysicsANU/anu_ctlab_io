"""CLI entry point for anu-ctlab-io-convert."""

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from dask.delayed import Delayed


# TODO: Make part of the library
class InputStorageFormat(str, Enum):
    auto = "auto"
    netcdf = "NetCDF"
    zarr = "zarr"


class OutputStorageFormat(str, Enum):
    auto = "auto"
    netcdf = "NetCDF"
    zarr = "zarr"
    raw = "raw"


def cli(
    input: Annotated[Path, typer.Argument(help="Input file path.")],
    output: Annotated[Path, typer.Argument(help="Output file path.")],
    input_format: Annotated[
        InputStorageFormat,
        typer.Option(
            "--input-format", help="Input format (default: auto-detect from extension)."
        ),
    ] = InputStorageFormat.auto,
    output_format: Annotated[
        OutputStorageFormat,
        typer.Option(
            "--output-format",
            help="Output format (default: auto-detect from extension).",
        ),
    ] = OutputStorageFormat.auto,
    mpi: Annotated[
        bool,
        typer.Option(
            "--mpi",
            help="Run under MPI using dask-mpi (launch with mpiexec). ",
        ),
    ] = False,
) -> None:
    """Convert between ANU CTLab array formats."""
    if mpi:
        from dask_mpi import initialize

        initialize()  # ranks 0 and 2+ block here as scheduler/workers; only rank 1 returns

        from dask.distributed import Client, progress

        with Client():
            result = _convert(input, output, input_format, output_format)
            if result is not None:
                progress(result)
                result.compute()
    else:
        from dask.diagnostics import ProgressBar

        with ProgressBar():
            result = _convert(input, output, input_format, output_format)
            if result is not None:
                result.compute()


def _convert(
    input: Path,
    output: Path,
    input_format: InputStorageFormat,
    output_format: OutputStorageFormat,
) -> Delayed | None:
    from anu_ctlab_io import Dataset

    dataset = Dataset.from_path(input, filetype=input_format.value)
    try:
        return dataset.to_path(output, filetype=output_format.value, compute=False)
    except ValueError as err:
        raise typer.BadParameter(
            f"cannot infer output format from '{output.name}'. "
            "Specify one of: NetCDF, zarr, raw.",
            param_hint="--output-format",
        ) from err


def main() -> None:
    typer.run(cli)
