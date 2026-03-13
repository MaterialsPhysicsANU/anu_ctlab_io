"""CLI entry point for anu-ctlab-io-convert."""

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer


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
        typer.Option("--input-format", help="Input format (default: auto-detect from extension)."),
    ] = InputStorageFormat.auto,
    output_format: Annotated[
        OutputStorageFormat,
        typer.Option("--output-format", help="Output format (default: auto-detect from extension)."),
    ] = OutputStorageFormat.auto,
) -> None:
    """Convert between ANU CTLab array formats."""
    from anu_ctlab_io import Dataset

    dataset = Dataset.from_path(input, filetype=input_format.value)
    try:
        dataset.to_path(output, filetype=output_format.value)
    except ValueError as err:
        raise typer.BadParameter(
            f"cannot infer output format from '{output.name}'. "
            "Specify one of: NetCDF, zarr, raw.",
            param_hint="--output-format",
        ) from err


def main() -> None:
    typer.run(cli)
