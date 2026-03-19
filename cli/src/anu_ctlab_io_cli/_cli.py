"""CLI entry point for anu-ctlab-io-convert."""

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from dask.delayed import Delayed
from dask.distributed import print


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


class Scheduler(str, Enum):
    threads = "threads"
    processes = "processes"
    distributed = "distributed"
    distributed_mpi = "distributed-mpi"


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
    scheduler: Annotated[
        Scheduler,
        typer.Option(
            "--scheduler",
            help="Dask scheduler to use.",
        ),
    ] = Scheduler.processes,
) -> None:
    """Convert between ANU CTLab array formats."""
    match scheduler:
        case Scheduler.distributed_mpi | Scheduler.distributed:
            if scheduler == Scheduler.distributed_mpi:
                from dask_mpi import initialize

                initialize()  # ranks 0 and 2+ block here as scheduler/workers; only rank 1 returns

            from dask.distributed import Client, progress, wait

            with Client() as client:
                result = _convert(input, output, input_format, output_format)
                if result is not None:
                    future = client.compute(result)
                    progress(future)
                    wait(future)
            if scheduler == Scheduler.distributed_mpi:
                import os

                os._exit(0)  # bypass atexit handlers to avoid dask-mpi hang on exit
        case Scheduler.threads | Scheduler.processes:
            from dask.diagnostics import ProgressBar

            with ProgressBar():
                result = _convert(input, output, input_format, output_format)
                if result is not None:
                    result.compute(scheduler=scheduler.value)


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def _print_dataset_info(dataset) -> None:
    data = dataset.data
    z, y, x = data.shape
    cz, cy, cx = data.chunksize
    chunk_bytes = cz * cy * cx * data.dtype.itemsize
    vz, vy, vx = dataset.voxel_size
    print(f"  shape:      ({z}, {y}, {x})")
    print(f"  dtype:      {data.dtype}")
    print(f"  size:       {_fmt_bytes(data.nbytes)}")
    print(f"  chunks:     ({cz}, {cy}, {cx})  —  {_fmt_bytes(chunk_bytes)} each")
    print(f"  num chunks: {data.npartitions}")
    print(f"  voxel size: ({vz}, {vy}, {vx}) {dataset.voxel_unit}")


def _convert(
    input: Path,
    output: Path,
    input_format: InputStorageFormat,
    output_format: OutputStorageFormat,
) -> Delayed | None:
    from anu_ctlab_io import Dataset

    dataset = Dataset.from_path(input, filetype=input_format.value)
    print(f"Input: {input}")
    _print_dataset_info(dataset)
    print(f"Output: {output}")
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
