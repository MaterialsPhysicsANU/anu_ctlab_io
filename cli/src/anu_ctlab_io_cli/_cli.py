"""CLI entry point for anu-ctlab-io."""

import logging
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

import typer
import zarr
from dask.delayed import Delayed

from anu_ctlab_io._datatype import DataType
from anu_ctlab_io._voxel_properties import VoxelUnit

logger = logging.getLogger(__name__)

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})

VALID_DATATYPES = ", ".join(str(dt) for dt in DataType)


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
    synchronous = "synchronous"
    threads = "threads"
    processes = "processes"
    distributed = "distributed"
    distributed_mpi = "distributed-mpi"


def _parse_voxel_size(value: str) -> tuple[float, float, float]:
    """Parse a comma-separated string into a tuple of three floats."""
    parts = [x.strip() for x in value.split(",")]
    if len(parts) != 3:
        raise typer.BadParameter(
            "Voxel size must be three comma-separated values (e.g., 0.5,0.5,0.5).",
            param_hint="--voxel-size",
        )
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError as e:
        raise typer.BadParameter(
            f"Voxel size values must be numbers: {value}",
            param_hint="--voxel-size",
        ) from e


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
    datatype: Annotated[
        str | None,
        typer.Option(
            "--datatype",
            help=f"Datatype to use when writing. Valid types: {VALID_DATATYPES}. Required when converting plain Zarr arrays without mango attributes to NetCDF.",
        ),
    ] = None,
    scheduler: Annotated[
        Scheduler,
        typer.Option(
            "--scheduler",
            help="Dask scheduler to use.",
        ),
    ] = Scheduler.threads,
    voxel_size: Annotated[
        str | None,
        typer.Option(
            "--voxel-size",
            help="Override voxel size as three comma-separated values (e.g., 0.5,0.5,0.5).",
        ),
    ] = None,
    voxel_unit: Annotated[
        VoxelUnit | None,
        typer.Option(
            "--voxel-unit",
            help="Override voxel unit (e.g., nm, um, mm, angstrom).",
        ),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Logging level (DEBUG, INFO, WARNING, ERROR).",
        ),
    ] = "INFO",
) -> None:
    """Convert between ANU CTLab array formats."""
    logging.getLogger().setLevel(log_level.upper())
    match scheduler:
        case Scheduler.distributed_mpi | Scheduler.distributed:
            if scheduler == Scheduler.distributed_mpi:
                try:
                    from dask_mpi import initialize
                except ImportError as e:
                    raise typer.BadParameter(
                        "dask-mpi is not installed. "
                        "Re-install with MPI support: pip install anu-ctlab-io-cli[mpi]",
                        param_hint="--scheduler",
                    ) from e

                initialize()  # ranks 0 and 2+ block here as scheduler/workers; only rank 1 returns

            from dask.distributed import Client, progress, wait

            with Client() as client:
                result = _convert(
                    input,
                    output,
                    input_format,
                    output_format,
                    datatype,
                    _parse_voxel_size(voxel_size) if voxel_size else None,
                    voxel_unit,
                )
                if result is not None:
                    future = client.compute(result)
                    progress(future)
                    wait(future)
            if scheduler == Scheduler.distributed_mpi:
                import os

                os._exit(0)  # bypass atexit handlers to avoid dask-mpi hang on exit
        case Scheduler.synchronous | Scheduler.threads | Scheduler.processes:
            from dask.diagnostics import ProgressBar

            parsed_voxel_size = _parse_voxel_size(voxel_size) if voxel_size else None

            with ProgressBar():
                result = _convert(
                    input,
                    output,
                    input_format,
                    output_format,
                    datatype,
                    parsed_voxel_size,
                    voxel_unit,
                )
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
    logger.info("  shape:      (%s, %s, %s)", z, y, x)
    logger.info("  dtype:      %s", data.dtype)
    logger.info("  size:       %s", _fmt_bytes(data.nbytes))
    logger.info(
        "  chunks:     (%s, %s, %s)  —  %s each", cz, cy, cx, _fmt_bytes(chunk_bytes)
    )
    logger.info("  num chunks: %s", data.npartitions)
    logger.info("  voxel size: (%s, %s, %s) %s", vz, vy, vx, dataset.voxel_unit)


def _convert(
    input: Path,
    output: Path,
    input_format: InputStorageFormat,
    output_format: OutputStorageFormat,
    datatype: str | None = None,
    voxel_size: tuple[float, float, float] | None = None,
    voxel_unit: VoxelUnit | None = None,
) -> Delayed | None:
    from anu_ctlab_io import Dataset

    dataset = Dataset.from_path(input, filetype=input_format.value)

    # Apply voxel overrides if provided
    if voxel_size is not None:
        dataset._voxel_size = voxel_size
    if voxel_unit is not None:
        dataset._voxel_unit = voxel_unit

    logger.info("Input: %s", input)
    _print_dataset_info(dataset)
    logger.info("Output: %s", output)
    kwargs: dict[str, Any] = {"filetype": output_format.value, "compute": False}
    if datatype is not None:
        kwargs["datatype"] = datatype
    try:
        return dataset.to_path(output, **kwargs)
    except ValueError as e:
        if "datatype must be provided" in str(e):
            raise typer.BadParameter(
                f"No datatype could be inferred from the input dataset. "
                f"Please specify one explicitly using --datatype. "
                f"Valid datatypes are: {VALID_DATATYPES}",
                param_hint="--datatype",
            ) from e
        raise


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    typer.run(cli)
