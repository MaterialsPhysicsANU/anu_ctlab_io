"""Create a 10000×3000×3000 NetCDF tomogram with smoothly varying data.

Launched via mpiexec from create_large_netcdf.sh. PBSRunner determines each
process's role (scheduler, client, or worker) from the MPI rank and bootstraps
via a scheduler file on the shared filesystem — no direct TCP required.
"""

import asyncio
import json
import os
from pathlib import Path

import dask.array as da
import numpy as np
from dask.distributed import Client
from dask_jobqueue.runner import BaseRunner, Role
from mpi4py import MPI

import anu_ctlab_io
import anu_ctlab_io.netcdf


class PBSRunner(BaseRunner):
    def __init__(self, *args, scheduler_file: str | Path | None = None, **kwargs):
        self.proc_id = MPI.COMM_WORLD.Get_rank()
        job_id = os.environ.get("PBS_JOBID", "local")
        self.scheduler_file = Path(scheduler_file or f"scheduler-{job_id}.json")
        options = {"scheduler_file": str(self.scheduler_file)}
        super().__init__(
            *args, worker_options=options, scheduler_options=options, **kwargs
        )

    async def get_role(self) -> Role:
        if self.proc_id == 0 and self.scheduler:
            return Role.scheduler
        elif self.proc_id == 1 and self.client:
            return Role.client
        else:
            return Role.worker

    async def get_scheduler_address(self) -> str:
        while not self.scheduler_file.exists():
            await asyncio.sleep(0.2)
        return json.loads(self.scheduler_file.read_text())["address"]


SHAPE = (10000, 3000, 3000)
MAX_FILE_SIZE_MB = 500
OUTPUT_DIR = Path(os.environ.get("PBS_O_WORKDIR", ".")) / "output"

# Align chunks to file boundaries:
#   bytes per z-slice = 3000 * 3000 * 2 (uint16) = ~17.2 MB
#   slices per 500 MB file = floor(500 * 1024^2 / (3000 * 3000 * 2)) = 29
BYTES_PER_SLICE = SHAPE[1] * SHAPE[2] * np.dtype(np.uint16).itemsize
SLICES_PER_FILE = max(1, int((MAX_FILE_SIZE_MB * 1024 * 1024) / BYTES_PER_SLICE))
CHUNKS = (SLICES_PER_FILE, SHAPE[1], SHAPE[2])

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def smooth_block(_block: np.ndarray, block_info: dict) -> np.ndarray:
    """Generate a smoothly varying block using a sum of low-frequency cosines.

    f(z, y, x) = cos(2π z/Z) + cos(4π y/Y) + cos(6π x/X), scaled to [0, 65535].
    """
    (z0, y0, x0), (z1, y1, x1) = block_info[0]["array-location"]

    z = np.arange(z0, z1)[:, None, None] / SHAPE[0]
    y = np.arange(y0, y1)[None, :, None] / SHAPE[1]
    x = np.arange(x0, x1)[None, None, :] / SHAPE[2]

    field = np.cos(2 * np.pi * z) + np.cos(4 * np.pi * y) + np.cos(6 * np.pi * x)
    # field is in [-3, 3]; scale to uint16
    return ((field + 3) / 6 * 65535).astype(np.uint16)


# Only the client rank (proc_id == 1) executes the body of the with block.
with PBSRunner() as runner, Client(runner) as client:
    print(f"Shape:   {SHAPE}")
    print(f"Chunks:  {CHUNKS}  (~{SLICES_PER_FILE} slices per file)")
    print(f"Workers: {len(client.scheduler_info()['workers'])}")
    print(f"Output:  {OUTPUT_DIR / 'tomoHiRes_nc'}")

    data = da.map_blocks(
        smooth_block,
        da.zeros(SHAPE, chunks=CHUNKS, dtype=np.uint16),
        dtype=np.uint16,
        meta=np.empty((0,), dtype=np.uint16),
    )

    dataset = anu_ctlab_io.Dataset(
        data=data,
        dimension_names=("z", "y", "x"),
        voxel_unit=anu_ctlab_io.VoxelUnit.MM,
        voxel_size=(1.0, 1.0, 1.0),
        datatype=anu_ctlab_io.DataType.TOMO,
    )

    anu_ctlab_io.netcdf.dataset_to_netcdf(
        dataset,
        OUTPUT_DIR / "tomoHiRes_nc",
        compression_level=0,
        max_file_size_mb=MAX_FILE_SIZE_MB,
    )

print("Done.")
