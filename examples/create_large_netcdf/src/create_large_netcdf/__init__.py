"""Create a 10000x3000x3000 NetCDF tomogram with smoothly varying data.

Launched via mpiexec from create_large_netcdf.pbs.py. PBSRunner determines each
process's role (scheduler, client, or worker) from the MPI rank and bootstraps
via a scheduler file on the shared filesystem — no direct TCP required.
"""

import os
import socket
from pathlib import Path


import dask.array as da  # noqa: E402
from dask.distributed import Client  # noqa: E402
from dask_jobqueue.runner import BaseRunner, Role  # noqa: E402
from mpi4py import MPI
import numpy as np  # noqa: E402

import anu_ctlab_io  # noqa: E402
import anu_ctlab_io.netcdf  # noqa: E402


class MPIRunner(BaseRunner):
    def __init__(self, *args, **kwargs):
        self.comm = MPI.COMM_WORLD
        self.proc_id = self.comm.Get_rank()
        scheduler_opts = kwargs.setdefault("scheduler_options", {})
        scheduler_opts.setdefault("port", 0)
        scheduler_opts.setdefault("dashboard_address", ":0")
        super().__init__(*args, **kwargs)

    async def get_role(self) -> Role:
        rank = self.proc_id
        if rank == 0 and self.scheduler:
            return Role.scheduler
        elif rank == 1 and self.client:
            return Role.client
        else:
            return Role.worker

    async def set_scheduler_address(self, scheduler) -> None:
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.comm.bcast, scheduler.address, 0)

    async def get_scheduler_address(self) -> str:
        import asyncio
        loop = asyncio.get_event_loop()
        address = await loop.run_in_executor(None, self.comm.bcast, None, 0)
        return address


SHAPE = (10000, 3000, 3000)
MAX_FILE_SIZE_MB = 500


def smooth_field(z, y, x):
    """Generate a smoothly varying field using scaled cosines over [0, 65534].

    Each component has amplitude 65534/6 ≈ 10922, so the sum of three
    components spans the full uint16 range smoothly without banding.
    """
    amp = 65534 / 6
    field = (
        amp * (np.cos(2 * np.pi * z / SHAPE[0]) + 1)
        + amp * (np.cos(4 * np.pi * y / SHAPE[1]) + 1)
        + amp * (np.cos(6 * np.pi * x / SHAPE[2]) + 1)
    )
    return field.astype(np.uint16)


def main():
    # Only the client rank (proc_id == 1) executes the body of the with block.
    with MPIRunner() as runner, Client(runner) as client:
        output_dir = Path(os.environ.get("PBS_O_WORKDIR", ".")) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        n_workers = MPI.COMM_WORLD.Get_size() - 2  # all ranks minus scheduler and client
        client.wait_for_workers(n_workers)

        scheduler_info = client.scheduler_info()
        dashboard_port = scheduler_info["services"]["dashboard"]
        scheduler_host = socket.gethostname()

        print(f"Output:  {output_dir / 'tomoHiRes_nc'}")
        print()
        print("Dashboard: To view the Dask dashboard, run:")
        user = os.environ.get("USER", "USERNAME")
        local_port = 8787
        print(f"  ssh -N -L {local_port}:{scheduler_host}:{dashboard_port} -J {user}@gadi.nci.org.au {user}@{scheduler_host}")
        print(f"  Then open http://localhost:{local_port}/status")
        print()

        data = da.fromfunction(smooth_field, shape=SHAPE, chunks=(256, 256, 256), dtype=np.uint16)

        dataset = anu_ctlab_io.Dataset(
            data=data,
            dimension_names=("z", "y", "x"),
            voxel_unit=anu_ctlab_io.VoxelUnit.MM,
            voxel_size=(1.0, 1.0, 1.0),
            datatype=anu_ctlab_io.DataType.TOMO,
        )

        anu_ctlab_io.netcdf.dataset_to_netcdf(
            dataset,
            output_dir / "tomoHiRes_nc",
            compression_level=0,
            max_file_size_mb=MAX_FILE_SIZE_MB,
        )

    print("Done.")
