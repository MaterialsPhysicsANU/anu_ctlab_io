"""Create a 10000x3000x3000 NetCDF tomogram with smoothly varying data.

Launched from a lightweight PBS driver job or login node.
PBSCluster submits worker PBS sub-jobs.
The Dask scheduler runs in the driver process.
"""

import os
import socket
from pathlib import Path

from dask.distributed import Client
from dask_jobqueue import PBSCluster

from create_large_netcdf import run

CORES_PER_JOB = 96
MEMORY_PER_JOB = "384GB"
WORKERS_PER_JOB = 96
WORKER_WALLTIME = "00:30:00"


def main():
    account = os.environ.get("PROJECT", "ap38")
    storage = "scratch/ap38+gdata/ap38+gdata/w09"

    log_dir = Path(os.environ.get("PBS_O_WORKDIR", ".")) / "output" / "worker_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Gadi login nodes have both a public IP and an internal 10.x.x.x interface.
    # Compute nodes can only reach the internal interface, so resolve hostname
    # to its internal IP and advertise that as the scheduler host.
    scheduler_host = socket.gethostbyname(socket.gethostname())

    cluster = PBSCluster(
        cores=CORES_PER_JOB,
        memory=MEMORY_PER_JOB,
        processes=WORKERS_PER_JOB,
        queue="normal",
        account=account,
        walltime=WORKER_WALLTIME,
        job_directives_skip=["select"],
        job_extra_directives=[
            f"-l ncpus={CORES_PER_JOB}",
            f"-l mem={MEMORY_PER_JOB}",
            "-l jobfs=800GB",
            f"-l storage={storage}",
            "-l wd",
            "-j oe",
        ],
        local_directory="$PBS_JOBFS/dask-scratch-space",
        log_directory=str(log_dir),
        scheduler_options={
            "host": scheduler_host,
            "port": 0,
            "dashboard_address": ":0",
        },
    )
    cluster.scale(jobs=1)

    with Client(cluster) as client:
        client.wait_for_workers(WORKERS_PER_JOB)
        run(client)

    cluster.close()
    print("Done.")
