"""Create a 10000x3000x3000 NetCDF tomogram with smoothly varying data.

Launched via mpiexec. dask-mpi assigns each process a role automatically:
rank 0 → scheduler, rank 1 → client (returns from initialize()), rank 2+ → workers.
"""

from dask.distributed import Client
from dask_mpi import initialize

from create_large_netcdf import run


def main():
    initialize()  # ranks 0 and 2+ block here as scheduler/workers; only rank 1 returns

    with Client() as client:
        run(client)

    print("Done.")
