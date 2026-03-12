# create_large_netcdf

Creates a 10000×3000×3000 uint16 NetCDF tomogram with smoothly varying data, split into 500 MB files.

Demonstrates two approaches to spinning up a multi-node `dask` job on Gadi.

## Approaches

### mpirun — `dask-mpi`

All processes are launched together via `mpirun`. `dask-mpi` assigns roles by rank:

| Rank | Role |
|------|------|
| 0 | Scheduler |
| 1 | Client (runs the computation) |
| 2+ | Workers |

**Resources:** 2 nodes × 48 CPUs = 96 workers.

```bash
qsub create_large_netcdf_mpirun.pbs.sh
# or ./create_large_netcdf_mpirun.pbs.sh # Self submitting
```

### PBSCluster — `dask-jobqueue`

A lightweight driver job runs the Dask scheduler and submits a worker sub-job via `qsub`.
The scheduler binds to the login node's internal IP so compute nodes can reach it.

**Resources:** interactive node (1 CPU) + 1 worker job (2 nodes × 48 CPUs = 96 workers).

```bash
uv run create-large-netcdf-pbscluster
```

## Dask dashboard

Both approaches print an SSH tunnel command on startup:

```
ssh -N -L <local_port>:<scheduler-host>:<port> -J <user>@gadi.nci.org.au <user>@<scheduler-host>
```

Then open <http://localhost:8787/status>.

The local port can be adjusted if it is already bound.
