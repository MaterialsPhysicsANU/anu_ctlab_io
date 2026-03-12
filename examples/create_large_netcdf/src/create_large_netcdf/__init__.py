import os
import socket
from pathlib import Path

import dask.array as da
import numpy as np
from dask.distributed import Client

import anu_ctlab_io
import anu_ctlab_io.netcdf

SHAPE = (10000, 3000, 3000)
MAX_FILE_SIZE_MB = 500


def run(client: Client) -> None:
    """Create and write the large NetCDF file using the given dask client."""
    output_dir = Path(os.environ.get("PBS_O_WORKDIR", ".")) / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    scheduler_info = client.scheduler_info()
    dashboard_port = scheduler_info["services"]["dashboard"]
    scheduler_host = socket.gethostname()

    print(f"Output:  {output_dir / 'tomoHiRes_nc'}")
    print()
    print("Dashboard: To view the Dask dashboard, run:")
    user = os.environ.get("USER", "USERNAME")
    local_port = 8787
    print(
        f"  ssh -N -L {local_port}:{scheduler_host}:{dashboard_port}"
        f" -J {user}@gadi.nci.org.au {user}@{scheduler_host}"
    )
    print(f"  Then open http://localhost:{local_port}/status")
    print()

    data = da.fromfunction(
        smooth_field, shape=SHAPE, chunks=(256, 256, 256), dtype=np.uint16
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
        output_dir / "tomoHiRes_nc",
        compression_level=0,
        max_file_size_mb=MAX_FILE_SIZE_MB,
    )


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
