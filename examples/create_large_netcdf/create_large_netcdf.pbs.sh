#!/bin/bash
#
# PBS job script — submit with: qsub create_large_netcdf.pbs.sh
#
#PBS -N create_netcdf
#PBS -q normal
#PBS -l ncpus=96
#PBS -l mem=384GB
#PBS -l jobfs=800GB
#PBS -l walltime=00:30:00
#PBS -l storage=scratch/ap38+gdata/ap38+gdata/w09
#PBS -l wd
#PBS -j oe

module load openmpi

if [ -z "${PBS_JOBID}" ]; then
    uv sync
    exec qsub "$0"
fi

# NOTE: Not portable, needs to run from project directory
mpirun -n "${PBS_NCPUS}" \
    "${PBS_O_WORKDIR}/.venv/bin/create-large-netcdf"
