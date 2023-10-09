#!/bin/bash
#SBATCH -A Faraon_Computing
#SBATCH --time=0:05:00   # walltime

#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1  # number of processor cores (i.e. tasks)
#SBATCH --mem=1G

#SBATCH -J "test-pool"   # job name

#SBATCH --mail-user=nmcnichols@caltech.edu
#SBATH --qos=debug

## /SBATCH -p general # partition (queue)
## /SBATCH -o slurm.%N.%j.out # STDOUT
## /SBATCH -e slurm.%N.%j.err # STDERR

## Load relevant modules
source /home/${USER}/.bashrc
source activate vipdopt-dev

time srun -n $SLURM_NPROCS python -m mpi4py.futures vipdopt/manager.py