#!/bin/bash
#SBATCH -A Faraon_Computing
#SBATCH --time=0:30:00   # walltime

#SBATCH --ntasks-per-node=1 # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=50M   # memory per CPU core

#SBATCH -J "test"   # job name

#SBATCH --mail-user=nmcnichols@caltech.edu
#SBATH --qos=debug

## /SBATCH -p general # partition (queue)
## /SBATCH -o slurm.%N.%j.out # STDOUT
## /SBATCH -e slurm.%N.%j.err # STDERR

source /home/${USER}/.bashrc
source activate vipdopt-dev

srun --mpi=pmi2 /groups/Faraon_Computing/nia/SonyBayerFilter/vipdopt/manager.py