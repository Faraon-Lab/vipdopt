#!/bin/bash
#SBATCH -A Faraon_Computing
#SBATCH --time=0:10:00   # walltime

#SBATCH --nodes=2   # number of nodes
#SBATCH --ntasks-per-node=8  # number of processor cores (i.e. tasks)
#SBATCH --mem=1G

#SBATCH -J "test-pool"   # job name

#SBATCH --mail-user=nmcnichols@caltech.edu
#SBATH --qos=debug

## /SBATCH -p general # partition (queue)
## /SBATCH -o slurm.%N.%j.out # STDOUT
## /SBATCH -e slurm.%N.%j.err # STDERR

## Load relevant modules
source /home/${USER}/.bashrc
source /central/groups/Faraon_Computing/nia/miniconda3/etc/profile.d/conda.sh
conda activate vipdopt-dev

srun hostname -s | sort -u >slurm.hosts
export MAX_WORKERS=$SLURM_NPROCS
mpiexec.hydra -f slurm.hosts -np 1 python -m vipdopt.mpi.pool

## nodelist=$(scontrol show hostname $SLURM_NODELIST)
## printf "%s\n" "${nodelist[@]}" > nodefile

# mpirun -n 1 python -m pytest tests/test_pool.py --with-mpi -v
# mpirun -n 1 python -m vipdopt.mpi.pool