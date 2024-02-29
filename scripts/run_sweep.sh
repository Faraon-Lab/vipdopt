#!/bin/bash
#SBATCH -A Faraon_Computing
#SBATCH --time=2:00:00   # walltime

#SBATCH --nodes=10   # number of nodes
#SBATCH --ntasks-per-node=8  # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=8G

#SBATCH -J "Sony full sweep"   # job name

#SBATCH --mail-user=nmcnichols@caltech.edu
#SBATH --qos=normal

## /SBATCH -p general # partition (queue)
## /SBATCH -o slurm.%N.%j.out # STDOUT
## /SBATCH -e slurm.%N.%j.err # STDERR

## Load relevant modules
source /home/${USER}/.bashrc
source /central/groups/Faraon_Computing/nia/miniconda3/etc/profile.d/conda.sh
conda activate vipdopt-dev

xvfb-run --server-args="-screen 0 1280x1024x24" python vipdopt optimize runs/test_run processed_config.yaml