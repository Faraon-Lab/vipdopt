#!/bin/bash
#SBATCH -A Faraon_Computing
#SBATCH --time=0:05:00   # walltime

#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=2  # number of processor cores (i.e. tasks)
#SBATCH --mem=1G

#SBATCH -J "jinja2-rendering"   # job name

#SBATCH --mail-user=nmcnichols@caltech.edu
#SBATH --qos=debug

## /SBATCH -p general # partition (queue)
## /SBATCH -o slurm.%N.%j.out # STDOUT
## /SBATCH -e slurm.%N.%j.err # STDERR

## Load relevant modules
source /home/${USER}/.bashrc
source /central/groups/Faraon_Computing/nia/miniconda3/etc/profile.d/conda.sh
conda activate vipdopt-dev

xvfb-run --server-args="-screen 0 1280x1024x24" python vipdopt/configuration/render.py