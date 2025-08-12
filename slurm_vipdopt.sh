#!/bin/bash
#SBATCH -A Faraon_Computing
#SBATCH --time=24:00:00   # walltime

#SBATCH --nodes=5   # number of nodes
#SBATCH --ntasks-per-node=8  # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=8G
##SBATCH --mem=16G

##SBATCH -J "vipdopt_v213_test"   # job name

#SBATCH --mail-user=ianfoomz@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --qos=normal

## /SBATCH -p general # partition (queue)
## /SBATCH -o slurm.%N.%j.out # STDOUT
## /SBATCH -e slurm.%N.%j.err # STDERR

## Load relevant modules
# source /home/${USER}/.bashrc
# source /central/groups/Faraon_Computing/nia/miniconda3/etc/profile.d/conda.sh
source activate vipdopt_v213

DIR=${1:-test_run}
CONFIG=${2:-processed_config.yml}

# python -m vipdopt.configuration.template derived_simulation_properties.j2 $DIR/config_example_2d.yml $DIR/processed_config.yml
# python -m vipdopt.configuration.template simulation_template.j2 $DIR/processed_config.yml $DIR/sim.json
xvfb-run --server-args="-screen 0 1280x1024x24" python -m vipdopt.__main__ optimize $DIR --config $CONFIG