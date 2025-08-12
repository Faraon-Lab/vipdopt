#!/bin/bash

DIR=${1:-test_run}
CONFIG=${2:-}
sbatch --job-name "vidpopt_v213_test_"$DIR slurm_vipdopt.sh $DIR $CONFIG