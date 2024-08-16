#!/bin/bash

DIR=${1:-test_run}
CONFIG=${2:-}
sbatch --job-name "Sb2S3_"$DIR slurm_vipdopt.sh $DIR $CONFIG