#!/bin/bash

#SBATCH --time=167:00:00
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=8
#SBATCH --qos=normal
#SBATCH --mem-per-cpu=8G
#SBATCH --comment="SONY th0 case, Area FoM Test"
#SBATCH --mail-user=ianfoomz@gmail.com
#SBATCH --mail-type=END

source activate fdtd

xvfb-run --server-args="-screen 0 1280x1024x24" python SonyBayerFilterOptimization.py 10 > stdout_mwir_g0.log 2> stderr_mwir_g0.log

exit $?
