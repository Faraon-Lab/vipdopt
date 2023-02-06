#!/bin/bash

#SBATCH -A Faraon_Computing
#SBATCH --time=167:00:00
#SBATCH --nodes=10
## SBATCH --ntasks-per-node=8
#SBATCH --ntasks=80
#SBATCH --qos=normal
#SBATCH --mem-per-cpu=12G
#SBATCH --comment="SONY22_varTh_case8_th63div13"
#SBATCH --mail-user=ianfoomz@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


source activate fdtd

xvfb-run --server-args="-screen 0 1280x1024x24" python SonyBayerFilterOptimization.py 10 > stdout_mwir_g0.log 2> stderr_mwir_g0.log

exit $?
