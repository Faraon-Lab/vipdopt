#!/bin/bash

#SBATCH -A Faraon_Computing
#SBATCH --time=84:00:00
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=8
## SBATCH --ntasks=80
#SBATCH --qos=normal
#SBATCH --mem-per-cpu=8G
#SBATCH --comment="ADAM Test 40 Layers - LR 5e-2"
#SBATCH --mail-user=ianfoomz@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


source activate fdtd

xvfb-run --server-args="-screen 0 1280x1024x24" python SonyBayerFilterOptimization.py --filename "configs/test_config_sony.yaml" > stdout_mwir_g0.log 2> stderr_mwir_g0.log

exit $?
