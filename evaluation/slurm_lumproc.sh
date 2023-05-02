#!/bin/bash

#SBATCH --time=2:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --qos=normal
#SBATCH --mem-per-cpu=4G
#SBATCH --comment="LumProc: Merge Test 5 - ADAM lr 1e-2, Restart from Epoch 5, Density"
#SBATCH --mail-user=ianfoomz@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

source activate fdtd

xvfb-run --server-args="-screen 0 1280x1024x24" python LumProcSweep.py --filename "configs/test_config_sony.yaml" > stdout_mwir_g0.log 2> stderr_mwir_g0.log

exit $?
