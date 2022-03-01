#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=8
#SBATCH --qos=normal
#SBATCH --mem-per-cpu=8G
#SBATCH --comment="LumProc: MWIR th0 case, srcRad 15um - Incident Angle Sweep"
#SBATCH --mail-user=ianfoomz@gmail.com
#SBATCH --mail-type=END

source activate fdtd

xvfb-run --server-args="-screen 0 1280x1024x24" python LumProcSweep.py 10 > stdout_mwir_g0.log 2> stderr_mwir_g0.log

exit $?
