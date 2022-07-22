#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=8
#SBATCH --qos=normal
#SBATCH --mem-per-cpu=8G
#SBATCH --comment="LumProc: SONY device divAng 13, th0, PEC 60nm extended"
#SBATCH --mail-user=ianfoomz@gmail.com
#SBATCH --mail-type=END

source activate fdtd

xvfb-run --server-args="-screen 0 1280x1024x24" python LumProcSweep.py 10 > stdout_mwir_g0.log 2> stderr_mwir_g0.log

exit $?
