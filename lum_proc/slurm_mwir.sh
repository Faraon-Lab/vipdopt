#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=8
#SBATCH --qos=normal
#SBATCH --mem-per-cpu=8G
#SBATCH --comment="LumProc: Cu sidewall 450nm - Sidewall Thickness Sweep"
#SBATCH --mail-user=ianfoomz@gmail.com
#SBATCH --mail-type=END

source activate fdtd

xvfb-run --server-args="-screen 0 1280x1024x24" python LumProcSweep.py 10 > stdout_lum.log 2> stderr_lum.log

exit $?
