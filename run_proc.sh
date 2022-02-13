#!/bin/bash


/central/home/gdrobert/Develompent/lumerical/v212/mpich2/nemesis/bin/mpiexec -verbose -n 8 -host $1 /central/home/gdrobert/Develompent/lumerical/v212/bin//fdtd-engine-mpich2nem -t 1 $2 > /dev/null 2> /dev/null


exit $?

