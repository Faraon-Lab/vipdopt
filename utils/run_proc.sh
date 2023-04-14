#!/bin/bash

/central/home/ifoo/lumerical/2021a_r22/mpich2/nemesis/bin/mpiexec -verbose -n 8 -host $1 /central/home/ifoo/lumerical/2021a_r22/bin//fdtd-engine-mpich2nem -t 1 $2 > /dev/null 2> /dev/null

exit $?

