"""Entry point for MPI workers."""

import sys

from mpi4py import MPI

from vipdopt.pool import FunctionWorker

if __name__ == '__main__':
    comm_parent = MPI.Comm.Get_parent()
    if comm_parent == MPI.COMM_NULL:
        sys.exit(1)

    s = FunctionWorker()
