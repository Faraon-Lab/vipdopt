"""Entry point for MPI workers."""

from mpi4py import MPI
from vipdopt.pool import FunctionWorker

if __name__ == '__main__':
    comm_parent = MPI.Comm.Get_parent()
    if comm_parent == MPI.COMM_NULL:
        exit(1)

    s = FunctionWorker()
