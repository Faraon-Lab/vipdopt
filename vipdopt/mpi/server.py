"""Entry point for MPI workers."""

if __name__ == '__main__':
    import logging
    import sys

    from mpi4py import MPI

    from vipdopt.mpi.pool import FunctionWorker

    log_level = int(sys.argv[1])

    logging.basicConfig(level=log_level)

    comm_parent = MPI.Comm.Get_parent()
    if comm_parent == MPI.COMM_NULL:
        sys.exit(1)

    s = FunctionWorker()
