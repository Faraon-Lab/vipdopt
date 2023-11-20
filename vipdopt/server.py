"""Entry point for MPI workers."""

from mpi4py import MPI
import sys
from vipdopt.pool import ServerWorker
import logging

if __name__ == '__main__':
    comm_parent = MPI.Comm.Get_parent()
    if comm_parent == MPI.COMM_NULL:
        exit(1)

    s = ServerWorker()

    # if sys.argv[1] == '0':
    #     s = ServerWorker()
    # else:
    #     logging.debug(sys.argv)
    #     # Need to run some MPI program
    #     comm_world = MPI.COMM_WORLD
    #     size = comm_world.Get_size()
    #     rank = comm_world.Get_rank()
    #     status = MPI.Status()

    #     mpi_info = comm_parent.recv(None, 0, tag=MPI.ANY_TAG, status=status)
    #     tag = status.Get_tag()

    #     info = MPI.Info.Create()
    #     info.update(mpi_info)
        
    #     if len(sys.argv) < 3:
    #         exit(1)
    #     command = sys.argv[2]
    #     args = sys.argv[3:] if len(sys.argv) > 3 else None

    #     comm_world.Barrier()


    #     # Call the other MPI program and sum the error codes
    #     try:
    #         logging.debug(f'Worker {rank}: Running {command} {args}...')
    #         intercomm = comm_world.Spawn(command, args, maxprocs=size, info=info)
    #         err = 0
    #     except BaseException as e:
    #         logging.exception()
    #         err = 1

    #     logging.debug(f'Worker {rank} done computing MPI file!')
    #     comm_world.Barrier()

    #     combined_err = comm_world.reduce(err, op=MPI.SUM, root=0)
    #     logging.debug(f'Worker {rank}: Done reducing result!')

    #     if rank == 0:
    #         comm_parent.send(combined_err > 0, 0, tag=tag)
        
    #     comm_world.Barrier()
        
    #     exit(0)
