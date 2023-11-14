"""Entry point for MPI workers."""

from mpi4py import MPI
import subprocess
import sys
from vipdopt.pool import ServerWorker

if __name__ == '__main__':

    comm_world = MPI.COMM_WORLD
    comm_parent = MPI.Comm.Get_parent()
    if comm_parent == MPI.COMM_NULL:
        exit(1)

    size = comm_world.Get_size()
    if size == 1:
        s = ServerWorker()
    else:
        # Need to run some MPI program
        rank = comm_world.Get_rank()
        status = MPI.Status
        if rank == 0:
            comm_parent.recv(None, 0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
        
        comm_world.Barrier()

        print(sys.argv)

        if len(sys.argv) < 2:
            exit(1)
        program = sys.argv[1:]

        # Call the other MPI program and sum the error codes
        res = subprocess.call(program)
        combined_res = comm_world.reduce(res, op=MPI.SUM, root=0)

        if rank == 0:
            comm_parent.send(int(combined_res > 0), 0, tag=tag)
        
        exit(0)
