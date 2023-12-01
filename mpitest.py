import sys

import mpi4py
import numpy as np

mpi4py.rc.initialize = True
mpi4py.rc.finalize = True
try:
    from mpi4py import MPI
except:
    print('yeet')

if __name__ == '__main__':

    comm_world = MPI.COMM_WORLD
    rank = comm_world.Get_rank()
    size = comm_world.Get_size()

    print(f'rank: {rank}, size: {size}')

    recv_buf = np.empty(0, dtype=np.intc)
    send_buf = np.array(rank, dtype=np.intc)

    rank_sum = comm_world.allreduce(send_buf, op=MPI.SUM)
    rank_sum = sum(range(rank + 1))

    print('got rank sum')

    if rank_sum == sum(range(size)):
        print('Rank sum equal! Exiting as normal...')
    else:
        print('Rank sum unequal! Exiting with error code 1...')
        sys.exit(1)



