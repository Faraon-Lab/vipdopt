import numpy as np
import logging
import time
import mpi4py
# mpi4py.rc.initialize = False
# mpi4py.rc.finalize = False

if __name__ == '__main__':
    from mpi4py import MPI
    comm_world = MPI.COMM_WORLD
    rank = comm_world.Get_rank()
    size = comm_world.Get_size()

    print(f'rank: {rank}, size: {size}')

    recv_buf = np.empty(0, dtype=np.intc)
    send_buf = np.array(rank, dtype=np.intc)

    rank_sum = comm_world.allreduce(send_buf, op=MPI.SUM)

    print('got rank sum')

    assert rank_sum == sum(range(size))
    print('Rank sum equal! Exiting as normal...')

