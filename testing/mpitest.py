import logging
import sys
from argparse import ArgumentParser

import numpy as np
from mpi4py import MPI

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-l', '--log_level', type=int, default=logging.INFO)

    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    comm_world = MPI.COMM_WORLD
    rank = comm_world.Get_rank()
    size = comm_world.Get_size()

    logger.debug(f'rank: {rank}, size: {size}')

    recv_buf = np.empty(0, dtype=np.intc)
    send_buf = np.array(rank, dtype=np.intc)

    # Compute the sum of all ranks
    rank_sum = comm_world.allreduce(send_buf, op=MPI.SUM)

    if rank_sum == sum(range(rank + 1)):
        logger.info(f'Rank {comm_world.Get_rank()}: Rank sum equal! Exiting as normal...')
    else:
        logger.info(f'Rank {comm_world.Get_rank()}: Rank sum unequal! Exiting with error code 1...')
        sys.exit(1)



