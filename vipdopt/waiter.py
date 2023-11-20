#!/usr/bin/env python
"""Script to wait for child processes to finish."""

import logging
import sys
import numpy as np

if __name__ == '__main__':
    from mpi4py import MPI
    logging.basicConfig(level=logging.DEBUG)
    parent = MPI.Comm.Get_parent()
    logging.debug(f'Parent communicator: {parent}')
    logging.debug(sys.argv)
    err_code = int(sys.argv[1])
    comm_world = MPI.COMM_WORLD

    logging.debug(f'Worker {parent.Get_rank()}: errcode={err_code}')

    logging.debug(f'Worker {parent.Get_rank()}: Starting barrier...')
    logging.debug(f'Worker {parent.Get_rank()}: Total size: {parent.Get_size() + parent.Get_remote_size()}...')

    if parent != MPI.COMM_NULL:
        request = parent.Ibarrier()
        request.wait()
        logging.debug(f'Worker {parent.Get_rank()}: barrier complete...')
        parent.gather(err_code, root=0)
        logging.debug(f'Worker {parent.Get_rank()}: gather complete...')
