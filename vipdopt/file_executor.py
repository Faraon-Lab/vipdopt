#!/usr/bin/env python
"""Script to wait for child processes to finish."""

import logging
import sys
import subprocess
import numpy as np


if __name__ == '__main__':
    from mpi4py import MPI
    logging.basicConfig(level=logging.DEBUG)
    parent = MPI.Comm.Get_parent()

    # logging.debug(f'Parent communicator: {parent}')
    # logging.debug(f'Worker {parent.Get_rank()}: Total size: {parent.Get_size() + parent.Get_remote_size()}...')
    
    # logging.debug(f'Worker {parent.Get_rank()}: Passed arguments: {sys.argv}')
    # comm_world = MPI.COMM_WORLD

    err_code = subprocess.call(sys.argv[1:])
    logging.debug(f'Worker {parent.Get_rank()}: errcode={err_code}')

    logging.debug(f'Worker {parent.Get_rank()}: Starting barrier...')
    request = parent.Ibarrier()
    request.wait()
    logging.debug(f'Worker {parent.Get_rank()}: barrier complete...')

    parent.gather(err_code, root=0)
    logging.debug(f'Worker {parent.Get_rank()}: gather complete...')

