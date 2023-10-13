import os
import sys
import atexit
import logging
import socket
from typing import Any, Callable, Iterable


from mpi4py import MPI
import numpy as np

if __name__ == '__main__':
    #  Here so that running this script directly doesn't blow up
    f = sys.modules[__name__].__file__
    if not f:
        raise ModuleNotFoundError('SHOULD NEVER REACH HERE')

    path = os.path.dirname(f)
    path = os.path.join(path, '..')
    sys.path.insert(0, path)

from vipdopt.utils import T, R

ROOT = 0

SLURM_ENV_VARS = (
    'SLURM_JOB_ID',
    'SLURM_SUBMIT_DIR',
    'SLURM_CPUS_ON_NODE',
    'SLURM_JOB_NAME',
    'SLURM_JOB_NODELIST',
    'SLURM_JOB_NUM_NODES',
    'SLURM_NPROCS'
) 


class MPIPool(object):

    def __init__(self, comm=None) -> None:
        if comm is None:
            comm = MPI.COMM_WORLD
        self.comm = comm

        self.rank = self.comm.Get_rank()

        atexit.register(MPIPool.close, self)

        if self.is_worker():
            self.wait()
            return
        
        comm_size = self.comm.Get_size()

        self.workers = set(range(comm_size))
        self.workers.discard(ROOT)
        self.size = comm_size - 1

        if self.size == 0:
            raise ValueError("More than one process required to create MPIPool.")

    def wait(self):
        """If a worker, wait for work to be provided."""
        if self.is_manager():
            return

        # logging.debug(f'Worker {self.rank} waiting for work...')
        
        status = MPI.Status()
        while True:

            # Blocking receive
            work = self.comm.recv(source=ROOT, tag=MPI.ANY_TAG, status=status)

            if work is None:  # None is the signal to stop
                logging.debug(f'Worker {self.rank} told to stop')
                return

            func, args = work

            # logging.debug(f'Worker {self.rank} received work {work} with tag {status.tag}...')

            res = func(args)
            # logging.debug(f'Worker {self.rank} sending result {res}...')

            self.comm.ssend(res, ROOT, status.tag)

    def is_manager(self):
        return self.rank == ROOT
    
    def is_worker(self):
        return self.rank != ROOT
    
    def submit(self, func, task):
        return self.map(func, [task])[0]
    
    def map(self, func: Callable[..., R], tasks: Iterable[Any]) -> Iterable[R]:
        if self.is_worker():
            self.wait()
            return

        available_workers = self.workers.copy()
        jobs = [(idx, (func, arg)) for idx, arg in enumerate(tasks)]
        logging.debug(f'Jobs: {jobs}\n')
        results = [None] * len(jobs)
        pending = len(jobs)

        logging.debug(f'Available workers: {available_workers}\n')

        while pending > 0:
            if available_workers and len(jobs) > 0:
                worker = available_workers.pop()  # This worker is busy now
                idx, job = jobs.pop()

                # logging.debug(f'Sending job {job} with index {idx}')
                # logging.debug(f'Anticipated result: {job[0](job[1])}\n')

                # Non-blocking send
                self.comm.send(job, dest=worker, tag=idx)
                logging.debug(f'Sent job {job} to worker {worker} with tag {idx}')
            
            if len(jobs) > 0:
                # check if any workers are sending results
                flag = self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)

                # If there are more jobs to do and no results yet,
                # we can keep assigning them while we wait for results
                if not flag:
                    continue
            # else:
            #     self.comm.Probe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
            
            status = MPI.Status()
            # Blocking receive
            res = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)

            worker = status.source
            idx = status.tag
            logging.debug(f'Received result from worker {worker} with tag {idx}')

            results[idx] = res

            available_workers.add(worker)  # Allow worker to be assigned work
            pending -= 1
        
        return results

    def close(self):
        if self.is_worker():
            return
        
        # Send 'None' to each worker to signal to stop the wait loop
        for worker in self.workers:
            self.comm.isend(None, worker)
        
    def __enter__(self):
        return self
    
    # TODO: May need to handle exceptions, cause it's just hanging
    def __exit__(self, *args):
        return self.close()


def test_work(n):
    # time.sleep(2 * n / N)
    return n ** 2

def test_multiproc(pow):
    comm_world = MPI.COMM_WORLD
    #size = comm_world.Get_size()
    my_rank = comm_world.Get_rank()
    data = np.array(np.power(pow, my_rank), dtype=np.intc)

    sum_buf = np.empty(1, dtype=np.intc)

    comm_world.allreduce(data, sum_buf)
    return sum_buf


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    N = int(1e6)

    comm_world = MPI.COMM_WORLD
    size = comm_world.Get_size()
    my_rank = comm_world.Get_rank()

    if my_rank == ROOT:
        logging.debug(f'MPI size:\t{size}')


        slurm_env_vars = {var:os.getenv(var) for var in SLURM_ENV_VARS}
        logging.debug(slurm_env_vars)
        nnodes = int(slurm_env_vars['SLURM_JOB_NUM_NODES'])
        nprocs_per_node = int(slurm_env_vars['SLURM_CPUS_ON_NODE'])
        nprocs = nnodes * nprocs_per_node

        logging.debug(f'\nComputing n^2 for all n in [1, ..., {N}] with {nprocs} proceess{(nprocs > 1) * "es"} across {nnodes} nodes\n')
    
    with MPIPool(comm_world) as pool:
        if pool.is_manager():
            for i, res in enumerate(pool.map(test_work, range(N))):
                if i % (N / 10) == 0:
                    logging.info(f'{i}^2\t=\t{res:e}')
