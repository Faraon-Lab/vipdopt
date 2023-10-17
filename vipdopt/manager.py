"""Workload manager using MPI to create a job pool."""
import atexit
import logging
import os
import sys
from collections.abc import Callable, Sequence
from typing import Any

from mpi4py import MPI

if __name__ == '__main__':
    #  Here so that running this script directly doesn't blow up
    f = sys.modules[__name__].__file__
    if not f:
        raise ModuleNotFoundError('SHOULD NEVER REACH HERE')

    path = os.path.dirname(f)
    path = os.path.join(path, '..')
    sys.path.insert(0, path)

from vipdopt.utils import R

ROOT = 0

# TODO: Allow multiple processes per job
class MPIPool:
    """A Job Pool for use with MPI.

    Splits into a manager/worker dynamic. The manager process is indicated by
    rank == ROOT (by default ROOT = 0). When the manager receives jobs it assigns them
    to available worker processes. The workers wait for jobs and send results back to
    the manager.

    Attributes:
        comm (MPI.Comm): The MPI communicator containing the manager and
            worker processes
        size: The total number of processes
        rank (int): The rank of this particular process.
        workers (set[int]): All the ranks of worker processes
    """

    def __init__(self, comm: MPI.Comm | None=None) -> None:
        """Create MPIPool object and start manager / worker relationship."""
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
            raise ValueError('More than one process required to create MPIPool.')

    def wait(self):
        """If a worker, wait for work to be provided."""
        if self.is_manager():
            return

        status = MPI.Status()
        while True:
            # Blocking receive
            work = self.comm.recv(source=ROOT, tag=MPI.ANY_TAG, status=status)

            if work is None:  # None is the signal to stop
                logging.debug(f'Worker {self.rank} told to stop')
                return

            func, args = work
            res = func(args)
            self.comm.ssend(res, ROOT, status.tag)

    def is_manager(self):
        """Return whether this process is the manager."""
        return self.rank == ROOT

    def is_worker(self):
        """Return whether this process is a worker."""
        return self.rank != ROOT

    def submit(self, func: Callable[..., R | None], task: Any) -> R | None:
        """Submit a single job to the pool."""
        return self.map(func, [task])[0]

    def map(self, # noqa: A003
            func: Callable[..., R | None],
            tasks: Sequence[Any]
        ) -> Sequence[R | None]:
        """Compute `func` on a list of inputs in parallel.

        The output order will always be the same as input. This method is equivalent to
        [self.submit(func, task) for task in tasks].

        Arguments:
            func (Callable[..., R]): The function to compute.
            tasks (Sequence[Any]): The list of arguments to pass to the function. Type
                of individual tasks should match the signature of `func`.


        Returns:
            (Sequence[R]) A list of `func(task)` for all the tasks provided.
        """
        results: list[R | None] = [None] * len(tasks)

        # Workers should be waiting for jobs
        if self.is_worker():
            self.wait()
            return results

        # Initialize jobs and output list
        available_workers = self.workers.copy()
        jobs = [(idx, (func, arg)) for idx, arg in enumerate(tasks)]
        logging.debug(f'Jobs: {jobs}\n')

        pending = len(jobs)

        logging.debug(f'Available workers: {available_workers}\n')

        while pending > 0:
            # If a worker is free and jobs need to be assigned still...
            if available_workers and len(jobs) > 0:
                worker = available_workers.pop()  # This worker is busy now
                idx, job = jobs.pop()

                # Blocking send
                self.comm.ssend(job, dest=worker, tag=idx)
                logging.debug(f'Sent job {job} to worker {worker} with tag {idx}')

            if len(jobs) > 0:
                # Check if any workers are sending results
                flag = self.comm.iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)

                # If there are more jobs to do and no results yet,
                # we can keep assigning them while we wait for results
                if not flag:
                    continue

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
        """Send the stop signal (None) to all processes."""
        if self.is_worker():
            return

        for worker in self.workers:
            self.comm.isend(None, worker)

    def __enter__(self):
        """Yield the MPIPool for context manager usage."""
        return self

    def __exit__(self, *args):
        """Close the MPIPool for context manager usage."""
        self.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    N = int(1e3)

    comm_world = MPI.COMM_WORLD
    size = comm_world.Get_size()
    my_rank = comm_world.Get_rank()

    def square(n: int) -> int:
        """Compute square of a number."""
        return n ** 2

    if my_rank == ROOT:
        logging.debug(f'\nComputing n^2 for all n in [1, ..., {N}] using'
                      f' {size} proceess{(size> 1) * "es"}\n')

    with MPIPool(comm_world) as pool:
        if pool.is_manager():
            for i, res in enumerate(pool.map(square, range(N))):
                if i % (N / 10) == 0:
                    logging.info(f'{i}^2\t=\t{res:e}')
