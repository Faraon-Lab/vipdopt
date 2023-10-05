#!/usr/bin/env python3
"""Code for managing the program's environemnt"""

import sys
import os
from typing import Any, Callable
import logging

from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
import numpy as np


SLURM_ENV_VARS = (
    'SLURM_JOB_ID',
    'SLURM_SUBMIT_DIR',
    'SLURM_CPUS_ON_NODE',
    'SLURM_JOB_NAME',
    'SLURM_JOB_NODELIST',
    'SLURM_JOB_NUMNODES',
    'SLURM_NPROCS'
)

VALID_JOB_STATES =  ['', 'all', 'pending', 'completed']


class Job(object):
    """Class for subprocess-related code."""

    def __init__(self, name: str, f: Callable, f_args: tuple) -> None:
        self.name = name
        self.f = f
        self.f_args = f_args

        self.complete = False
    
    def __str__(self) -> str:
        return f'Job: \'{self.name}\''
    
    def execute(self) -> Any:
        res = self.f(*self.f_args)
        self.complete = True
        logging.debug(f'Completed {self}. Returned result: {res}.')
        return res


# TODO: Run it normally (python manager.py)
# Make it create a slurm file that runs all of the jobs and asks for the 
# appropriate amount of resources ? Then run_jobs is opening a subprocess and running
# sbatch <slurm_filename>. For example:
"""
def run_jobs_inner( queue_in ):
	processes = []
	job_idx = 0
	while not queue_in.empty():
		get_job_path = queue_in.get()

		logging.info(f'Node {cluster_hostnames[job_idx]} is processing file at {get_job_path}.')

		process = subprocess.Popen(
			[
				'/home/gdrobert/Develompent/adjoint_lumerical/inverse_design/run_proc.sh',
				cluster_hostnames[ job_idx ],
				get_job_path
			]
		)
		processes.append( process )
"""
# where run_proc.sh contains:
"""/central/home/ifoo/lumerical/2021a_r22/mpich2/nemesis/bin/mpiexec -verbose -n 8 -host $1 /central/home/ifoo/lumerical/2021a_r22/bin//fdtd-engine-mpich2nem -t 1 $2 > /dev/null 2> /dev/null"""
# ? Maybe this is what srun does already ?

class WorkloadManager(object):
    """Class for managing jobs and resources needed for parallelized code."""

    def __init__(self):
        self.__dict__.update({var:os.getenv(var) for var in SLURM_ENV_VARS})
        
        self.jobs: list[Job] = []
        self.pending_jobs: list[Job] = []
        self.completed_jobs: list [Job] = []
    
    def __str__(self) -> str:
        return f'WorkloadManager with {({var:self.__getattribute__(var) for var in SLURM_ENV_VARS})}'
    
    def num_jobs(self, state: str='pending') -> int:
        state = state.lower()
        match state:
            case '' | 'all':
                return len(self.jobs)
            case 'pending':
                return len(self.pending_jobs)
            case 'completed':
                return len(self.completed_jobs)
            case _:
                raise ValueError(f'Expected state in {VALID_JOB_STATES}, got {state}.')

    def add_job(self, job: Job):
        self.jobs.append(job)
        self.pending_jobs.append(job)
    
    def run_jobs(self):
        comm_world = MPI.COMM_WORLD
        size = comm_world.Get_size()
        logging.debug(f'Executing {self.num_jobs()} jobs with comm_world size of {size}')
        my_rank = comm_world.Get_rank()
        
        # TODO: Doesn't work for when there are more jobs than processes
        # If we don't need this process, don't bother
        if my_rank >= self.num_jobs():
            return
        logging.debug(f'PE{my_rank}:\tstarting job...')
        
        my_job = self.pending_jobs[my_rank]
        my_job.execute()
        logging.debug(f'PE{my_rank}:\t{my_job} Complete')

        self.clean_jobs()
    
    def clean_jobs(self):
        self.completed_jobs = self.completed_jobs + \
            [j for j in self.pending_jobs if j.complete]
        self.pending_jobs = [j for j in self.pending_jobs if not j.complete]
    

class Worker(object):

    def __init__(self, comm_world, rank_world, manager: WorkloadManager) -> None:
        self.comm_world = comm_world
        self.rank_world = rank_world
        self.manager = manager
    
    def _idle(self):
        pass
    

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')

    comm_world = MPI.COMM_WORLD
    size = comm_world.Get_size()
    my_rank = comm_world.Get_rank()

    if my_rank == 0:
        logging.debug(f'\nComputing 2^n for all n in [1, ..., 1000] with {size} proceess{(size > 1) * "es"}\n')

    with MPIPoolExecutor(max_workers=size, main=False) as executor:
        for res in executor.starmap(pow, [(2, n) for n in range(1000)]):
            logging.info(res)


    # # Create work queues
    # work_input_buffer = np.empty()

    # if my_rank == 0:  # root process
    #     man = WorkloadManager()
    #     logging.debug(f'Created workload manager in process {my_rank}/{size}:\n{man}')

    # #logging.debug(f'Executing {self.num_jobs()} jobs with comm_world size of {size}')





    # man = WorkloadManager()
    # logging.debug(man)

    # man.run_jobs()

