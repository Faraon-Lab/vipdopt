"""Code for managing the program's environemnt"""

import sys
import os

from mpi4py import MPI


SLURM_JOB_NODELIST = 'SLURM_JOB_NODELIST'

class WorkloadManager(object):
    """Class for managing jobs and resources needed for parallelized code."""

    def __init__():
        WorkloadManager.get_slurm_nodes()
    
    @staticmethod
    def get_slurm_nodes():
        nodelist = os.getenv(SLURM_JOB_NODELIST)
        return nodelist


