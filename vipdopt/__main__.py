"""Volumetric Inverse Photonic Design Optimizer."""
import logging
import os
import sys
from argparse import ArgumentParser

from mpi4py import MPI

f = sys.modules[__name__].__file__
if not f:
    raise ModuleNotFoundError('SHOULD NEVER REACH HERE')

path = os.path.dirname(f)
path = os.path.join(path, '..')
sys.path.insert(0, path)


from vipdopt.configuration.config import Config

ROOT = 0  # Index of root process


if __name__ == '__main__':
    parser = ArgumentParser(
        prog='vipdopt',
        description='Volumetric Inverse Photonic Design Optimizer',
    )
    subparsers = parser.add_subparsers()
    opt_parser = subparsers.add_parser('optimize')
    eval_parser = subparsers.add_parser('evaluate')

    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='Enable verbose output.')

    args = parser.parse_args()

    # Set up MPI
    comm_world = MPI.COMM_WORLD
    my_rank = comm_world.Get_rank()

    # Set verbosity
    levels = [logging.INFO, logging.WARNING, logging.DEBUG]
    level = levels[min(args.verbose, len(levels) - 1)]  # cap to last level index
    logging.basicConfig(level=level, format='%(message)s')

    # Load configuration file
    if my_rank == ROOT:
        cfg = Config()
        cfg.read_file('config.yml.example')


    # Load Simulations...

    # Setup Simulation environment and connect with Lumerical API...

    # with MPIPool(comm_world) as pool:
        # MPIPool allows distributing work between available processes using MPI
        # if pool.ismanager():

            # While optimize:
                # Run simulations in parallel
                # Combine results from simulations
                # Compute gradient & run optimizer
                # Update simulation values

    # Extract final data from optimization
    # Evaluate device and generate plots (could also be parallelized if needed)


