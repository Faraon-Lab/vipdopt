"""Optimization sweep for a Bayer filter, in collaboration with SONY"""
import logging
import os
import sys
from argparse import ArgumentParser

from vipdopt.simulation import LumericalSimulation
from vipdopt.utils import setup_logger

f = sys.modules[__name__].__file__
if not f:
    raise ModuleNotFoundError('SHOULD NEVER REACH HERE')

path = os.path.dirname(f)
path = os.path.join(path, '..')
sys.path.insert(0, path)


import numpy as np

from vipdopt.optimization.filter import Sigmoid
from vipdopt.source import Source
from vipdopt.configuration import SonyBayerConfig
from vipdopt.optimization import Optimization, BayerFilterFoM, AdamOptimizer
from vipdopt.optimization.device import Device

ROOT = 0  # Index of root process


if __name__ == '__main__':
    parser = ArgumentParser(
        prog='vipdopt',
        description='Volumetric Inverse Photonic Design Optimizer',
    )
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Enable verbose output.')

    subparsers = parser.add_subparsers()
    opt_parser = subparsers.add_parser('optimize')

    opt_parser.add_argument(
        'config',
        type=str,
        help='Configuration file to use in the optimization',
    )

    opt_parser.add_argument(
        'simulation_json',
        type=str,
        help='Simulation JSON to create a simulation from',
    )

    args = parser.parse_args()

    # Set verbosity
    level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger(level)

    #
    # Step 0
    #
    logger.info('Beginnning Step 0: Lumerical Setup')

    # Load configuration file
    cfg = SonyBayerConfig()
    cfg.read_file(args.config)

    # TODO: Need a way to enable / disable sources mid optimization
    # Load Simulations...
    logger.info(f'Loading simulation from {args.simulation_json}...')
    sim = LumericalSimulation(args.simulation_json)
    logger.info('...succesfully loaded simulation!')
    sim.setup_hpc_resources()

    logger.info('Completed Step 0: Lumerical Setup')

    #
    # Step 1
    #
    logger.info('Beginning Step 1: Design Region Setup')

    bayer_filter = Device(
        (cfg['device_voxels_simulation_mesh_vertical'], cfg['device_voxels_simulation_mesh_lateral']),
        (cfg['min_device_permittivity'], cfg['max_device_permittivity']),
        (0, 0, 0),
        randomize=True,
        init_seed=0,
        filters=[Sigmoid(0.5, 1.0), Sigmoid(0.5, 1.0)],
    )
    # TODO: Add reinterpolation of the device
    logger.info('Completed Step 1: Design Region Setup')

    #
    # Step 2:
    #
    # TODO: FIX THIS to use fom_srcs and grad_srcs, and to get the good stuff
    logger.info('Beginning Step 2: Optimization Setup')

    # Create sources and figures of merit (FoMs)
    fwd_srcs = [Source(sim, f'transmission_monitor_{i}') for i in range(2)]
    adj_srcs = [Source(sim, f'focal_monitor_{i}') for i in range(4)]
    foms = [
        BayerFilterFoM(
            [fwd_srcs[i]],
            [adj_srcs[j]],
            'TE',
            cfg['lambda_values_um'],
        ) for i, j in zip(range(2), range(4))
    ]
    weights = np.ones(len(foms))
    full_fom = sum(np.multiply(weights, foms))

    # Create optimizer
    adam_moments = np.zeros((2,) + bayer_filter.size)

    optimizer = AdamOptimizer(
        step_size=cfg['fixed_step_size'],
        betas=cfg['adam_betas'],
        moments=adam_moments,
        fom_func=lambda foms: sum(f[1] for f in map(BayerFilterFoM.compute, foms)),
        grad_func=lambda foms: sum(map(BayerFilterFoM.gradient, foms)),
    )

    # Create optimization
    max_epochs = cfg['num_epochs']
    max_iter = cfg['num_iterations_per_epoch'] * max_epochs

    optimization = Optimization(
        sim,
        bayer_filter,
        full_fom,
        optimizer,
        max_iter,
        max_epochs,
        start_iter=0,
        start_epoch=0,
    )

    logger.info('Completed Step 2: Optimization Setup')

    #
    # Step 3
    #
    logger.info('Beginning Step 3: Run Optimization')

    optimization.run()

    logger.info('Completed Step 3: Run Optimization')

    #
    # Cleanup
    #
    sim.save('test_sim_after_opt')
    sim.close()

    