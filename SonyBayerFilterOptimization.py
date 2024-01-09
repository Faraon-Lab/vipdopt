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

from vipdopt.filter import Sigmoid
from vipdopt.source import Source
from vipdopt.configuration import SonyBayerConfig
from vipdopt.optimization import Optimization, BayerFilterFoM, AdamOptimizer
from vipdopt.device import Device

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
    # eval_parser = subparsers.add_parser('evaluate')

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

    # opt_args = opt_parser.parse_args()

    logger.info('Beginnning Step 0: Lumerical Setup')
    # Load configuration file
    cfg = SonyBayerConfig()
    cfg.read_file(args.config)


    logger.info(f'Loading simulation from {args.simulation_json}...')
    # Load Simulations...
    sim = LumericalSimulation(args.simulation_json)
    logger.info('...succesfully loaded simulation!')
    sim.setup_hpc_resources()

    logger.info('Completed Step 0: Lumerical Setup')

    logger.info('Beginning Step 1: Design Region Setup')

    bayer_filter = Device(
        (cfg['device_voxels_simulation_mesh_vertical'], cfg['device_voxels_simulation_mesh_lateral']),
        (cfg['min_device_permittivity'], cfg['max_device_permittivity']),
        (0, 0, 0),
        randomize=True,
        init_seed=0,
        filters=[Sigmoid(0.5, 1.0), Sigmoid(0.5, 1.0)],
    )
    logger.info('Completed Step 1: Design Region Setup')

    logger.info('Beginning Step 2: Optimization Setup')

    # Create sources and figures of merit (FoMs)
    fwd_srcs = [Source(sim, f'forward_monitor_{i}') for i in range(2)]
    adj_srcs = [Source(sim, f'focal_monitor_{i}') for i in range(4)]
    foms = [
        BayerFilterFoM(
            [fwd_srcs[i]],
            [adj_srcs[j]],
            'TE',
            cfg['lambda_values_um'],
        ) for i, j in zip(range(2), range(4))
    ]


    adam_moments = np.zeros((2,) + bayer_filter.size)
    optimizer = AdamOptimizer(
        adam_moments,
        cfg['fixed_step_size'],
        cfg['adam_betas'],
        fom_func=lambda foms: sum(f[1] for f in map(BayerFilterFoM.compute, foms)),
        grad_func=lambda foms: sum(map(BayerFilterFoM.gradient, foms)),
    )
    optimization = Optimization(
        sim,
        foms,
        bayer_filter,
        optimizer,
    )

    logger.info('Completed Step 2: Optimization Setup')

    logger.info('Beginning Step 3: Run Optimization')

    optimization.run(cfg['num_iterations_per_epoch'] * cfg['num_epochs'], cfg['num_epochs'])

    logger.info('Completed Step 3: Run Optimization')

    sim.save('test_sim_after_opt')
    sim.close()

    