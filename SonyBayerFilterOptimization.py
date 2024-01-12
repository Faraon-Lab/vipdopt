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
from vipdopt.optimization import Optimization, BayerFilterFoM, AdamOptimizer, FOM_ZERO
from vipdopt.optimization.device import Device


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
    logging.basicConfig(level=level)
    logger = setup_logger(level)

    #
    # Step 0
    #
    logger.info('Beginnning Step 0: Lumerical Setup')

    # Load configuration file
    cfg = SonyBayerConfig()
    cfg.read_file(args.config)

    # Load Simulations...
    logger.info(f'Loading base simulation from {args.simulation_json}...')
    base_sim = LumericalSimulation(args.simulation_json)
    logger.info('...succesfully loaded base simulation!')
    base_sim.setup_hpc_resources()

    # Create all different versions of simulation that are run
    base_sim.disable_all_sources()
    fwd_srcs = [f'forward_src_{d}' for d in 'xy']
    adj_srcs = [f'adj_src_{n}{d}' for n in range(cfg['num_adjoint_sources']) for d in 'xy']

    logger.info('Creating forward simulations...')
    fwd_sims = [base_sim.enabled([src]) for src in fwd_srcs]

    logger.info('Creating adjoint simulations...')
    adj_sims = [base_sim.enabled([src]) for src in adj_srcs]

    all_sims = fwd_sims + adj_sims

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
    logger.info('Completed Step 1: Design Region Setup')

    #
    # Step 2:
    #
    logger.info('Beginning Step 2: Optimization Setup')

    # Create sources and figures of merit (FoMs)
    fom_srcs = [(Source(fwd_sims[i], f'focal_monitor_{i}'),) for i in range(2)]

    grad_fwd_srcs = [Source(sim, f'design_efield_monitor') for sim in fwd_sims]
    grad_adj_srcs = [Source(sim, f'design_efield_monitor') for sim in adj_sims]
    source_pairs = list(zip([0, 1] * 4, range(8)))  # [(0, 0), (1, 1), (0, 2), (1, 3), ...]
    grad_srcs = [(grad_fwd_srcs[i], grad_adj_srcs[j]) for i, j in source_pairs]

    foms = [
        BayerFilterFoM(
            fom_srcs[i],
            grad_srcs[j],
            'TE',
            cfg['lambda_values_um'],
        ) for i, j in source_pairs
    ]

    # Add weighted FoMs
    start = FOM_ZERO
    start.polarization = 'TE'
    start.freq = cfg['lambda_values_um']
    start.opt_ids = list(range(len(start.freq)))

    weights = np.ones(len(foms))
    full_fom = sum(np.multiply(weights, foms), start)

    # Create optimizer
    adam_moments = np.zeros((2,) + bayer_filter.size)

    optimizer = AdamOptimizer(
        step_size=cfg['fixed_step_size'],
        betas=cfg['adam_betas'],
        moments=adam_moments,
    )

    # Create optimization
    # max_epochs = cfg['num_epochs']
    # max_iter = cfg['num_iterations_per_epoch'] * max_epochs
    max_epochs = 1
    max_iter = 2

    optimization = Optimization(
        all_sims,
        bayer_filter,
        full_fom,
        optimizer,
        max_iter,
        max_epochs,
        start_iter=0,
        start_epoch=0,
    )

    # TODO: Add reinterpolation of the device

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
    for i in range(len(all_sims)):
        all_sims[i].save(f'sim{i}_after_opt')
        all_sims[i].close()

    