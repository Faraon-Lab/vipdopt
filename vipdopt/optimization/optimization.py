"""Class for handling optimization setup, running, and saving."""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy.typing as npt

from vipdopt.optimization.device import Device
from vipdopt.optimization.optimizer import GradientOptimizer
from vipdopt.optimization.fom import FoM, BayerFilterFoM
from vipdopt.optimization.adam import AdamOptimizer, DEFAULT_ADAM
from vipdopt.simulation import ISimulation, LumericalSimulation
from vipdopt.configuration import Config
from vipdopt.utils import ensure_path
from vipdopt.monitor import Monitor


class Optimization:
    """Class for orchestrating all the pieces of an optimization."""

    def __init__(
            self,
            sims: list[ISimulation],
            device: Device,
            config: Config,
            fom: FoM,
            optimizer: GradientOptimizer,
            max_iter: int,
            max_epochs: int,
            start_iter: int=0,
            start_epoch: int=0,
    ):
        """Initialize Optimzation object."""
        self.sims = sims
        self.fom = fom
        self.device = device

        self.fom_hist: list[npt.NDArray] = []
        self.param_hist: list[npt.NDArray] = []
        self._callbacks: list[Callable[[Optimization], None]] = []

        self.epoch = 0
        self.iteration = 0

        self._load_config(config)
        self.config = config

    def _load_config(self, config: Config):
        """Load and setup optimization from an appropriate JSON config file."""
        # Setup optimizer
        optimizer: str = config.get('optimizer', 'adam')
        optimizer_settings: dict = config.get('optimizer_settings', DEFAULT_ADAM)
        match optimizer.lower():
            case 'adam':
                optimizer_type = AdamOptimizer
            case _:
                raise NotImplementedError(f'Optimizer {optimizer} not currently supported')
        self.optimizer = optimizer_type(**optimizer_settings)

        # Setup simulations
        base_sim = LumericalSimulation(config['base_simulation'])
        base_sim.setup_env_resources(
            mpi_exe=config['mpi_path'],
            nprocs=config.get('nprocs'),
            solver_exe=config['solver_path']
        )
        src_to_sim_map = {src: base_sim.with_enabled([src]) for src in base_sim.sources()}
        self.sims = src_to_sim_map.values()

        # Setup FoMs
        foms: list[FoM] = []
        weights = []
        fom: dict
        for fom in config.get('figures_of_merit', []):
            match fom['type']:
                case 'bayer_filter':
                    fom_cls = BayerFilterFoM
                case _:
                    raise NotImplementedError
            foms.append(new_fom = fom_cls(
                    fom_monitors=[
                        Monitor(src_to_sim_map[src], mname) for src, mname in fom['fom_monitors']
                    ],
                    grad_monitors=[
                        Monitor(src_to_sim_map[src], mname) for src, mname in fom['grad_monitors']
                    ],
                    polarization=fom['polarization'],
                    freq=fom['freq'],
                    opt_ids=fom.get('opt_ids', None),
                ) 
            )
            weights.append(fom['weight'])
        self.foms = foms

    def add_callback(self, func: Callable):
        """Register a callback function to call after each iteration."""
        self._callbacks.append(func)
    
    def save(self):
        """Save this optimization to it's pre-existing project directory."""
        self.save_as(self.config['project_directory'])

    @ensure_path
    def save_as(self, project_dir: Path): 
        pass

    def _simulation_dispatch(self, sim_idx: int):
        """Target for threads to run simulations."""
        sim = self.sims[sim_idx]
        logging.debug(f'Running simulation on thread {sim_idx}')
        sim.save(f'_sim{sim_idx}_epoch_{self.epoch}_iter_{self.iteration}')
        sim.run()
        logging.debug(f'Completed running on thread {sim_idx}')

    def _pre_run(self):
        """Final pre-processing before running the optimization."""
        # Connect to Lumerical
        for sim in self.sims:
            sim.connect()
        
        self.save()


    def run(self):
        """Run the optimization."""
        self._pre_run()
        while self.epoch < self.max_epochs:
            while self.iteration < self.max_iter:
                for callback in self._callbacks:
                    callback(self)

                logging.debug(
                    f'Epoch {self.epoch}, iter {self.iteration}: Running simulations...'
                )
                # Run all the simulations
                n_sims = len(self.sims)
                with ThreadPoolExecutor() as ex:
                    ex.map(self._simulation_dispatch, range(n_sims))

                # Compute FoM and Gradient
                fom = self.fom.compute()
                self.fom_hist.append(fom)

                gradient = self.fom.gradient()

                logging.debug(
                    f'FoM at epoch {self.epoch}, iter {self.iteration}: {fom}\n'
                    f'Gradient {self.epoch}, iter {self.iteration}: {gradient}'
                )

                # Step with the gradient
                self.param_hist.append(self.device.get_design_variable())
                self.optimizer.step(self.device, gradient, self.iteration)

                self.iteration += 1
            self.iteration = 0
            self.epoch += 1

        final_fom = self.fom_hist[-1]
        logging.info(f'Final FoM: {final_fom}')
        final_params = self.param_hist[-1]
        logging.info(f'Final Parameters: {final_params}')
