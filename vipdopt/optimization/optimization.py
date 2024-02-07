"""Class for handling optimization setup, running, and saving."""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from copy import copy
from typing import Type

import numpy.typing as npt

from vipdopt.optimization.device import Device
from vipdopt.optimization.optimizer import GradientOptimizer
from vipdopt.optimization.fom import FoM, BayerFilterFoM
from vipdopt.optimization.adam import AdamOptimizer, DEFAULT_ADAM
from vipdopt.simulation import ISimulation, LumericalSimulation, LumericalEncoder
from vipdopt.configuration import Config
from vipdopt.utils import ensure_path, save_config_file, read_config_file, PathLike
from vipdopt.monitor import Monitor


class Optimization:
    """Class for orchestrating all the pieces of an optimization."""

    def __init__(
            self,
            config: Config | None=None,
    ):
        """Initialize Optimzation object."""
        self.fom_hist: list[npt.NDArray] = []
        self.param_hist: list[npt.NDArray] = []
        self._callbacks: list[Callable[[Optimization], None]] = []
        self.dir = Path('.')  # Project directory; defaults to root

        self.epoch = 0
        self.iteration = 0
        self.config = Config()

        if config is not None:
            self.load_config(config)
        
    @classmethod
    def from_dir(cls: Type[Optimization], project_dir: PathLike) -> Optimization:
        opt = Optimization()
        opt.load_project(project_dir)
        return opt
    
    @ensure_path
    def load_project(self, project_location: Path):
        """Load settings from a project directory."""
        self.dir = project_location
        cfg = Config.from_file(self.dir / 'config.json')
        self.load_config(cfg)

    def load_config(self, config: Config | dict):
        """Load and setup optimization from an appropriate JSON config file."""
        if isinstance(config, dict):
            config = Config(config)
        cfg = copy(config)
        # Setup optimizer
        optimizer: str = cfg.pop('optimizer')
        optimizer_settings: dict = cfg.pop('optimizer_settings')
        try:
            optimizer_type = getattr(sys.modules[__name__], optimizer)
        except AttributeError:
            raise NotImplementedError(f'Optimizer {optimizer} not currently supported')
        self.optimizer = optimizer_type(**optimizer_settings)

        # Setup simulations
        base_sim = LumericalSimulation(cfg.pop('base_simulation'))
        base_sim.promise_env_setup(
            mpi_exe=cfg['mpi_exe'],
            nprocs=cfg['nprocs'],
            solver_exe=cfg['solver_exe'],
        )
        src_to_sim_map = {src: base_sim.with_enabled([src]) for src in base_sim.source_names()}
        self.sims = src_to_sim_map.values()
        self.base_sim = base_sim

        # Setup FoMs
        foms: list[FoM] = []
        weights = []
        fom: dict
        for name, fom in cfg.pop('figures_of_merit').items():
            fom_cls = getattr(sys.modules[__name__], fom['type'])
            foms.append(fom_cls(
                fom_monitors=[
                    Monitor(
                        src_to_sim_map[src],
                        src,
                        mname,
                    ) for src, mname in fom['fom_monitors']
                ],
                grad_monitors=[
                    Monitor(
                        src_to_sim_map[src],
                        src,
                        mname,
                    ) for src, mname in fom['grad_monitors']
                ],
                polarization=fom['polarization'],
                freq=fom['freq'],
                opt_ids=fom.get('opt_ids', None),
                name=name,
            ))
            weights.append(fom['weight'])
        self.foms = foms
        self.weights = weights

        # TODO Load the Device

        self.config = cfg

    def add_callback(self, func: Callable):
        """Register a callback function to call after each iteration."""
        self._callbacks.append(func)
    
    def save(self):
        """Save this optimization to it's pre-existing project directory."""
        self.save_as(self.config['project_directory'])
    
    @ensure_path
    def save_as(self, project_dir: Path): 
        """Save this optimization to a specified directory, creating it if necessary."""
        project_dir.mkdir(parents=True, exist_ok=True)
        cfg = self._generate_config()
        cfg.save_file(project_dir / 'config.json', cls=LumericalEncoder)

        # TODO Save Device, histories, and plots
    
    def _generate_config(self) -> Config:
        """Create a JSON config for this optimization's settings."""
        cfg = copy(self.config)

        # Simulation
        cfg['base_simulation'] = self.base_sim.as_dict()

        # FoMs
        foms = []
        for i, fom in enumerate(self.foms):
            data = {}
            data['type'] = type(fom).__name__
            data['fom_monitors'] = [(mon.source_name, mon.monitor_name) for mon in fom.fom_monitors]
            data['grad_monitors'] = [
                (mon.source_name, mon.monitor_name) for mon in fom.grad_monitors
            ]
            data['polarization'] = fom.polarization
            data['freq'] = fom.freq
            data['opt_ids'] = fom.opt_ids
            data['weight'] = self.weights[i]
            foms.append(data)
        cfg['figures_of_merit'] = {fom.name: foms[i] for i, fom in enumerate(self.foms)}

        # Optimizer
        cfg['optimizer'] = type(self.optimizer).__name__
        cfg['optimizer_settings'] = vars(self.optimizer)

        # Remaining Settings
        cfg['epoch'] = self.epoch
        cfg['iteration'] = self.iteration

        return cfg


    def _simulation_dispatch(self, sim_idx: int):
        """Target for threads to run simulations."""
        sim = self.sims[sim_idx]
        logging.debug(f'Running simulation on thread {sim_idx}')
        sim.save_lumapi(f'_sim{sim_idx}_epoch_{self.epoch}_iter_{self.iteration}')
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

if __name__ == '__main__':
    project_dir = Path('./test_project/')
    output_dir = Path('./test_output_optimization/')

    opt = Optimization.from_dir(project_dir)
    opt.save_as(output_dir)

    # Test that the saved format is loadable
    reload = Optimization.from_dir(output_dir)

    # Also this way
    reload2 = Optimization()
    reload2.load_project(output_dir)
