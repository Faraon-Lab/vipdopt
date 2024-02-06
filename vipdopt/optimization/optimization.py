"""Class for handling optimization setup, running, and saving."""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy.typing as npt

from vipdopt.optimization.device import Device
from vipdopt.optimization.fom import FoM
from vipdopt.optimization.optimizer import GradientOptimizer
from vipdopt.simulation import ISimulation
from vipdopt.utils import ensure_path, PathLike


class Optimization:
    """Class for orchestrating all the pieces of an optimization."""

    def __init__(
            self,
            sims: list[ISimulation],
            device: Device,
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
        self.optimizer = optimizer
        self.fom_hist: list[npt.NDArray] = []
        self.param_hist: list[npt.NDArray] = []
        self._callbacks: list[Callable[[Optimization], None]] = []
        self.max_iter = max_iter
        self.max_epochs = max_epochs
        self.iteration = start_iter
        self.epoch = start_epoch

    def add_callback(self, func: Callable):
        """Register a callback function to call after each iteration."""
        self._callbacks.append(func)
    
    @ensure_path
    def save(self, directory: PathLike): 
        pass

    def _simulation_dispatch(self, sim_idx: int):
        """Target for threads to run simulations."""
        sim = self.sims[sim_idx]
        logging.debug(f'Running simulation on thread {sim_idx}')
        sim.save(f'_sim{sim_idx}_epoch_{self.epoch}_iter_{self.iteration}')
        sim.run()
        logging.debug(f'Completed running on thread {sim_idx}')

    def run(self):
        """Run the optimization."""
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
