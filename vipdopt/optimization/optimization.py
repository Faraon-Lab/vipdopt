"""Class for handling optimization setup, running, and saving."""

from __future__ import annotations
from typing import Callable
import logging


from vipdopt.optimization.device import Device
from vipdopt.optimization.optimizer import GradientOptimizer
from vipdopt.simulation import ISimulation
from vipdopt.optimization.fom import FoM


class Optimization:
    """Class for orchestrating all the pieces of an optimization."""

    def __init__(
            self,
            sim: ISimulation,
            device: Device,
            fom: FoM,
            optimizer: GradientOptimizer,
            max_iter: int,
            max_epochs: int,
            start_iter: int=0,
            start_epoch: int=0,
    ):
        """Initialize Optimzation object."""
        self.sim = sim
        self.fom = fom
        self.device = device
        self.optimizer = optimizer
        self.fom_hist = []
        self.param_hist = []
        self._callbacks = []
        self.max_iter = max_iter
        self.max_epochs = max_epochs
        self.iteration = start_iter
        self.epoch = start_epoch

    def add_callback(self, func: Callable):
        """Register a callback function to call after each iteration."""
        self._callbacks.apend(func)
        
    def run(self):
        while self.epoch < self.max_epoch:
            while self.iteration < self.max_iter:
                for callback in self._callbacks:
                    callback()

                self.sim.save(f'_sim_epoch_{self.epoch}_iter_{self.iteration}')
                self.sim.run()
                
                fom = self.fom.compute()
                self.fom_hist.append(fom)

                gradient = self.fom.gradient()

                # Step with the gradient
                self.param_hist.append(self.device.get_design_variable())
                self.optimizer.step(self.device, gradient)
            
                self.iteration += 1
            self.epoch += 1

        final_fom = self.optimizer.fom_hist[-1]
        logging.info(f'Final FoM: {final_fom}')
        final_params = self.optimizer.param_hist[-1]
        logging.info(f'Final Parameters: {final_params}')
