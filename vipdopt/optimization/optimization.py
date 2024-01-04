"""Class for handling optimization setup, running, and saving."""

from __future__ import annotations
from typing import Callable
import logging

from vipdopt.device import Device
from vipdopt.simulation import ISimulation
from vipdopt.optimization.fom import FoM


class Optimization:
    """Class for orchestrating all the pieces of an optimization."""

    def __init__(
            self,
            sim: ISimulation,
            fom: list[FoM],
            device: Device,
            optimizer: Optimizer,
    ):
        """Initialize Optimzation object."""
        self.sim = sim
        self.fom = fom
        self.device = device
        self.optimizer = optimizer
    
    def run(
            self,
            max_iter: int,
            max_epochs: int,
    ):
        """Run this optimization."""
        self.optimizer.run()

        final_fom = self.optimizer.fom_hist[-1]
        logging.info(f'Final FoM: {final_fom}')
        final_params = self.optimizer.param_hist[-1]
        logging.info(f'Final Parameters: {final_params}')
        return final_fom, final_params
    
        

class Optimizer:
    """Abstraction class for all optimizers."""
    # TODO: SHOULD NOT BE JUST GRADIENT BASED (e.g. genetic algorithms)

    def __init__(
            self,
            max_iter: int=100,
            start_from: int=0,
            fom_func: Callable=None,
            grad_func: Callable=None,
    ) -> None:
        """Initialize an Optimizer object."""
        self.fom_hist = []
        self.param_hist = []
        self._callbacks = []
        self.iteration = start_from
        self.max_iter = max_iter
        self.fom_func = fom_func
        self.grad_func = fom_func


    def add_callback(self, func: Callable):
        """Register a callback function to call after each iteration."""
        self._callbacks.apend(func)

    def step(self, device: Device, *args, **kwargs):
        """Step forward one iteration in the optimization process."""
        # grad = self.device.backpropagate(gradient)

        # x = some_func(x)

        # self.device.set_design_variable(x)
    
    def run(self):
        """Run an optimization."""
        for callback in self._callbacks:
            callback()