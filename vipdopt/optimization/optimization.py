"""Class for handling optimization setup, running, and saving."""

from __future__ import annotations

from collections.abc import Callable
from multiprocessing import Pool
from pathlib import Path

import numpy.typing as npt

import vipdopt
from vipdopt.optimization.device import Device
from vipdopt.optimization.fom import FoM
from vipdopt.optimization.optimizer import GradientOptimizer
from vipdopt.simulation import LumericalSimulation


def _simulation_dispatch(
        idx: int,
        work_dir: Path,
        objects: dict,
        env_vars: dict,
) -> Path:
    """Create and run a simulation and return the path of the save file."""
    output_path = work_dir / f'sim_{idx}.fsp'
    with LumericalSimulation(objects) as sim:
        sim.promise_env_setup(**env_vars)
        sim.save_lumapi(output_path)
        vipdopt.logger.debug(f'Starting simulation {idx}')
        sim.run()
    vipdopt.logger.debug(f'Completed running simulation {idx}')
    return output_path


class Optimization:
    """Class for orchestrating all the pieces of an optimization."""

    def __init__(
            self,
            sims: list[LumericalSimulation],
            device: Device,
            optimizer: GradientOptimizer,
            fom: FoM,
            start_epoch: int=0,
            start_iter: int=0,
            max_epochs: int=1,
            iter_per_epoch: int=100,
            work_dir: Path = Path('.'),
    ):
        """Initialize Optimzation object."""
        self.sims = sims
        self.device = device
        self.optimizer = optimizer
        self.fom = fom
        self.dir = work_dir

        self.fom_hist: list[npt.NDArray] = []
        self.param_hist: list[npt.NDArray] = []
        self._callbacks: list[Callable[[Optimization], None]] = []

        self.epoch = start_epoch
        self.iteration = start_iter
        self.max_epochs = max_epochs
        self.iter_per_epoch = iter_per_epoch

    def add_callback(self, func: Callable):
        """Register a callback function to call after each iteration."""
        self._callbacks.append(func)


    def _pre_run(self):
        """Final pre-processing before running the optimization."""
        # Connect to Lumerical
        for sim in self.sims:
            sim.connect()

    def run(self):
        """Run the optimization."""
        while self.epoch < self.max_epochs:
            while self.iteration < self.iter_per_epoch:
                for callback in self._callbacks:
                    callback(self)

                vipdopt.logger.debug(
                    f'Epoch {self.epoch}, iter {self.iteration}: Running simulations...'
                )
                # Run all the simulations
                jobs = [
                    (i, self.dir, sim.as_dict(), sim.get_env_vars())
                    for i, sim in enumerate(self.sims)
                ]

                with Pool(len(self.sims)) as pool:
                    results_locations = pool.starmap(_simulation_dispatch, jobs)

                vipdopt.logger.debug(f'Sim results saved to {results_locations}')

                break

                # Compute FoM and Gradient
                fom = self.fom.compute()
                self.fom_hist.append(fom)

                gradient = self.fom.gradient()

                vipdopt.logger.debug(
                    f'FoM at epoch {self.epoch}, iter {self.iteration}: {fom}\n'
                    f'Gradient {self.epoch}, iter {self.iteration}: {gradient}'
                )

                # Step with the gradient
                self.param_hist.append(self.device.get_design_variable())
                self.optimizer.step(self.device, gradient, self.iteration)

                self.iteration += 1
            break
            self.iteration = 0
            self.epoch += 1

        return
        final_fom = self.fom_hist[-1]
        vipdopt.logger.info(f'Final FoM: {final_fom}')
        final_params = self.param_hist[-1]
        vipdopt.logger.info(f'Final Parameters: {final_params}')
