"""Class for handling optimization setup, running, and saving."""

from __future__ import annotations

from collections.abc import Callable
from multiprocessing import Queue, Manager
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path

import numpy as np
import numpy.typing as npt

import vipdopt
from vipdopt.optimization.device import Device
from vipdopt.optimization.fom import FoM
from vipdopt.optimization.optimizer import GradientOptimizer
from vipdopt.simulation import LumericalSimulation
from vipdopt.utils import wait_for_results


def _simulation_dispatch(
        idx: int,
        work_dir: Path,
        objects: dict,
        env_vars: dict,
) -> Path:
    """Create and run a simulation and return the path of the save file."""
    output_path = work_dir / f'sim_{idx}.fsp'
    vipdopt.logger.debug(f'Starting simulation {idx}')
    # with LumericalSimulation(objects) as sim:
    #     sim.promise_env_setup(**env_vars)
    #     sim.save_lumapi(output_path)
    return output_path


class Optimization:
    """Class for orchestrating all the pieces of an optimization."""

    def __init__(
            self,
            sims: list[LumericalSimulation],
            device: Device,
            optimizer: GradientOptimizer,
            fom: FoM|None,
            start_epoch: int=0,
            start_iter: int=0,
            max_epochs: int=1,
            iter_per_epoch: int=100,
            work_dir: Path = Path('.'),
            env_vars: dict = {}
    ):
        """Initialize Optimzation object."""
        self.sims = list(sims)
        self.nsims = len(sims)
        self.sim_files = [work_dir / f'sim_{i}.fsp' for i in range(self.nsims)]
        self.device = device
        self.optimizer = optimizer
        self.fom = fom
        self.dir = work_dir
        self.env_vars = env_vars

        self.runner_sim = LumericalSimulation()  # Dummy sim for running in parallel
        self.runner_sim.promise_env_setup(**env_vars)


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

    def save_sims(self):
        """Save all simulation files."""
        with ThreadPool() as pool:
            pool.starmap(
                LumericalSimulation.save_lumapi,
                zip(self.sims, self.sim_files),
            )

    def load_sims(self):
        """Load all simulation files."""
        with ThreadPool() as pool:
            pool.starmap(
                LumericalSimulation.load,
                zip(self.sims, self.sim_files)
            )

    def _pre_run(self):
        """Final pre-processing before running the optimization."""
        # Connect to Lumerical
        with ThreadPool() as pool:
            pool.apply(LumericalSimulation.connect, (self.runner_sim,))
            pool.map(LumericalSimulation.connect, self.sims)

    def _post_run(self):
        """Final post-processing after running the optimization."""
        # Disconnect from Lumerical
        with ThreadPool() as pool:
            pool.map(LumericalSimulation.close, self.sims)

    def run_simulations(self):
        """Run all of the simulations in parallel."""

        # jobs = [
        #     (i, self.dir, sim.as_dict(), sim.get_env_vars())
        #     for i, sim in enumerate(self.sims)
        # ]

        # with Pool() as pool:
        #     sim_files = pool.starmap(_simulation_dispatch, jobs)

        # vipdopt.logger.debug('Creating new fdtd...')

        self.save_sims()
        # Use dummy simulation to run all of them at once using lumapi
        for fname in self.sim_files:
            self.runner_sim.fdtd.addjob(str(fname), 'FDTD')

        vipdopt.logger.debug('Running all simulations...')
        self.runner_sim.fdtd.runjobs('FDTD')
        vipdopt.logger.debug('Done running simulations')

        self.load_sims()
        
        # self.sims[0].connect()
        # vipdopt.logger.info(f'field_shape before: {self.sims[0].get_field_shape()}')
        # self.sims[0].load(self.dir / 'sim_0.fsp')
        # vipdopt.logger.info(f'field_shape after: {self.sims[0].get_field_shape()}')
        # Use a thread pool for this part so that state is shared
        
        # self.sims[0].connect()
        # self.sims[0].load(sim_files[0])
        # vipdopt.logger.debug(self.sims[0].fdtd.getresult('design_index_monitor'))
        
    
    def run(self):
        """Run the optimization."""
        self._pre_run()
        while self.epoch < self.max_epochs:
            while self.iteration < self.iter_per_epoch:
                for callback in self._callbacks:
                    callback(self)

                vipdopt.logger.debug(
                    f'Epoch {self.epoch}, iter {self.iteration}: Running simulations...'
                )

                # Run all the simulations
                self.run_simulations()
                
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

                # TODO: Update simulation device mesh

                self.iteration += 1
                break
            break
            self.iteration = 0
            self.epoch += 1

        final_fom = self.fom_hist[-1]
        vipdopt.logger.info(f'Final FoM: {final_fom}')
        final_params = self.param_hist[-1]
        vipdopt.logger.info(f'Final Parameters: {final_params}')
        self._post_run()
