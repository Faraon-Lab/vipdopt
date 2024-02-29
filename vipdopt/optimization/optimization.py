"""Class for handling optimization setup, running, and saving."""

from __future__ import annotations

from collections.abc import Callable
from multiprocessing import Queue, Manager
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
import signal
import pickle

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

import vipdopt
from vipdopt.optimization.device import Device
from vipdopt.optimization.fom import FoM
from vipdopt.optimization.optimizer import GradientOptimizer
from vipdopt.simulation import LumericalSimulation
from vipdopt.utils import wait_for_results



DEFAULT_OPT_FOLDERS = {'temp': Path('.'), 'opt_info': Path('.'), 'opt_plots': Path('.')}
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
            dirs: dict[str, Path]=DEFAULT_OPT_FOLDERS,
            env_vars: dict = {}
    ):
        """Initialize Optimzation object."""
        self.sims = list(sims)
        self.nsims = len(sims)
        self.device = device
        self.optimizer = optimizer
        self.fom = fom
        self.dirs = dirs
        self.sim_files = [dirs['temp'] / f'sim_{i}.fsp' for i in range(self.nsims)]
        self.env_vars = env_vars
        self.loop = True

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
    
    def save_histories(self):
        """Save the fom and parameter histories to file."""
        folder = self.dirs['opt_info']
        foms = np.array(self.fom_hist)
        with (folder / 'fom_history.npy').open('wb') as f:
            np.save(f, foms)
        params = np.array(self.param_hist)
        with (folder / 'paramater_history.npy').open('wb') as f:
            np.save(f, params)
    
    def generate_plots(self):
        """Generate the plots and save to file."""
        folder = self.dirs['opt_plots']

        # Create Plots
        foms = np.array(self.fom_hist)
        fig, axs = plt.subplots(1, 1)
        fom_plot = axs.plot(range(len(foms)), foms.mean(axis=tuple(range(1, foms.ndim))))
        with (folder / 'fom.pkl').open('wb') as f:
            pickle.dump(fom_plot, f)
        

    def _pre_run(self):
        """Final pre-processing before running the optimization."""
        # Connect to Lumerical
        self.loop = True
        return
        with ThreadPool() as pool:
            pool.apply(LumericalSimulation.connect, (self.runner_sim,))
            pool.map(LumericalSimulation.connect, self.sims)

    def _post_run(self):
        """Final post-processing after running the optimization."""
        # Disconnect from Lumerical
        self.loop = False
        self.save_histories()
        self.generate_plots()
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
        if self.nsims == 0:
            return

        vipdopt.logger.debug(
            f'Epoch {self.epoch}, iter {self.iteration}: Running simulations...'
        )

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
        vipdopt.logger.info(f'Initial Device: {self.device.get_design_variable()}')
        while self.epoch < self.max_epochs:
            while self.iteration < self.iter_per_epoch:
                if not self.loop:
                    break
                for callback in self._callbacks:
                    callback(self)

                vipdopt.logger.debug(
                    f'Progress at epoch {self.epoch}, iter {self.iteration}:\n'
                    f'\tDesign Variable: {self.device.get_design_variable()}'
                )

                # Run all the simulations
                self.run_simulations()
                
                # Compute FoM and Gradient
                fom = self.fom.compute(self.device.get_design_variable())
                self.fom_hist.append(fom)

                gradient = self.fom.gradient(self.device.get_design_variable())

                vipdopt.logger.debug(
                    f'\tFoM: {fom}\n'
                    f'\tGradient: {gradient}'
                )

                # Step with the gradient
                self.param_hist.append(self.device.get_design_variable())
                self.optimizer.step(self.device, gradient, self.iteration)

                # TODO: Update simulation device mesh

                self.iteration += 1
            if not self.loop:
                break
            self.epoch += 1

        final_fom = self.fom_hist[-1]
        vipdopt.logger.info(f'Final FoM: {final_fom}')
        final_params = self.param_hist[-1]
        vipdopt.logger.info(f'Final Parameters: {final_params}')
        self._post_run()
