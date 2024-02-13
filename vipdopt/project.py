"""Code for managing a project."""

from __future__ import annotations

import logging
import sys
from copy import copy
from pathlib import Path
from typing import Any

import numpy as np

from vipdopt.configuration import Config, SonyBayerConfig
from vipdopt.optimization import Device, FoM, GradientOptimizer, Optimization
from vipdopt.simulation import LumericalEncoder, LumericalSimulation
from vipdopt.utils import PathLike, ensure_path


class Project:
    """Class for managing the loading and saving of projects."""

    def __init__(self, config_type: type[Config]=SonyBayerConfig) -> None:
        """Initialize a Project."""
        self.dir = Path('.')  # Project directory; defaults to root
        self.config = config_type()
        self.config_type = config_type

        self.optimization: Optimization = None
        self.optimizer: GradientOptimizer = None
        self.device: Device = None
        self.base_sim: LumericalSimulation = None

    @classmethod
    def from_dir(
        cls: type[Project],
        project_dir: PathLike,
        config_type: type[Config]=SonyBayerConfig,
    ) -> Project:
        """Create a new Project from an existing project directory."""
        proj = Project(config_type=config_type)
        proj.load_project(project_dir)
        return proj

    @ensure_path
    def load_project(self, project_dir: Path):
        """Load settings from a project directory."""
        self.dir = project_dir
        cfg = Config.from_file(project_dir / 'config.json')
        self._load_config(cfg)

    def _load_config(self, config: Config | dict):
        """Load and setup optimization from an appropriate JSON config file."""
        if isinstance(config, dict):
            config = self.config_type(config)
        cfg = copy(config)

        # Setup optimizer
        optimizer: str = cfg.pop('optimizer')
        optimizer_settings: dict = cfg.pop('optimizer_settings')
        try:
            optimizer_type = getattr(sys.modules[__name__], optimizer)
        except AttributeError:
            raise NotImplementedError(f'Optimizer {optimizer} not currently supported')\
            from None
        self.optimizer = optimizer_type(**optimizer_settings)

        # Setup simulations
        base_sim = LumericalSimulation(cfg.pop('base_simulation'))
        base_sim.promise_env_setup(
            mpi_exe=cfg['mpi_exe'],
            nprocs=cfg['nprocs'],
            solver_exe=cfg['solver_exe'],
        )
        self.base_sim = base_sim
        src_to_sim_map = {
            src: base_sim.with_enabled([src]) for src in base_sim.source_names()
        }
        sims = src_to_sim_map.values()

        # General Settings
        iteration = cfg.get('current_iteration', 0)
        epoch = cfg.get('current_epoch', 0)

        # Setup FoMs
        foms: list[FoM] = []
        weights = []
        fom_dict: dict
        for name, fom_dict in cfg.pop('figures_of_merit').items():
            foms.append(FoM.from_dict(name, fom_dict, src_to_sim_map))
            weights.append(fom_dict['weight'])
        self.foms = foms
        self.weights = weights
        full_fom = sum(np.multiply(weights, foms), FoM.zero(foms[0]))

        # Load device
        device_source = cfg.pop('device')
        self.device = Device.from_source(device_source)

        # Setup Optimization
        self.optimization = Optimization(
            sims,
            self.device,
            self.optimizer,
            full_fom,
            start_epoch=epoch,
            start_iter=iteration,
            max_epochs=cfg.get('max_epochs', 1),
            iter_per_epoch=cfg.get('iter_per_epoch', 100),
        )

        # TODO Register any callbacks for Optimization here

        self.config = cfg
        logging.debug(f'set config to {cfg}')

    def get(self, prop: str, default: Any=None) -> Any | None:
        """Get parameter and return None if it doesn't exist."""
        return self.config.get(prop, default)

    def pop(self, name: str) -> Any:
        """Pop a parameter."""
        return self.config.pop(name)

    def __getitem__(self, name: str) -> Any:
        """Get value of a parameter."""
        return self.config[name]

    def __setitem__(self, name: str, value: Any) -> None:
        """Set the value of an attribute, creating it if it doesn't already exist."""
        self.config[name] = value

    def current_device_path(self) -> Path:
        """Save the current device."""
        epoch = self.optimization.epoch
        iteration = self.optimization.iteration
        return self.dir / 'device' / f'e_{epoch}_i_{iteration}.npy'

    def save(self):
        """Save this project to it's pre-assigned directory."""
        self.save_as(self.dir)

    @ensure_path
    def save_as(self, project_dir: Path):
        """Save this project to a specified directory, creating it if necessary."""
        (project_dir / 'device').mkdir(parents=True, exist_ok=True)
        self.dir = project_dir
        cfg = self._generate_config()

        self.device.save(cfg['device'])  # Save device to file for current epoch/iter
        cfg.save(project_dir / 'config.json', cls=LumericalEncoder)


        # TODO Save histories and plots

    def _generate_config(self) -> Config:
        """Create a JSON config for this project's settings."""
        cfg = copy(self.config)

        # Miscellaneous Settings
        cfg['current_epoch'] = self.optimization.epoch
        cfg['current_iteration'] = self.optimization.iteration

        # Optimizer
        cfg['optimizer'] = type(self.optimizer).__name__
        cfg['optimizer_settings'] = vars(self.optimizer)

        # FoMs
        foms = []
        for i, fom in enumerate(self.foms):
            data = fom.as_dict()
            data['weight'] = self.weights[i]
            foms.append(data)
        cfg['figures_of_merit'] = {fom.name: foms[i] for i, fom in enumerate(self.foms)}

        # Device
        cfg['device'] = self.current_device_path()

        # Simulation
        cfg['base_simulation'] = self.base_sim.as_dict()

        return cfg


if __name__ == '__main__':
    project_dir = Path('./test_project/')
    output_dir = Path('./test_output_optimization/')

    opt = Project.from_dir(project_dir)
    opt.save_as(output_dir)

    # Test that the saved format is loadable

    # # Also this way
