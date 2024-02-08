"""Code for managing a project"""

from __future__ import annotations

from typing import Any, Type, Generic
from pathlib import Path
from copy import copy
import sys

import numpy as np

from vipdopt.utils import PathLike, ensure_path, T
from vipdopt.configuration import Config, SonyBayerConfig
from vipdopt.optimization import *
from vipdopt.monitor import Monitor
from vipdopt.simulation import LumericalSimulation, LumericalEncoder, ISimulation

class Project:

    def __init__(self, config_type: Type[Config]=SonyBayerConfig) -> None:
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
        cls: Type[Project],
        project_dir: PathLike,
        config_type: Type[Config]=SonyBayerConfig,
    ) -> Project:
        """Create a new Project from an existing project directory."""
        proj = Project(config_type=config_type)
        proj.load_project(project_dir)
        return proj
    
    @ensure_path
    def load_project(self, project_location: Path):
        """Load settings from a project directory."""
        self.dir = project_location
        cfg = Config.from_file(self.dir / 'config.json')
        self.load_config(cfg)

    def load_config(self, config: Config | dict):
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
            raise NotImplementedError(f'Optimizer {optimizer} not currently supported')
        self.optimizer = optimizer_type(**optimizer_settings)

        # Setup simulations
        base_sim = LumericalSimulation(cfg.pop('base_simulation'))
        base_sim.promise_env_setup(
            mpi_exe=cfg['mpi_exe'],
            nprocs=cfg['nprocs'],
            solver_exe=cfg['solver_exe'],
        )
        self.base_sim = base_sim
        src_to_sim_map = {src: base_sim.with_enabled([src]) for src in base_sim.source_names()}
        sims = src_to_sim_map.values()

        # General Settings
        iteration = cfg.get('current_iteration', 0)
        epoch = cfg.get('current_epoch', 0)

        # Setup FoMs
        foms: list[FoM] = []
        weights = []
        fom: dict
        for name, fom in cfg.pop('figures_of_merit').items():
            fom_cls: Type[FoM] = getattr(sys.modules[__name__], fom['type'])
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
        full_fom = sum(np.multiply(weights, foms), fom_cls.zero(foms[0]))

        # Load the Device
        filters: list[Filter] = []
        filt: dict
        for filt in cfg['device']['filters']:
            filter_cls: Type[Filter] = getattr(sys.modules[__name__], filt['type'])
            filters.append(filter_cls(
                **filt['parameters'],
            ))
        device_data = cfg.pop('device')
        device_data['filters'] = filters
        self.device = Device(**device_data)
        # TODO Actually load the design variable. Likely will need to rework device.py

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
    
    def save(self):
        """Save this project to it's pre-assigned directory."""
        self.save_as(self.dir)
    
    @ensure_path
    def save_as(self, project_dir: Path): 
        """Save this project to a specified directory, creating it if necessary."""
        project_dir.mkdir(parents=True, exist_ok=True)
        cfg = self._generate_config()
        cfg.save_file(project_dir / 'config.json', cls=LumericalEncoder)

        # TODO Save Device, histories, and plots

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

        # Simulation
        cfg['base_simulation'] = self.base_sim.as_dict()

        return cfg


if __name__ == '__main__':
    project_dir = Path('./test_project/')
    output_dir = Path('./test_output_optimization/')

    opt = Project.from_dir(project_dir)
    # opt.save_as(output_dir)

    # Test that the saved format is loadable
    # reload = Project.from_dir(output_dir)

    # # Also this way
    # reload2 = Optimization()
    # reload2.load_project(output_dir)
