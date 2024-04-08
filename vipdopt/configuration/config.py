"""Module for handling configuration parameters."""

from __future__ import annotations

import json
from collections import UserDict
from pathlib import Path

import yaml

import vipdopt
from vipdopt.utils import convert_path, ensure_path, read_config_file


class Config(UserDict):
    """A generic class for storing parameters from a configuration file."""

    def __str__(self):
        """Return shorter string version of the Config object."""
        return f'Config with parameters {super().__str__()}'

    @ensure_path
    def read_file(self, fname: Path, cfg_format: str = 'auto') -> None:
        """Read a config file and update the dictionary."""
        # Get the correct loader method for the format and load the config file
        config = read_config_file(fname, cfg_format)
        vipdopt.logger.debug(f'Loaded config file:\n {config}')

        # Create an attribute for each of the parameters in the config file
        if config is not None:
            self.update(config)

        vipdopt.logger.info(f'\nSuccesfully loaded configuration from {fname}')

    @classmethod
    def from_file(cls: type[Config], fname: Path) -> Config:
        """Create config object from a file."""
        cfg = cls()
        cfg.read_file(fname)
        return cfg

    @ensure_path
    def save(self, fname: Path, cfg_format: str = 'auto', **kwargs) -> None:
        """Save a configuration file."""
        path_filename = convert_path(fname)
        if cfg_format.lower() == 'auto':
            cfg_format = path_filename.suffix

        config_data = self.data

        match cfg_format.lower():
            case '.yaml' | '.yml':
                with path_filename.open('w') as f:
                    yaml.dump(config_data, f, **kwargs)
            case '.json':
                with path_filename.open('w') as f:
                    json.dump(config_data, f, indent=4, ensure_ascii=True, **kwargs)
            case _:
                msg = f'{cfg_format} file saving not yet supported.'
                raise NotImplementedError(msg)
