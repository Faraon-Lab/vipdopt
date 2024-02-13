"""Module for handling configuration parameters."""

from __future__ import annotations

import logging
from collections import UserDict
from pathlib import Path

from vipdopt.utils import ensure_path, read_config_file, save_config_file


class Config(UserDict):
    """A generic class for storing parameters from a configuration file."""

    # def __init__(self, data: dict | Config | None=None):
    #     """Initialize a Config object.

    #     Creates an attribute for each parameter in a provided config file.
    #     """
    #     if data is not None:

    def __str__(self):
        """Return shorter string version of the Config object."""
        return f'Config with parameters {super().__str__()}'

    @ensure_path
    def read_file(self, fname: Path, cfg_format: str='auto') -> None:
        """Read a config file and update the dictionary."""
        # Get the correct loader method for the format and load the config file
        config = read_config_file(fname, cfg_format)
        logging.debug(f'Loaded config file:\n {config}')

        # Create an attribute for each of the parameters in the config file
        if config is not None:
            self.update(config)

        logging.info(f'\nSuccesfully loaded configuration from {fname}')

    @classmethod
    @ensure_path
    def from_file(cls: type[Config], fname: Path) -> type[Config]:
        """Create config object from a file."""
        cfg = cls()
        cfg.read_file(fname)
        return cfg

    @ensure_path
    def save(self, fname: Path, cfg_format: str='auto', **kwargs) -> None:
        """Save a config file to a file."""
        save_config_file(self, fname, cfg_format, **kwargs)
