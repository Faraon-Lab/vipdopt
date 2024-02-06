"""Module for handling configuration parameters."""

import logging
from typing import Any
from pathlib import Path

from vipdopt.utils import ensure_path, read_config_file


class Config:
    """A generic class for storing parameters from a configuration file."""

    def __init__(self):
        """Initialize a Config object.

        Creates an attribute for each parameter in a provided config file.
        """
        self._files = set()
        self._parameters = {}

    def __str__(self):
        """Return shorter string version of the Config object."""
        return f'Config object for {self._files} with parameters {self._parameters}'

    def get(self, prop: str, default: Any=None) -> Any | None:
        """Get parameter and return None if it doesn't exist."""
        return self._parameters.get(prop, default)

    def __getitem__(self, name: str) -> Any:
        """Get value of a parameter."""
        return self._parameters[name]

    def __setitem__(self, name: str, value: Any) -> None:
        """Set the value of an attribute, creating it if it doesn't already exist."""
        self._parameters[name] = value

    @ensure_path
    def read_file(self, fname: Path, cfg_format: str='auto') -> None:
        """Read a config file and update the dictionary."""
        # Get the correct loader method for the format and load the config file
        config = read_config_file(fname, cfg_format)
        logging.debug(f'Loaded config file:\n {config}')

        # Create an attribute for each of the parameters in the config file
        if config is not None:
            self._parameters.update(config)

        self._files.add(fname)
        logging.info(f'\nSuccesfully loaded configuration from {fname}')
