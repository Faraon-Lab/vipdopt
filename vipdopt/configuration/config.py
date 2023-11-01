"""Module for handling configuration parameters."""

import logging
from collections.abc import Callable
from typing import Any

from vipdopt.utils import read_config_file


class Config:
    """A generic class for stroing parameters from a configuration file."""

    def __init__(self):
        """Initialize a Config object.

        Creates an attribute for each parameter in a provided config file.
        """
        self._files = set()

    def __str__(self):
        """Return shorter string version of the Config object."""
        return f'Config object for {self._files}'


    def safe_get(self, name: str | Callable) -> Any | None:
        """Get attribute and return None if it doesn't exist."""
        if isinstance(name, str):
            return getattr(self, name, None)
        try:
            return name.__get__(self)
        except AttributeError:
            return None

    def __setattr__(self, name: str, value: Any) -> Any:
        """Set the value of an attribute, creating it if it doesn't already exist."""
        self.__dict__[name] = value

    def read_file(self, filename: str, cfg_format: str='auto'):
        """Read a config file and update the dictionary."""
        # Get the correct loader method for the format and load the config file
        config = read_config_file(filename, cfg_format)
        logging.debug(f'Loaded config file:\n {config}')

        # Create an attribute for each of the parameters in the config file
        if config is not None:
            self.__dict__.update(config)

        self._files.add(filename)
        logging.info(f'\nSuccesfully loaded configuration from {filename}')
