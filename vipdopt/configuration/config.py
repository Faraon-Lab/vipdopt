"""Module for handling configuration parameters."""

import logging
from collections.abc import Callable
from typing import Any

import yaml
from yaml.constructor import SafeConstructor


def get_config_loader(config_format) -> Callable[[str], dict]:
    """Return a configuration file loader depending on the format."""
    match config_format:
        case 'yaml':
            return _yaml_loader
        case _:
            msg = f'{config_format} file loading not yet supported.'
            raise NotImplementedError(msg)


def _yaml_loader(filepath: str) -> dict:
    """Config file loader for YAML files."""
    # Allow the safeloader to convert sequences to tuples
    SafeConstructor.add_constructor(
        'tag:yaml.org,2002:python/tuple',
        lambda self, x: tuple(self.construct_sequence(x))
    )
    with open(filepath, 'rb') as stream:
        return yaml.safe_load(stream)


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

    def __getattr__(self, name: str) -> Any:
        """Get an attrbitute of the Config object."""
        try:
            return self.__dict__[name]
        except KeyError as e:
            msg = f"'{type(self).__name__}' has no attribute '{name}'"
            raise KeyError(msg) from e

    def __setattr__(self, name: str, value: Any) -> Any:
        """Set the value of an attribute, creating it if it doesn't already exist."""
        self.__dict__[name] = value

    def read_file(self, filename: str, cfg_format: str='yaml'):
        """Read a config file and update the dictionary."""
        # Get the correct loader method for the format and load the config file
        config_loader = get_config_loader(cfg_format)
        config = config_loader(filename)
        logging.debug(f'Loaded config file:\n {config}')

        # Create an attribute for each of the parameters in the config file
        if config is not None:
            self.__dict__.update(config)

        self._files.add(filename)
        logging.info(f'\nSuccesfully loaded configuration from {filename}')
