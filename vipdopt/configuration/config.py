"""Module for handling configuration parameters."""

from collections.abc import Callable
from typing import Any

import yaml


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
    with open(filepath, 'rb') as stream:
        return yaml.safe_load(stream)


class Config:
    """A wrapper class for a loaded configuration file."""

    def __init__(self, config_file, config_format='yaml'):
        """Initialize a Config option.

        Creates an attribute for each parameter in a provided config file.
        """
        self.config_file = config_file

        # Get the correct loader method for the format and load the config file
        config_loader = get_config_loader(config_format)
        config = config_loader(config_file)

        # Create an attribute for each of the parameters in the resulting dict
        if config is not None:
            for (k, v) in config.items():
                self.__dict__[k] = v

    def __repr__(self):
        """Return a string representation of the Config object."""
        return f'Config object representing {self.config_file}'

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


class SonyBayerConfig(Config):
    """Config object specifically for use with the Sony bayer filter optimization."""

    # List of all the cascading parameters, their dependencies, and a function to
    # compute it.
    # TODO: Is this going to have to be hard-coded?
    _cascading_parameters = (
        # e.g. ('sum_of_a_and_b', ('a', 'b'), lambda a, b: a + b),
        (
            'total_iterations',
            ('num_epochs', 'num_iterations_per_epoch'),
            lambda a, b: a * b
        ),
    )

    def __init__(self, config_file, config_format='yaml'):
        """Initialize SonyBayerConfig object."""
        super().__init__(config_file, config_format)
        self._derive_cascading_params()

    def _derive_cascading_params(self):
        """Compute all the cascading parameters."""
        for (name, deps, f) in self._cascading_parameters:
            # Map all dependency attributes to their values
            deps_vals = tuple(self.__getattr__(x) for x in deps)

            # Call the provided function to compute `name` with the dependencies
            self.__setattr__(name, f(*deps_vals))
