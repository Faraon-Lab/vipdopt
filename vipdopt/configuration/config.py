"""Module for handling configuration parameters."""

from __future__ import annotations

import logging
from typing import Any, Type
from collections.abc import ItemsView, KeysView, ValuesView
from collections import UserDict
from pathlib import Path

from vipdopt.utils import ensure_path, read_config_file, save_config_file, T


class Config(UserDict):
    """A generic class for storing parameters from a configuration file."""

    # def __init__(self, data: dict | Config | None=None):
    #     """Initialize a Config object.

    #     Creates an attribute for each parameter in a provided config file.
    #     """
    #     self._parameters = {}
    #     if data is not None:
    #         self.update(data)

    def __str__(self):
        """Return shorter string version of the Config object."""
        return f'Config with parameters {super().__str__()}'

    # def get(self, prop: str, default: Any=None) -> Any | None:
    #     """Get parameter and return `default` if it doesn't exist."""
    #     return self._parameters.get(prop, default)
    
    # def pop(self, name: str) -> Any:
    #     """Pop a parameter."""
    #     return self._parameters.pop(name)

    # def __getitem__(self, name: str) -> Any:
    #     """Get value of a parameter."""
    #     return self._parameters[name]

    # def __setitem__(self, name: str, value: Any) -> None:
    #     """Set the value of an attribute, creating it if it doesn't already exist."""
    #     self._parameters[name] = value
    
    # def update(self, new_vals: dict | Config):
    #     """Update a Config with provided values."""
    #     vals = new_vals if isinstance(new_vals, dict) else new_vals._parameters
    #     self._parameters.update(vals)
    
    # def __copy__(self) -> Config:
    #     """Return a (shallow) copy of this Config."""
    #     c = Config()
    #     c.update(self)
    #     return c
    
    # def __contains__(self, item: Any) -> bool:
    #     return item in self._parameters
    
    # def items(self) -> ItemsView:
    #     return self._parameters.items()
    
    # def keys(self) -> KeysView:
    #     return self._parameters.keys()

    # def values(self) -> ValuesView:
    #     return self._parameters.values()
    
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
    def from_file(cls: Type[Config], fname: Path) -> Type[Config]:
        cfg = cls()
        cfg.read_file(fname)
        return cfg
    
    @ensure_path
    def save(self, fname: Path, cfg_format: str='auto', **kwargs) -> None:
        """Save a config file to a file."""
        save_config_file(self, fname, cfg_format, **kwargs)
