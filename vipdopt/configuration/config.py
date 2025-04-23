"""Module for handling configuration parameters."""

from __future__ import annotations
from collections import UserDict
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, overload
if TYPE_CHECKING:
    from _typeshed import SupportsKeysAndGetItem

import json
from pathlib import Path
from overrides import override

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

        vipdopt.logger.info(f'\nSuccessfully loaded configuration from {fname}')

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
                    try:
                        yaml.dump(config_data, f, **kwargs)
                    except Exception as err:    # yaml dump function can't take unexpected kwargs
                        yaml.dump(config_data, f)
            case '.json':
                with path_filename.open('w') as f:
                    json.dump(config_data, f, indent=4, ensure_ascii=True, **kwargs)
            case _:
                msg = f'{cfg_format} file saving not yet supported.'
                raise NotImplementedError(msg)
    
    @classmethod
    @ensure_path
    def save_new(self, fname: Path, data, cfg_format: str = 'auto', **kwargs) -> None:
        """Save a configuration file."""
        path_filename = convert_path(fname)
        if cfg_format.lower() == 'auto':
            cfg_format = path_filename.suffix

        config_data = data

        match cfg_format.lower():
            case '.yaml' | '.yml':
                with path_filename.open('w') as f:
                    try:
                        yaml.dump(config_data, f, **kwargs)
                    except Exception as err:    # yaml dump function can't take unexpected kwargs
                        yaml.dump(config_data, f)
            case '.json':
                with path_filename.open('w') as f:
                    json.dump(config_data, f, indent=4, ensure_ascii=True, **kwargs)
            case _:
                msg = f'{cfg_format} file saving not yet supported.'
                raise NotImplementedError(msg)

class ProjectConfig(Config):
    """Config object used to save and load Project classes (see vipdopt/project.py)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @override
    def __setitem__(self, name: str, value: Any) -> None:
        super().__setitem__(name, value)

    @ensure_path
    @override
    def read_file(self, fname: Path, cfg_format: str = 'auto') -> None:
        super().read_file(fname, cfg_format=cfg_format)

    @overload
    def update(self, __m: SupportsKeysAndGetItem, **kwargs: Any) -> None: ...

    @overload
    def update(self, __m: Iterable[tuple[Any, Any]], **kwargs) -> None: ...

    @overload
    def update(self, **kwargs: Any) -> None: ...

    def update(self, *args, **kwargs: Any) -> None:
        """Update self with values from another dictionary-like object."""
        self._do_validation = False
        if len(args) == 0:
            super().update(**kwargs)
        else:
            super().update(args[0], **kwargs)