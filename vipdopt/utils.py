"""Module containing common functions used throughout the package."""

import importlib.util as imp
import logging
from collections.abc import Callable
from importlib.abc import Loader
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt
import yaml
from yaml.constructor import SafeConstructor

Number = int | float | complex


# Generic types for type hints
T = TypeVar('T')
R = TypeVar('R')


class TruncateFormatter(logging.Formatter):
    """Loggign formatter for truncating large output."""
    def __init__(self, max_length: int=300, log_file: str='dev.log', **kwargs):
        """Initialize a TruncateFormatter."""
        super().__init__(**kwargs)
        self.max_length = max_length
        self.log_file = log_file

    def format(self, record):  # noqa: A003
        """Format the record. Truncates long messages if not DEBUG level."""
        msg = super().format(record)
        if len(msg) > self.max_length and logging.getLogger().level > logging.DEBUG:
            return f"""{msg[:self.max_length]}...\nOutput truncated.
To see full output, run with -vv or check {self.log_file}\n"""
        return msg


def setup_logger(
        name: str,
        level: int=logging.INFO,
        log_file: str='dev.log',
    ) -> logging.Logger:
    """Setup logger to use across the program."""
    logger = logging.Logger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False


    # Handle stream output by truncating long messages.
    # Only prints messages with no additional info.
    shandler = logging.StreamHandler()
    shandler.setFormatter(TruncateFormatter(fmt='%(message)s', log_file=log_file))
    shandler.setLevel(level)

    # Create a handler for putting info into a .log file. Includes time stamps etc.
    # Will write EVERYTHING to the log (i.e. level = debug)
    fhandler = logging.FileHandler(log_file)
    fhandler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )

    logger.addHandler(shandler)
    logger.addHandler(fhandler)
    return logger


def sech(z: npt.ArrayLike | Number) -> npt.ArrayLike | Number:
    """Hyperbolic Secant."""
    return 1.0 / np.cosh(np.asanyarray(z))

def import_lumapi(loc: str):
    """Import the Lumerical python api from a specified location."""
    lumapi = object()
    try:
        spec_lin = imp.spec_from_file_location('lumapi', loc)

        # These assertions are so MyPy doesn't get mad
        assert spec_lin is not None
        assert isinstance(spec_lin.loader, Loader)

        lumapi = imp.module_from_spec(spec_lin)
        spec_lin.loader.exec_module(lumapi)
    except FileNotFoundError:
        logging.exception('lumapi not found. Using dummy values\n')
    return lumapi


def read_config_file(filename: str | Path, cfg_format: str='auto') -> dict[str, Any]:
    """Read a configuration file."""
    filename = ensure_path(filename)
    if cfg_format == 'auto':
        cfg_format = filename.suffix

    config_loader = _get_config_loader(cfg_format)
    return config_loader(filename)


def _get_config_loader(config_format: str) -> Callable[[Path], dict]:
    """Return a configuration file loader depending on the format."""
    match config_format.lower():
        case '.yaml' | '.yml':
            return _yaml_loader
        case _:
            msg = f'{config_format} file loading not yet supported.'
            raise NotImplementedError(msg)


def _yaml_loader(filepath: Path) -> dict:
    """Config file loader for YAML files."""
    # Allow the safeloader to convert sequences to tuples
    SafeConstructor.add_constructor(  # type: ignore
        'tag:yaml.org,2002:python/tuple',
        lambda self, x: tuple(SafeConstructor.construct_sequence(self, x))
    )
    with open(filepath, 'rb') as stream:
        return yaml.safe_load(stream)


def ensure_path(path: str | Path) -> Path:
    """Ensure that a Path is a Path object."""
    if not isinstance(path, Path):
        return Path(path)
    return path
