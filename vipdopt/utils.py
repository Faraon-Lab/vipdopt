import os
from pathlib import Path
import logging
import functools
import itertools
import json
import yaml
from yaml.constructor import SafeConstructor
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping
from typing import Any, Concatenate, ParamSpec, TypeAlias, TypedDict, TypeVar
import numpy as np
import numpy.typing as npt

#* Type Hints - Generic Types
from numbers import Number
PathLike = TypeVar('PathLike', str, Path, bytes, os.PathLike)
T = TypeVar('T')
R = TypeVar('R')
Nested: TypeAlias = T | Iterable['Nested[T]']
P = ParamSpec('P')
Q = ParamSpec('Q')

class Coordinates(TypedDict):
    """Class representing coordinates in 3D space."""

    x: npt.NDArray
    y: npt.NDArray
    z: npt.NDArray


#* Paths and Directories

def convert_path(path: PathLike) -> Path:
    """Ensure that a Path is a Path object."""
    if isinstance(path, Path):
        return path
    if isinstance(path, bytes):
        return Path(path.decode())
    if isinstance(path, str | os.PathLike):
        return Path(path)

    # Input was not a PathLike
    raise ValueError(f'Argument must be PathLike, got {type(path)}')


def ensure_path(
    func: Callable[Concatenate[Any, Path, P], R],
) -> Callable[Concatenate[Any, PathLike, P], R]:
    """Function decorator for converting PathLike's to Path's."""

    @functools.wraps(func)
    def wrapper(
        arg0: Any,
        path: PathLike,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        return func(arg0, convert_path(path), *args, **kwargs)

    return wrapper

def split_glob(pattern: str) -> Generator[str, None, None]:
    """Convert a glob pattern into multiple patterns if necessary."""
    if '{' in pattern:
        start = pattern.index('{')
        end = pattern.index('}')

        prefix = pattern[:start]
        choices = pattern[start + 1 : end].split(',')
        for choice in choices:
            yield from ladd_to_all(
                split_glob(pattern[end + 1 :]), prefix + choice.strip()
            )
    else:
        yield pattern

def glob_first(search_dir: PathLike, pattern: str) -> Path:
    """Find the first file to match the given pattern."""
    patterns = list(split_glob(pattern))
    search_path = convert_path(search_dir)
    matches = itertools.chain.from_iterable(map(search_path.glob, patterns))
    try:
        first_match = next(matches)
    except StopIteration as e:
        msg = f'No path found with pattern {pattern} in {search_path}'
        raise FileNotFoundError(msg) from e
    return first_match

def rmtree(path: Path, keep_dir: bool = False):
    """Delete a directory recursively."""
    if not path.is_dir():
        raise ValueError('Path must be a directory')
    for child in path.iterdir():
        if child.is_file():
            child.unlink()
        else:
            rmtree(child, keep_dir=False)

    if not keep_dir:
        path.rmdir()


def read_config_file(fname: PathLike, cfg_format: str = 'auto') -> dict:
    """Read a configuration file."""
    path_filename = convert_path(fname)
    if cfg_format.lower() == 'auto':
        cfg_format = path_filename.suffix

    config_loader = _get_config_loader(cfg_format)
    return config_loader(path_filename)


def _get_config_loader(cfg_format: str) -> Callable[[PathLike], dict]:
    """Return a configuration file loader depending on the format."""
    match cfg_format.lower():
        case '.yaml' | '.yml':
            return _yaml_loader
        case '.json':
            return _json_loader
        case _:
            msg = f'{cfg_format} file loading not yet supported.'
            raise NotImplementedError(msg)


def _json_loader(fname: PathLike) -> dict:
    with open(fname) as stream:
        return json.load(stream)


def _yaml_loader(fname: PathLike) -> dict:
    """Config file loader for YAML files."""

    # Allow the safeloader to convert sequences to tuples
    def new_constructor(self, x: yaml.SequenceNode):
        return tuple(SafeConstructor.construct_sequence(self, x))

    SafeConstructor.add_constructor(  # type: ignore
        'tag:yaml.org,2002:python/tuple', new_constructor
    )
    
    with open(fname, 'rb') as stream:
        return yaml.safe_load(stream)


#* Logging

class TruncateFormatter(logging.Formatter):
    """Logging formatter for truncating large output."""

    def __init__(
        self,
        max_length: int = 300,
        log_file: str = 'dev.log',
        level: int = logging.WARNING,
        **kwargs,
    ):
        """Initialize a TruncateFormatter."""
        super().__init__(**kwargs)
        self.max_length = max_length
        self.log_file = log_file
        self.level = level

    def format(self, record):
        """Format the record. Truncates long messages if not DEBUG level."""
        msg = super().format(record)
        if len(msg) > self.max_length and self.level > logging.DEBUG:
            return f"""{msg[: self.max_length]}...\nOutput truncated.
To see full output, run with -vv or check {self.log_file}\n"""
        return msg

def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: str = 'dev.log',
) -> logging.Logger:
    """Setup logger to use across the program."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    Path(log_file).touch()

    # Reset logger
    for _ in range(len(logger.handlers)):
        logger.removeHandler(logger.handlers[0])

    # Handle stream output by truncating long messages.
    # Only prints messages with no additional info.
    shandler = logging.StreamHandler()
    shandler.setFormatter(
        TruncateFormatter(
            fmt='%(message)s',
            log_file=log_file,
            level=level,
        )
    )
    shandler.setLevel(level)

    # Create a handler for putting info into a .log file. Includes time stamps etc.
    # Will write EVERYTHING to the log (i.e. level = debug)
    fhandler = logging.FileHandler(log_file, mode='a')
    fhandler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )

    logger.addHandler(shandler)
    logger.addHandler(fhandler)

    return logger

#* Lists, Dicts, and Iterables

def ladd_to_all(iterable: Iterable[T], val: T) -> Generator[T, None, None]:
    """Add a value to all memebers of a list on the left."""
    for item in iterable:
        yield val + item


def radd_to_all(iterable: Iterable[T], val: T) -> Generator[T, None, None]:
    """Add a value to all memebers of a list on the right."""
    for item in iterable:
        yield item + val

def flatten(data: Nested[T]) -> Iterable[T]:
    """Return a copy of the data as single, collapsed iterator."""
    if isinstance(data, Iterable):
        for x in data:
            yield from flatten(x)
    else:
        yield data

#* Function Tools

def starmap_with_kwargs(
    function: Callable[P, R],
    args_iter: Iterable[Iterable],
    kwargs_iter: Iterable[Mapping],
) -> Iterator[R]:
    """Wrapper around itertools.starmap that can take kwargs."""
    args_for_starmap = zip(itertools.repeat(function), args_iter, kwargs_iter)
    return itertools.starmap(apply_args_and_kwargs, args_for_starmap)


def apply_args_and_kwargs(function: Callable[P, R], args: tuple, kwargs: dict) -> R:
    """Call a function with the provided args and kwargs."""
    return function(*args, **kwargs)

#* Math

def sech(z: npt.ArrayLike | Number) -> npt.ArrayLike | Number:
    """Hyperbolic Secant."""
    return 1.0 / np.cosh(np.asanyarray(z))

def real_part_complex_product(z1, z2):
    """Explanation: For two complex numbers, Re(z1*z2) = Re(z1)*Re(z2) + [-Im(z1)]*Im(z2)"""
    return np.real(z1)*np.real(z2) + np.imag(z1) * (-1*np.imag(z2))