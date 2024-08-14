"""Module containing common functions used throughout the package."""

from __future__ import annotations

import functools
import operator
import importlib.util as imp
import itertools
import json
import logging
import os
import threading
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping
from importlib.abc import Loader
from numbers import Number
from pathlib import Path
from types import ModuleType
from typing import Any, Concatenate, ParamSpec, TypeAlias, TypedDict, TypeVar

import numpy as np
import numpy.typing as npt
import yaml
from yaml.constructor import SafeConstructor
from scipy import ndimage

PathLike = TypeVar('PathLike', str, Path, bytes, os.PathLike)

# Generic types for type hints
T = TypeVar('T')
R = TypeVar('R')

Nested: TypeAlias = T | Iterable['Nested[T]']

P = ParamSpec('P')
Q = ParamSpec('Q')


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


def sech(z: npt.ArrayLike | Number) -> npt.ArrayLike | Number:
    """Hyperbolic Secant."""
    return 1.0 / np.cosh(np.asanyarray(z))


def import_lumapi(loc: str) -> ModuleType:
    """Import the Lumerical python api from a specified location."""
    lumapi = ModuleType('lumapi')

    spec_lin = imp.spec_from_file_location('lumapi', loc)

    # These assertions are so MyPy doesn't get mad
    assert spec_lin is not None
    assert isinstance(spec_lin.loader, Loader)

    lumapi = imp.module_from_spec(spec_lin)
    spec_lin.loader.exec_module(lumapi)

    return lumapi


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
    


def subclasses(cls: type) -> list[str]:
    """Get all subclasses of a given class."""
    return [scls.__name__ for scls in cls.__subclasses__()]


def ladd_to_all(iterable: Iterable[T], val: T) -> Generator[T, None, None]:
    """Add a value to all memebers of a list on the left."""
    for item in iterable:
        yield val + item


def radd_to_all(iterable: Iterable[T], val: T) -> Generator[T, None, None]:
    """Add a value to all memebers of a list on the right."""
    for item in iterable:
        yield item + val


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


class StoppableThread(threading.Thread):
    """Thread class with a stop() method.

    The thread itself has to check regularly for the stopped() condition.
    """

    def __init__(self, *args, **kwargs):
        """Initialize a StoppableThread."""
        super().__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        """Stop this thread."""
        self._stop_event.set()

    def stopped(self):
        """Return whther this thread has been stopped."""
        return self._stop_event.is_set()


class Coordinates(TypedDict):
    """Class representing coordinates in 3D space."""

    x: npt.NDArray
    y: npt.NDArray
    z: npt.NDArray


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


def flatten(data: Nested[T]) -> Iterable[T]:
    """Return a copy of the data as single, collapsed iterator."""
    if isinstance(data, Iterable):
        for x in data:
            yield from flatten(x)
    else:
        yield data


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


def repeat(a: npt.NDArray, shape: tuple[int, ...]) -> npt.NDArray:
    """Apply numpy's `repeat` function along multiple axes.

    Solution from https://stackoverflow.com/questions/7656665/how-to-repeat-elements-of-an-array-along-two-axes
    """
    return np.kron(a, np.ones(shape, dtype=a.dtype))

def real_part_complex_product(z1, z2):
    """Explanation: For two complex numbers, Re(z1*z2) = Re(z1)*Re(z2) + [-Im(z1)]*Im(z2)"""
    return np.real(z1)*np.real(z2) + np.imag(z1) * (-1*np.imag(z2))

# Borders, Masking, and Padding
def pad_all_sides(arr:npt.NDArray, pad_width:int=2, constant:float=0) -> npt.NDArray:
    return np.pad(arr,(pad_width,),'constant',constant_values=(constant,))

def unpad(arr: npt.NDArray, pad_widths: tuple[tuple[int,int],...]) -> npt.NDArray:
    b = arr.copy()
    slice_list = [slice(None)]*b.ndim
    for i in range(b.ndim):
        sz = b.shape[i]
        slice_list[i] = slice(pad_widths[i][0], sz - pad_widths[i][1])
    return b[tuple(slice_list)]

def unpad_all_sides(arr: npt.NDArray, pad_width:int=2):
    pad_widths = tuple([(pad_width, pad_width) for i in range(arr.ndim)])
    return unpad(arr, pad_widths)

def replace_border(arr: npt.NDArray, pad_width:int=2, const:float=0, only_sides:bool=False) -> npt.NDArray:
    if arr.ndim == 3 and only_sides:
        # split it into 2d layers and replace borders for those instead
        new_arr = arr.copy()
        for i in range(arr.shape[3-1]):
            new_arr[...,i] = pad_all_sides(unpad_all_sides(arr[...,i], pad_width),
                                            pad_width, constant=const)
        return new_arr
    else:
        return pad_all_sides(unpad_all_sides(arr, pad_width),
                        pad_width, constant=const)

def create_centered_square_mask(arr: npt.NDArray, width):
    coord_axes = np.ogrid[[slice(-x//2+1,x//2+1,1) for x in arr.shape]]
    full_coord_grid = sum(coord_axes)       # NOT NP.SUM
    # NOTE: Will be off-center for arrays of odd shape-length.
    mask = np.abs(coord_axes[0]) <= width
    for i in range(1,len(coord_axes)):
        mask = mask & (np.abs(coord_axes[i]) <= width)
    return mask

def create_centered_square_ring_mask(arr: npt.NDArray, outer_width, inner_width):
    # NOTE: Will be off-center for arrays of odd shape-length.
    mask_outer = create_centered_square_mask(arr, outer_width)
    mask_inner = create_centered_square_mask(arr, inner_width)
    return mask_outer ^ mask_inner

def create_centered_circle_mask(arr: npt.NDArray, radius):
    coord_axes = np.ogrid[[slice(-x//2+1,x//2+1,1) for x in arr.shape]]
    full_coord_grid = sum(coord_axes)       # NOT NP.SUM
    # NOTE: Will be off-center for arrays of odd shape-length.
    mask = sum([x**2 for x in coord_axes]) <= radius**2
    return mask

def get_mask_edge(mask, thickness=1):
    '''Uses a binary structure to erode an existing mask, then takes the XOR to produce the edge.
    Higher thicknesses mean more iterations of the binary erosion.'''
    # https://stackoverflow.com/questions/33742098/border-edge-operations-on-numpy-arrays
    
    struct = ndimage.generate_binary_structure(mask.ndim, mask.ndim)
    erode = ndimage.binary_erosion(mask, struct, iterations=thickness)
    edges = mask ^ erode
    return edges

def replace_mask_values(arr, mk, value):
    '''Applies a mask (mk) to an array, then replaces masked values with a specific fill value.'''
    # https://numpy.org/doc/stable/reference/maskedarray.generic.html#accessing-the-data
    import numpy.ma as ma
    maskArr = ma.masked_array(arr,mask=mk)
    return ma.filled(maskArr, fill_value=value)
    
# Nested dictionary handling - https://stackoverflow.com/a/14692747
def get_by_path(root, items):
    """Access a nested object in root by item sequence."""
    return functools.reduce(operator.getitem, items, root)


def set_by_path(root, items, value):
    """Set a value in a nested object in root by item sequence."""
    get_by_path(root, items[:-1])[items[-1]] = value


def del_by_path(root, items):
    """Delete a key-value in a nested object in root by item sequence."""
    del get_by_path(root, items[:-1])[items[-1]]