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

def arb_broadcast(x,y):
    '''Broadcasts x to new axes depending on the shape of y so that we can get element-wise xy.'''
    # https://stackoverflow.com/a/69419989
    a = x.shape
    b = y.shape
    broadcast_idxs = [idx for idx in range(len(b)) if b[idx] not in a]
    return np.broadcast_to(np.expand_dims(x, broadcast_idxs), b)

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

def cross_product(A,B, axis=0):
    # Simple function for the cross product of two vector fields in one plane

    def gs(component_idx):
        _axis = axis % len(A.shape)
        '''Gets the component of the vector at axis given above.
        Also handles negative indices for axis.'''
        return (slice(None),) * _axis + (slice(component_idx, component_idx+1),)

    result = dict()
    result['x'] = np.multiply(A[gs(1)],B[gs(2)]) - np.multiply(A[gs(2)],B[gs(1)])
    result['y'] = -(np.multiply(A[gs(0)],B[gs(2)]) - np.multiply(A[gs(2)],B[gs(0)]))
    result['z'] = np.multiply(A[gs(0)],B[gs(1)]) - np.multiply(A[gs(1)],B[gs(0)])

    # result['x'] = np.multiply(A[1,...],B[2,...]) - np.multiply(A[2,...],B[1,...])
    # result['y'] = -(np.multiply(A[0,...],B[2,...]) - np.multiply(A[2,...],B[0,...]))
    # result['z'] = np.multiply(A[0,...],B[1,...]) - np.multiply(A[1,...],B[0,...])

    return result

def dot_product(A,B, axis=0):
    # Simple function for the dot product of two vector fields
    # Ef = np.squeeze(A)
    # Eb = np.squeeze(B)
    
    def einsum_string(_axis):
        _axis += 1
        from string import ascii_lowercase
        pre_string = ascii_lowercase[:_axis]+'...,'+ascii_lowercase[:_axis]+'...'
        post_string = ascii_lowercase[:_axis-1] +'...'
        return pre_string + '->' + post_string
    
    result = np.einsum(einsum_string(axis), A, B)
    #result = np.multiply(A[0,...], B[0,...]) + np.multiply(A[1,...], B[1,...]) + np.multiply(A[2,...], B[2,...])

    return result

def rescale(input, min_output=0, max_output=1):
    '''Rescales image values to 0 and 1.'''
    output = (input-np.min(input))/(np.max(input)-np.min(input))
    return (max_output - min_output)*(min_output + output)

#* Plotting

WL_TO_COLOR_MAP = [[6, 1, 31],[12, 0, 40],[14, 0, 51], [16, 1, 60],
 [17, 1, 76], [23, 0, 90], [26, 1, 105], [28, 0, 119], [28, 0, 136], 
[34, 0, 151],[36, 1, 165], [37, 0, 176], [37, 1, 187], [36, 0, 194],
[37, 0, 202], [34, 0, 209], [31, 0, 217], [28, 1, 220], [25, 0, 224],
[18, 1, 227], [16, 0, 229], [14, 0, 233], [10, 0, 237], [9, 0, 237],
[7, 0, 240], [3, 0, 242], [0, 0, 244], [0, 0, 244], [2, 5, 244], 
[1, 8, 244], [0, 13, 242], [0, 18, 242], [2, 22, 239],
[0, 28, 236], [0, 33, 236], [0, 37, 232], [0, 44, 229], [2, 49, 227],
[0, 55, 220], [0, 60, 218], [0, 66, 214], [1, 73, 209], [0, 77, 205],
[0, 84, 200], [0, 91, 194], [0, 96, 193], [0, 101, 189], [0, 106, 182],
[0, 111, 177], [1, 118, 172], [0, 120, 165], [0, 122, 159], 
[0, 128, 153], [1, 131, 147], [1, 132, 140], [1, 135, 134], 
[0, 140, 131], [0, 145, 126], [0, 148, 124], [0, 152, 122], 
[0, 158, 118], [1, 162, 118], [1, 168, 116], [0, 172, 114],
[0, 178, 113],[0, 182, 112], [0, 186, 111], [1, 188, 109], 
[2, 191, 107], [0, 194, 107], [1, 195, 108], [0, 198, 101], 
[1, 200, 99], [0, 204, 96], [1, 209, 97], [2, 211, 94], [1, 217, 90],
[0, 220, 88], [0, 225, 81], [1, 228, 77], [1, 231, 71], [1, 232, 68], 
[0, 230, 60], [0, 230, 52], [0, 230, 43], [0, 230, 33], [0, 228, 21],
[0, 228, 11], [2, 229, 0], [16, 229, 0], [28, 229, 0], [40, 230, 0], 
[56, 232, 0], [72, 232, 2], [84, 230, 1],
[98, 231, 0],[111, 230, 0],[124, 230, 0], [137, 230, 1], [151, 228, 0],
[162, 227, 0], [173, 229, 0], [186, 227, 0], [198, 224, 1], 
[211, 226, 0], [221, 221, 0], [227, 216, 0], [230, 210, 1], 
[237, 201, 1], [240, 193, 1],[242, 184, 0], [245, 173, 0], 
[248, 165, 1], [250, 155, 0], [251, 145, 0], [252, 136, 1], 
[254, 126, 1], [255, 115, 0], [255, 104, 3], [254, 95, 1], [255, 83, 1],
[255, 72, 2], [255, 61, 0], [253, 49, 0], [255, 39, 2],
[253, 28, 0], [255, 17, 4], [255, 8, 1], [254, 2, 1], [254, 0, 10],
[255, 0, 14], [255, 0, 18], [251, 0, 24], [250, 0, 30], [250, 0, 30], 
[248, 0, 35], [246, 0, 41], [246, 0, 41], [242, 0, 40], [242, 0, 40],
[240, 0, 45], [237, 0, 46], [233, 0, 45], [230, 1, 44], [226, 0, 42], 
[222, 0, 41], [218, 0, 39], [214, 0, 38], [206, 0, 36], [200, 1, 34], 
[195, 0, 32], [189, 0, 30], [185, 0, 31], [177, 0, 28], [169, 0, 26],
[162, 0, 24], [152, 0, 23],[144, 1, 21], [136, 1, 18], [128, 1, 20], 
[121, 0, 19], [111, 0, 16], [104, 0, 14], [96, 0, 12], [88, 1, 10], 
[83, 0, 12], [73, 0, 9], [67, 0, 9], [62, 1, 9], [57, 0, 7], [51, 0, 7],
[46, 0, 5], [42, 0, 4], [39, 0, 5], [33, 1, 4], [30, 0, 4],[25, 0, 3], 
[25, 0, 3], [22, 0, 2],[21, 0, 1], [16, 0, 0], [15, 1, 1], [14, 0, 0], 
[12, 0, 0], [9, 0, 1], [9, 0, 1], [8, 0, 0]]


def wl_to_rgb(wl:float) -> npt.NDArray:
    """Input : a float describing a wavelength in nanometers
    Output : a numpy array giving the rgb values (between 0 and 1) 
    associated with the colour percieved at this wavelength
    We just use hardcoded values."""
    a = np.linspace(400, 700, len(WL_TO_COLOR_MAP))
    colorindex = min(range(len(a)), key=lambda i: abs(a[i]-wl))
    col = WL_TO_COLOR_MAP[colorindex]
    return np.asarray(col)/255