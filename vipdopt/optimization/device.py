"""Module for Device class and any subclasses."""

from __future__ import annotations

import sys
from copy import copy
from numbers import Number, Rational, Real
from pathlib import Path

import numpy as np
import numpy.typing as npt

from vipdopt.optimization.filter import Filter
from vipdopt.utils import PathLike, ensure_path

CONTROL_AVERAGE_PERMITTIVITY = 3
GAUSSIAN_SCALE = 0.27

def index_from_permittivity(permittivity_):
	'''Checks all permittivity values are real and then takes square root to give index.'''
	assert np.all(np.imag(permittivity_) == 0), 'Not expecting complex index values right now!'

	return np.sqrt(permittivity_)

# TODO: Add `feature_dimensions` for allowing z layers to have thickness otehr than 1
class Device:
    """An optical device / object that can be optimized.

    Attributes:
        size (tuple[int, int]): The (x, y) dimensions of the device
            region in voxels.
        permittivity_constraints (tuple[Rational, Rational]): Min and max permittivity.
        coords (tuple[Rational, Rational, Rational]): The 3D coordinates in space of
            the device.
        name (str): Name of the device; defaults to 'device'.
        init_density (float): The initial density to use in the device; defaults to 0.5.
        randomize (bool): Whether to initialize with random values; Defaults to False.
        init_seed (None | int): The random seed to use when initializing.
        symmetric (bool): Whether to initialize with symmetric design vairables;
            defaults to False. Does nothing if `randomize` is False.
        filters (list[Filter]): Filters to initialize the device with.
        w (npt.NDArray[np.complex128]): The w variable for the device;
            has shape equal to size x (len(filters) + 1).
    """

    def __init__(
            self,
            size: tuple[int, int] ,
            permittivity_constraints: tuple[Real, Real],
            coords: tuple[Rational, Rational, Rational],
            name: str='device',
            init_density: float=0.5,
            randomize: bool=False,
            init_seed: None | int=None,
            symmetric: bool=False,
            filters: list[Filter] | None=None,
            **kwargs
    ):
        """Initialize Device object."""
        if filters is None:
            filters = []
        if len(size) != 2:  # noqa: PLR2004
            raise ValueError(f'Device size must be 2 dimensional; got {len(size)}')
        if any(d < 1 for d in size) or any(not isinstance(d, int) for d in size):
            raise ValueError('Expected positive, integer dimensions;'
                             f' received {size}')
        self.size = size

        if len(permittivity_constraints) != 2:  # noqa: PLR2004
            raise ValueError('Expected 2 permittivity constraints;'
                             f'got {permittivity_constraints}')
        if any(not isinstance(pc, Real) for pc in permittivity_constraints):
            raise ValueError('Expected real permittivity;'
                             f' got {permittivity_constraints}')
        if permittivity_constraints[0] >= permittivity_constraints[1]:
            raise ValueError('Maximum permittivity must be greater than minimum')
        self.permittivity_constraints = permittivity_constraints

        if len(coords) != 3:  # noqa: PLR2004
            raise ValueError('Expected device coordinates to be 3 dimensional; '
                             f'got {len(coords)}')
        if any(not isinstance(coord, Rational) for coord in coords):
            raise ValueError('Expected device coordinates to be rational;'
                             f' got {coords}')
        self.coords = coords

        # Optional arguments
        self.name = name
        self.init_density = init_density
        self.randomize = randomize
        self.init_seed = init_seed
        self.symmetric = symmetric
        self.filters = filters
        vars(self).update(kwargs)

        self._init_variables()
        self.update_density()

    def __eq__(self, __value: object) -> bool:
        """Test equality."""
        if isinstance(__value, Device):
            return np.array_equal(self.w, __value.w)
        return super().__eq__(__value)

    def _init_variables(self):
        """Creates the w variable for the device.

        For each variable of the device, creates an array with all values representing
        density. Each subsequent layer represents the density after being passed
        through the corresponding number of filters. The first entry / array is the
        design variable, and the last entry / array is the updated density.
        """
        n_var = self.num_filters() + 1

        w = np.zeros((*self.size, n_var), dtype=np.complex128)
        if self.randomize:
            rng = np.random.default_rng(self.init_seed)
            w[:] = rng.normal(self.init_density, GAUSSIAN_SCALE, size=w.shape)

            if self.symmetric:
                w[..., 0] = np.tril(w[..., 0]) + np.triu(w[..., 0].T, 1)
                w[..., 0] = np.flip(w[..., 0], axis=1)

        w[..., 0] = self.init_density * np.ones(self.size, dtype=np.complex128)
        self.w = w

    def as_dict(self) -> dict:
        """Return a dictionary representation of this device, sans `self.w`."""
        data = copy(vars(self))

        # Design variable is stored separately
        del data['w']

        # Add all filters
        filters: list[Filter] = []
        for filt in data.pop('filters'):
            filt_dict = {}
            filt_dict['type'] = type(filt).__name__
            filt_dict['parameters'] = filt.init_vars
            filters.append(filt_dict)
        data['filters'] = filters

        return data

    def load_dict(self, device_data: dict):
        """Load attributes from a dictionary."""
        vars(self).update(vars(Device._from_dict(device_data)))

    @classmethod
    def _from_dict(cls, device_data: dict) -> Device:
        """Create a new device from a dictionary."""
        data = copy(device_data)
        filters: list[Filter] = []
        filt_dict: dict
        for filt_dict in data.pop('filters'):
            filter_cls: type[Filter] = getattr(
                sys.modules['vipdopt.optimization.filter'],
                filt_dict['type'],
            )
            filters.append(filter_cls(
                **filt_dict['parameters'],
            ))
        data['filters'] = filters
        return cls(**data)

    @ensure_path
    def save(self, fname: Path, binarize: bool=False):
        """Save device to a .npz file."""
        with fname.open('wb') as f:
            np.save(f, self.as_dict())
            if binarize:
                np.save(f, self.pass_through_filters(self.get_design_variable(), True))
            else:
                np.save(f, self.get_design_variable())

    @classmethod
    def from_source(cls, source: dict | PathLike) -> Device:
        """Create a new device from a dictionary or load a saved .npy file."""
        if isinstance(source, dict):
            return Device._from_dict(source)
        return Device._from_file(source)

    @ensure_path
    def load_file(self, fname: Path):
        """Load device from a .npz file."""
        with fname.open('rb') as f:
            attributes = np.load(f, allow_pickle=True).flat[0]
            w = np.load(f)
        self.load_dict(attributes)
        self.set_design_variable(w)

    @classmethod
    @ensure_path
    def _from_file(cls, fname: Path) -> Device:
        """Create a new device by loading from a saved file."""
        with fname.open('rb') as f:
            attributes = np.load(f, allow_pickle=True).flat[0]
            w = np.load(f)
        d = Device._from_dict(attributes)
        d.set_design_variable(w)
        return d

    def get_design_variable(self) -> npt.NDArray[np.complex128]:
        """Return the design variable of the device region (i.e. first layer)."""
        return self.w[..., 0]

    def set_design_variable(self, value: npt.NDArray[np.complex128]):
        """Set the first layer of the device region to specifieed values."""
        self.w[..., 0] = value.copy()
        self.update_density()

    def num_filters(self):
        """Return the number of filters in this device."""
        return len(self.filters)

    def get_density(self):
        """Return the density of the device region (i.e. last layer)."""
        return self.w[..., -1]

    def get_permittivity(self) -> npt.NDArray[np.complex128]:
        """Return the permittivity of the design variable."""
        eps_min, eps_max = self.permittivity_constraints
        return self.get_density() * (eps_max - eps_min) + eps_min

    def update_density(self):
        """Pass each layer of density through the devices filters."""
        for i in range(self.num_filters()):
            var_in = self.w[..., i]
            var_out = self.filters[i].forward(var_in)
            self.w[..., i + 1] = var_out

    def pass_through_filters(
            self,
            x: npt.NDArray | Number,
            binarize=False,
    ) -> npt.NDArray | Number:
        """Pass x through the device filters, returning the output.

        Does NOT modify the original x, but instead operates on a copy.

        Attributes:
            x: (npt.NDArray | Number): The input density to pass through the filters.
            binarize (bool): Whether to binarize values.


        Returns:
            (npt.NDArray | Number): The density after passing through filters.
        """
        y = np.copy(np.array(x))
        for i in range(self.num_filters()):
            y = self.filters[i].fabricate(y) if binarize \
                else self.filters[i].forward(y)  # type: ignore
        return y

    #! TODO: UPDATE_FILTERS() METHOD

    def backpropagate(self, gradient):
        """Backpropagate a gradient to be applied to pre-filtered design variables."""
        grad = np.copy(gradient)
        for i in range(self.num_filters() - 1, -1, -1):
            filt = self.filters[i]
            y = self.w[..., i]
            x = self.w[..., i - 1]

            grad = filt.chain_rule(grad, y, x)

        return grad
