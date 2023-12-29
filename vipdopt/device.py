"""Module for Device class and any subclasses."""

import pickle
from numbers import Number, Rational, Real

import numpy as np
import numpy.typing as npt

from vipdopt.filter import Filter
from vipdopt.utils import PathLike, ensure_path

CONTROL_AVERAGE_PERMITTIVITY = 3

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
        init_seed (None | int): The random seed to use when initializing; defaults to
            None, in which case all design variable values will be init_density.
        iteration (int): The current iteration in optimization; defaults to 0.
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
            init_seed: None | int=None,
            iteration: int=0,
            filters: list[Filter] | None=None,
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
        self.init_seed = init_seed
        self.iteration = iteration
        self.filters = filters

        self._init_variables()

    def _init_variables(self):
        """Creates the w variable for the device.

        For each variable of the device, creates an array with all values representing
        density. Each subsequent layer represents the density after being passed
        through the corresponding number of filters. The first entry / array is the
        design variable, and the last entry / array is the updated density.
        """
        n_var = len(self.filters) + 1

        w = np.zeros((*self.size, n_var), dtype=np.complex128)
        w[..., 0] = self.init_density * np.ones(self.size, dtype=np.complex128)
        self.w = w

    # TODO Figure out how to save device since pickling numpy arrays is bad. MyPy is mad
    def save_device(self, folder: PathLike):
        """Save device as pickled bytes.

        Attributes:
            folder (PathLike): The folder to save the device to. Anything after the
                directory name will be stripped. Defaults to root directory.
        """
        raise NotImplementedError
        path_folder = ensure_path(folder).parent
        data = vars(self)
        with (path_folder / self.name / '.pkl').open('wb') as f:
            pickle.dump(data, f)

    def load_device(self, fname: PathLike):
        """Load a device from a pickled file."""
        raise NotImplementedError
        fpath = ensure_path(fname)
        with fpath.open('rb') as f:
            vars(self).update(pickle.load(f))

    def get_design_variable(self) -> npt.NDArray[np.complex128]:
        """Return the design variable of the device region (i.e. first layer)."""
        return self.w[..., 0]

    def set_design_variable(self, value: npt.NDArray[np.complex128]):
        """Set the first layer of the device region to specifieed values."""
        self.w[..., 0] = value.copy()
        self.update_density()

    def get_density(self):
        """Return the density of the device region (i.e. last layer)."""
        return self.w[..., -1]

    def get_permittivity(self) -> npt.NDArray[np.complex128]:
        """Return the permittivity of the design variable."""
        eps_min, eps_max = self.permittivity_constraints
        return self.get_density() * (eps_max - eps_min) + eps_min

    def update_density(self):
        """Pass each layer of density through the devices filters."""
        for i in range(len(self.filters)):
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
            (npt.NDArray | Number): The density after passing theough filters.
        """
        y = np.copy(x)
        for i in range(len(self.filters)):
            y = self.filters[i].fabricate(y) if binarize else self.filters[i].forward(y)
        return y




