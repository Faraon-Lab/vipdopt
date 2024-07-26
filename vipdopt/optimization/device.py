"""Module for Device class and any subclasses."""

from __future__ import annotations

import sys
from collections.abc import Iterable
from copy import copy
from numbers import Real
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
from scipy import interpolate

from vipdopt import STL, GDS
from vipdopt.optimization.filter import Filter, Scale, Sigmoid
from vipdopt.simulation import Import
from vipdopt.utils import Coordinates, PathLike, ensure_path, repeat

CONTROL_AVERAGE_PERMITTIVITY = 3
GAUSSIAN_SCALE = 0.27
REINTERPOLATION_SIZE = (300, 300, 306)


# TODO: Add `feature_dimensions` for allowing z layers to have thickness otehr than 1
class Device:
    """An optical device / object that can be optimized.

    Attributes:
        size (tuple[int, int]): The (x, y) dimensions of the device
            region in voxels.
        permittivity_constraints (tuple[Rational, Rational]): Min and max permittivity.
        coords (Coordinates): The 3D coordinates in space of the device.
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
        size: tuple[int, int, int],  # ! 20240227 Ian - Added second device
        permittivity_constraints: tuple[Real, Real],
        coords: Coordinates,
        name: str = 'device',
        init_density: float = 0.5,
        randomize: bool = False,
        init_seed: None | int = None,
        symmetric: bool = False,
        filters: list[Filter] | None = None,
        **kwargs,
    ):
        """Initialize Device object."""
        if len(size) != 3:  # noqa: PLR2004
            raise ValueError(f'Device size must be 3 dimensional; got {len(size)}')
        if any(d < 1 for d in size) or any(not isinstance(d, int) for d in size):
            raise ValueError(
                'Expected positive, integer dimensions;' f' received {size}'
            )
        self.size = size

        if len(permittivity_constraints) != 2:  # noqa: PLR2004
            raise ValueError(
                'Expected 2 permittivity constraints;' f'got {permittivity_constraints}'
            )
        if any(not isinstance(pc, Real) for pc in permittivity_constraints):
            raise ValueError(
                'Expected real permittivity;' f' got {permittivity_constraints}'
            )
        if permittivity_constraints[0] >= permittivity_constraints[1]:
            raise ValueError('Maximum permittivity must be greater than minimum')
        self.permittivity_constraints = permittivity_constraints

        # Make sure there is always a Scale filter at the end
        if filters is None:
            filters = [Scale(permittivity_constraints)]
        elif not isinstance(filters[-1], Scale):
            filters.append(Scale(permittivity_constraints))

        if (
            not isinstance(coords, dict)
            or any(c not in coords for c in 'xyz')
            or any(k not in 'xyz' for k in coords)
        ):
            raise TypeError(
                'Expected device coordinates to be a dictionary with '
                f'entries {{x, y, z}}; got {coords}'
            )
        if any(not isinstance(coord, np.ndarray) for coord in coords.values()):
            raise TypeError(
                'Expected device coordinates to be ndarrays;' f' got {coords}'
            )
        self.coords = coords

        # Optional arguments
        self.name = name
        self.init_density = init_density
        self.randomize = randomize
        self.init_seed = init_seed
        self.symmetric = symmetric
        self.filters = filters
        vars(self).update(kwargs)

        # Give default value for the shape of the field.
        self.field_shape: tuple[int, int, int] = self.size

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
        if self.randomize:  # ! 20240228 Ian - Fixed randomize
            rng = np.random.default_rng(self.init_seed)
            w[:] = rng.normal(self.init_density, GAUSSIAN_SCALE, size=w.shape)

            if self.symmetric:
                w[..., 0, 0] = np.tril(w[..., 0, 0]) + np.triu(w[..., 0, 0].T, 1)
                w[..., 0, 0] = np.flip(w[..., 0, 0], axis=1)

            w[..., 0] = np.maximum(np.minimum(w[..., 0], 1), 0)
        else:
            w[..., 0] = self.init_density * np.ones(self.size, dtype=np.complex128)
        self.w = w

    def as_dict(self) -> dict:
        """Return a dictionary representation of this device, sans `self.w`."""
        data = copy(vars(self))

        # Design variable is stored separately
        del data['w']

        del data['coords']
        data['coords'] = {
            k: list(v) for k, v in cast(Iterable[npt.NDArray], self.coords.items())
        }

        # Add all filters
        filters: list[dict] = []
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
        data['coords'] = {k: np.array(v) for k, v in data['coords'].items()}
        filters: list[Filter] = []
        filt_dict: dict
        for filt_dict in data.pop('filters'):
            filter_cls: type[Filter] = getattr(
                sys.modules['vipdopt.optimization.filter'],
                filt_dict['type'],
            )
            filters.append(
                filter_cls(
                    **filt_dict['parameters'],
                )
            )
        data['filters'] = filters
        return cls(**data)

    @ensure_path
    def save(self, fname: Path, binarize: bool = False):
        """Save device to a .npz file."""
        with fname.open('wb') as f:
            np.save(f, self.as_dict())  # type: ignore
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
        """Set the first layer of the device region to specified values."""
        self.w[..., 0] = value.copy()
        self.update_density()

    def num_filters(self):
        """Return the number of filters in this device."""
        return len(self.filters)

    def update_filters(self, epoch=0):
        """Update the filters of the device."""
        # TODO: Make this more general.
        sigmoid_beta = 0.0625 * (2**epoch)
        sigmoid_eta = 0.5
        self.filters = [
            Sigmoid(sigmoid_eta, sigmoid_beta),
            Scale(self.permittivity_constraints),
        ]

    def get_density(self):
        """Return the density of the device region (i.e. last layer)."""
        return self.w[..., -2]

    def get_permittivity(self) -> npt.NDArray[np.complex128]:
        """Return the permittivity of the design variable."""
        # # Now that there is a Scale filter, the commented-out line below is unnecessary.
        # eps_min, eps_max = self.permittivity_constraints
        # return self.get_density() * (eps_max - eps_min) + eps_min

        return self.w[..., -1]

    def update_density(self):
        """Pass each layer of density through the devices filters."""
        for i in range(self.num_filters()):
            var_in = self.w[..., i]
            var_out = self.filters[i].forward(var_in)
            self.w[..., i + 1] = var_out

    def index_from_permittivity(self, permittivity):
        """Takes square root of permittivty to give the index."""
        return np.sqrt(np.real(permittivity))

    def permittivity_to_density(self, eps, eps_min, eps_max):
        """Convert permittivity to density."""
        return (eps - eps_min) / (eps_max - eps_min)

    def density_to_permittivity(self, density, eps_min, eps_max):
        """Convert density to permittivity."""
        return density * (eps_max - eps_min) + eps_min

    @classmethod
    def binarize(self, variable_in):
        """Assumes density - if not, convert explicitly."""
        return 1.0 * np.greater_equal(variable_in, 0.5)

    def compute_binarization(self, input_variable, set_point=0.5):
        """Compute what percentage of the device is binarized."""
        # TODO: rewrite for multiple materials
        input_variable = np.real(input_variable)
        total_shape = np.prod(input_variable.shape)
        return (2.0 / total_shape) * np.sum(np.sqrt((input_variable - set_point) ** 2))

    def pass_through_filters(
        self,
        x: npt.NDArray | float,
        binarize=False,
    ) -> npt.NDArray | float:
        """Pass x through the device filters, returning the output.

        Does NOT modify the original x, but instead operates on a copy.

        Attributes:
            x: (npt.NDArray | float): The input density to pass through the filters.
            binarize (bool): Whether to binarize values.


        Returns:
            (npt.NDArray | float): The density after passing through filters.
        """
        y = np.copy(np.array(x))
        for i in range(self.num_filters()):
            y = self.filters[i].fabricate(y) if binarize else self.filters[i].forward(y)  # type: ignore
        return y

    def backpropagate(self, gradient):
        """Backpropagate a gradient to be applied to pre-filtered design variables."""
        grad = np.copy(gradient)

        # Previous for-loop just in case
        for f_idx in range(self.num_filters()):
            get_filter = self.filters[self.num_filters() - f_idx - 1]
            variable_out = self.w[..., self.num_filters() - f_idx]
            variable_in = self.w[..., self.num_filters() - f_idx - 1]

            grad = get_filter.chain_rule(gradient, variable_out, variable_in)

        # ! 20240228 Ian - Fixed, previous version had something wrong
        # for i in range(self.num_filters() - 1, -1, -1):
        #     filt = self.filters[i]
        #     y = self.w[..., i]
        #     x = self.w[..., i - 1]

        #     grad = filt.chain_rule(grad, y, x)

        return grad

    def import_cur_index(
        self,
        import_primitive: Import,
        reinterpolation_factor: float = 1,
        reinterpolation_size: tuple[int, int, int] = REINTERPOLATION_SIZE,
        binarize: bool = False,
    ):
        """Reinterpolate and import cur_index into design regions.

        Arguments:
            import_primitive (Import): The import primitive to import the index to. Will
                call `import_primitive.set_nk2` with the appropriate values.
            reinterpolation_factor (float): The scale factor to use when
                reinterpolating. If 1, this function will not perform reinterpolation.
                Defaults to 1.
            reinterpolation_size (tuple[int, int, int]): The number of voxels for the
                output when reinterpolating. Defaults to (300, 300, 306).
            binarize (bool): Whether to binarize the index before importing. Defaults to
                False.
        """
        # Here cur_density will have values between 0 and 1
        cur_density = self.get_density().copy()

        # Perform repetition of density values across each axis for shrinking during reinterpolation process
        if reinterpolation_factor != 1:
            new_size = tuple(int(x) for x in np.ceil(np.multiply(reinterpolation_factor, self.size)))
            cur_density_import = repeat(cur_density, new_size)
            # cur_density_import = np.repeat(
            #     np.repeat(
            #         np.repeat(
            #             cur_density,
            #             np.ceil(reinterpolation_factor),
            #             axis=0,
            #         ),
            #         np.ceil(reinterpolation_factor),
            #         axis=1,
            #     ),
            #     np.ceil(reinterpolation_factor),
            #     axis=2,
            # )
            design_region = [
                1e-6
                * np.linspace(
                    self.coords[axis][0],
                    self.coords[axis][-1],
                    len(self.coords[axis]) * reinterpolation_factor,
                )
                for axis in self.coords
            ]
            design_region_import = [
                1e-6
                * np.linspace(
                    self.coords[axis][0],
                    self.coords[axis][-1],
                    reinterpolation_size[i],
                )
                for i, axis in enumerate(self.coords)
            ]
        else:
            cur_density_import = cur_density.copy()
            design_region = [
                1e-6 * np.linspace(
                    self.coords[axis][0], self.coords[axis][-1], cur_density_import.shape[i],
                )
                for i, axis in enumerate(self.coords)
            ]
            design_region_import = [
                1e-6 * np.linspace(
                    self.coords[axis][0], self.coords[axis][-1], self.field_shape[i]
                )
                for i, axis in enumerate(self.coords)
            ]

        design_region_import_grid = np.array(
            np.meshgrid(*design_region_import, indexing='ij')
        ).transpose((1, 2, 3, 0))

        # Perform reinterpolation so as to fit into Lumerical mesh.
        # NOTE: The imported array does not need to have the same size as the
        # Lumerical mesh. Lumerical will perform its own reinterpolation
        # (params. of which are unclear).
        cur_density_import = interpolate.interpn(
            tuple(design_region),
            cur_density_import,
            design_region_import_grid,
            method='linear',
        )
        #! TODO: CHECK - Don't think this is going to work for 2D.

        # Binarize before importing
        if binarize:
            cur_density_import = self.binarize(cur_density_import)

        # TODO: Add dispersion. Account for dispersion by using the dispersion_model
        dispersive_max_permittivity = self.permittivity_constraints[1]
        self.index_from_permittivity(dispersive_max_permittivity)

        # Convert device to permittivity and then index
        cur_permittivity = self.density_to_permittivity(
            cur_density_import,
            self.permittivity_constraints[0],
            dispersive_max_permittivity,
        )
        # TODO: Another way to do this that can include dispersion is to update the
        # Scale filter with new permittivity constraint maximum
        cur_index = self.index_from_permittivity(cur_permittivity)

        # TODO (maybe): cut cur_index by region
        # Import to simulation
        import_primitive.set_nk2(cur_index, *design_region_import)

        return cur_density, cur_permittivity

    def interpolate_gradient(
        self,
        g: npt.NDArray,
        dimension: str = '3D'
        ):
        """Reinterpolate and import gradient into the shape of the design regions.

        Arguments:
            g (NDArray): The uninterpolated, but perhaps processed, gradient with shape obtained from
                        the design efield monitors for adjoint optimization.
            dimension (str): The dimension of the simulation, either "2D" or "3D". Defaults to "3D".
        """

        gradient_region_x = 1e-6 * np.linspace(self.coords['x'][0], self.coords['x'][-1], g.shape[0]) #cfg.pv.device_voxels_simulation_mesh_lateral_bordered)
        gradient_region_y = 1e-6 * np.linspace(self.coords['y'][0], self.coords['y'][-1], g.shape[1])
        try:
            gradient_region_z = 1e-6 * np.linspace(self.coords['z'][0], self.coords['z'][-1], g.shape[2])
        except IndexError as err:	# 2D case
            gradient_region_z = 1e-6 * np.linspace(self.coords['z'][0], self.coords['z'][-1], 3)
        design_region_geometry = np.array(np.meshgrid(1e-6*self.coords['x'], 1e-6*self.coords['y'], 1e-6*self.coords['z'], indexing='ij')
                            ).transpose((1,2,3,0))

        if dimension in '2D':
            design_gradient_interpolated = interpolate.interpn( ( gradient_region_x, gradient_region_y, gradient_region_z ),
                                                np.repeat(g[..., np.newaxis], 3, axis=2), 	# Repeats 2D array in 3rd dimension, 3 times
                                                design_region_geometry, method='linear' )
        elif dimension in '3D':
            design_gradient_interpolated = interpolate.interpn( ( gradient_region_x, gradient_region_y, gradient_region_z ),
                                                g,
                                                design_region_geometry, method='linear' )

        return design_gradient_interpolated


    def export_density_as_stl(self, filename):
        '''Binarizes device density and exports as STL file for fabrication.'''
        
        # STL Export requires density to be fully binarized
        full_density = self.binarize(self.get_density())
        stl_generator = STL.STL(full_density)
        stl_generator.generate_stl()
        stl_generator.save_stl( filename )

    def export_density_as_gds(self, gds_layer_dir):
        
        # STL Export requires density to be fully binarized
        full_density = self.binarize(self.get_density())
        
        # GDS Export must be done in layers. We split the full design into individual layers, (not related to design layers)
        # Express each layer as polygons in an STL Mesh object and write that layer into its own cell of the GDS file
        layer_mesh_array = []
        for z_layer in range(full_density.shape[2]):
            stl_generator = STL.STL(full_density[..., z_layer][..., np.newaxis])
            stl_generator.generate_stl()
            layer_mesh_array.append(stl_generator.stl_mesh)
        
        # Create a GDS object that contains a Library with Cells corresponding to each 2D layer in the 3D device.
        gds_generator = GDS.GDS().set_layers(full_density.shape[2], 
                unit = 1e-6 * np.abs(self.coords['x'][-1] - self.coords['x'][0])/self.size[0])
        gds_generator.assemble_device(layer_mesh_array, listed=False)

        # Create directory for export
        Path(gds_layer_dir).mkdir(parents=True, exist_ok=True)
        # Export both GDS and SVG for easy visualization.
        gds_generator.export_device(gds_layer_dir, filetype='gds')
        gds_generator.export_device(gds_layer_dir, filetype='svg')
        
        # for layer_idx in range(0, full_density.shape[2]):         # Individual layer as GDS file export
        #     gds_generator.export_device(gds_layer_dir, filetype='gds', layer_idx=layer_idx)
        #     gds_generator.export_device(gds_layer_dir, filetype='svg', layer_idx=layer_idx)
        
        # # Here is a function for GDS device import to Lumerical - be warned this takes maybe 3-5 minutes per layer.
        # def import_gds(sim, device, gds_layer_dir):
        #     import time
            
        #     sim.fdtd.load(sim.info['name'])
        #     sim.fdtd.switchtolayout()
        #     device_name = device.import_names()[0]
        #     sim.disable([device_name])
            
        #     z_vals = device.coords['z']
        #     for z_idx in range(len(z_vals)):
        #         t = time.time()
        #         # fdtd.gdsimport(os.path.join(gds_layer_dir, 'device.gds'), f'L{layer_idx}', 0)
        #         sim.fdtd.gdsimport(os.path.join(gds_layer_dir, 'device.gds'), f'L{z_idx}', 0, "Si (Silicon) - Palik",
        #                     z_vals[z_idx], z_vals[z_idx+1])
        #         sim.fdtd.set({'x': device.coords['x'][0], 'y': device.coords['y'][0]})      # Be careful of units - 1e-6
        #         vipdopt.logger.info(f'Layer {z_idx} imported in {time.time()-t} seconds.')