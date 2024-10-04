"""Module for the abstract Filter class and all its implementations."""

import abc
import copy

import numpy as np
import numpy.typing as npt
from overrides import override

from vipdopt.utils import sech, flatten

SIGMOID_BOUNDS = (0.0, 1.0)


# TODO: design a way for different filters to take different arguments in methods
# TODO: Make code more robust to inputs being arrays iinstead of scalars
class Filter(abc.ABC):
    """An abstract interface for Filters."""

    @property
    @abc.abstractmethod
    def _bounds(self) -> tuple[float, float]:
        pass

    @property
    @abc.abstractmethod
    def init_vars(self) -> dict:
        """The variables used to initalize this Filter."""

    def verify_bounds(self, variable: npt.NDArray | float | float) -> bool:
        """Checks if variable is within bounds of this filter.

        Variable can either be a single number or an array of numbers.
        """
        return bool(
            (np.min(np.array(variable)) >= self._bounds[0])
            and (np.max(np.array(variable)) <= self._bounds[1])
        )

    @abc.abstractmethod
    def forward(self, x: npt.NDArray | float) -> npt.NDArray | float:
        """Propagate x through the filter and return the result."""

    @abc.abstractmethod
    def fabricate(self, x: npt.NDArray | float) -> npt.NDArray | float:
        """Propogate x through the filter and return the result, binarizing values."""

    @abc.abstractmethod
    def chain_rule(
        self,
        deriv_out: npt.NDArray | float,
        var_out: npt.NDArray | float,
        var_in: npt.NDArray | float,
    ) -> npt.NDArray | float:
        """Apply the chain rule and propagate the derivative back one step."""


class Sigmoid(Filter):
    """Applies a sigmoidal projection filter to binarize an input.

    Takes an input auxiliary density p(x) ranging from 0 to 1 and applies a sigmoidal
    projection filter to binarize / push it to either extreme. This depends on the
    strength of the filter. See OPTICA paper supplement Section IIA,
    https://doi.org/10.1364/OPTICA.384228,  for details. See also Eq. (9) of
    https://doi.org/10.1007/s00158-010-0602-y.

    Attributes:
        eta (float): The center point of the sigmoid. Must be in range [0, 1].
        beta (float): The strength of the sigmoid
        _bounds (tuple[float, float]): The bounds of the filter.
            Always equal to (0, 1)
        _denominator (float | npt.NDArray): The denominator used in various methods;
            for reducing re-computation.
    """

    @property
    @override
    def _bounds(self):
        return SIGMOID_BOUNDS

    @property
    @override
    def init_vars(self) -> dict:
        return {'eta': self.eta, 'beta': self.beta}

    def __init__(self, eta: float, beta: float, *args, **kwargs) -> None:
        """Initialize a sigmoid filter based on eta and beta values."""
        if not self.verify_bounds(eta):  # type: ignore
            raise ValueError('Eta must be in the range [0, 1]')

        self.eta = eta
        self.beta = beta

        # Calculate denominator for use in methods
        self._denominator = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))

    def __repr__(self) -> str:
        """Return a string representation of the filter."""
        return f'Sigmoid filter with eta={self.eta:0.3f} and beta={self.beta:0.3f}'

    def __eq__(self, __value: object) -> bool:
        """Test equality."""
        if isinstance(__value, Sigmoid):
            return self.eta == __value.eta and self.beta == __value.beta
        return super().__eq__(__value)

    @override
    def forward(self, x: npt.NDArray | float) -> npt.NDArray | float:
        """Propogate x through the filter and return the result.
        All input values of x above the threshold eta, are projected to 1, and the
        values below, projected to 0. This is Eq. (9) of https://doi.org/10.1007/s00158-010-0602-y.
        """
        numerator = np.tanh(self.beta * self.eta) + np.tanh(
            self.beta * (np.copy(np.array(x)) - self.eta)
        )
        return numerator / self._denominator

    @override  # type: ignore
    def chain_rule(
        self,
        deriv_out: npt.NDArray | float,
        var_out: npt.NDArray | float,
        var_in: npt.NDArray | float,
    ) -> npt.NDArray | float:
        """Apply the chain rule and propogate the derivative back one step.

        Returns the first argument, multiplied by the direct derivative of forward()
        i.e. Eq. (9) of https://doi.org/10.1007/s00158-010-0602-y, with respect to
        \\tilde{p_i}.
        """
        del var_out  # not needed for sigmoid filter

        numerator = self.beta * np.power(sech(self.beta * (var_in - self.eta)), 2)  # type: ignore
        return (
            deriv_out * numerator / self._denominator
        )  # ! 20240228 Ian - Fixed, was missing a deriv_out factor

    @override
    def fabricate(self, x: npt.NDArray | float) -> npt.NDArray | float:
        """Apply filter to input as a hard step-function instead of sigmoid.

        Returns:
            _bounds[0] where x <= eta, and _bounds[1] otherwise
        """
        fab = np.array(x)
        fab[fab <= self.eta] = self._bounds[0]
        fab[fab > self.eta] = self._bounds[1]
        if isinstance(x, float):  # type: ignore
            return fab.item()
        return fab


class Scale(Filter):
    """Filter for scaling an input to a specified range.

    Assuming that there is an input between 0 and 1, scales it to the range, min, and
    max, that are declared during initialization. See OPTICA paper supplement Section
    IIA, https://doi.org/10.1364/OPTICA.384228,  for details

    This class is used directly after the sigmoid filter is applied.

    Attributes:
        _bounds (tuple[float]): The min/max permittivity
        variable_bounds (tuple[float]): The min/max permittivity
            (equivalent to `_bounds`).
        range (float): The total range of the bounds of the filter.
    """

    @property
    @override
    def _bounds(self):
        return self.variable_bounds

    @property
    @override
    def init_vars(self) -> dict:
        return {'variable_bounds': self.variable_bounds}

    def __init__(self, variable_bounds: tuple[float, float], *args, **kwargs):
        """Initialize a Scale filter."""
        self.variable_bounds = variable_bounds

        self.range = self._bounds[1] - self._bounds[0]

    def __eq__(self, __value: object) -> bool:
        """Test equality."""
        if isinstance(__value, Scale):
            return self.variable_bounds == __value.variable_bounds
        return super().__eq__(__value)

    def __repr__(self):
        """Return a string representation of the filter."""
        return 'Scale filter with range: [{:0.3f}, {:0.3f}]'.format(*self._bounds)

    @override
    def forward(self, x: npt.NDArray | float) -> npt.NDArray | float:
        """Performs scaling according to min and max values declared.

        Assumes that input is between 0 and 1."""
        return np.add(self._bounds[0], np.multiply(self.range, x))

    @override  # type: ignore
    def chain_rule(
        self,
        deriv_out: npt.NDArray | float,
        var_out: npt.NDArray | float,  # noqa: ARG002
        var_in: npt.NDArray | float,  # noqa: ARG002
    ) -> npt.NDArray | float:
        """Apply the chain rule and propagate the derivative back one step."""
        return np.multiply(self.range, deriv_out)

    @override
    def fabricate(self, x: npt.NDArray | float) -> npt.NDArray | float:
        """Calls forward()."""
        return self.forward(x)


class Layering(Filter):
    """'''Takes a device and splits it into layers in the dimension requested. Gives the
    functionality to get the indices of each layer, and add spacers in between each layer.

    Restricts optimization by averaging the calculated sensitivity in the vertical direction
    within EACH layer, such that voxels within each layer are governed by a shared 2D profile.
    i.e. n layers, 1 layer has m voxels, all voxels have a shared 2D permittivity profile
    WITHIN the nth layer, but each layer differs from the rest in its 2D profile

    See OPTICA paper supplement Section IIC, https://doi.org/10.1364/OPTICA.384228, for details.
    """

    @property
    @override
    def _bounds(self):
        return self.variable_bounds

    @property
    @override
    def init_vars(self) -> dict:
        return {'dim': self.dim,
                'num_layers': self.num_layers,
                'num_spacers': self.num_spacers,
                'variable_bounds': self.variable_bounds,
                'layer_height_voxels': self.layer_height_voxels,
                'spacer_height_voxels':self.spacer_height_voxels,
                'spacer_voxels_value':self.spacer_voxels_value,
                }

    def __init__(self, dim:str|int, num_layers:int, num_spacers:int,
                variable_bounds:tuple[float, float]=(0,1),
                layer_height_voxels=None, spacer_height_voxels=0,
                spacer_voxels_value=0,):
        """Initialize a Layering filter.
        dim: dimension to layer on
        num_layers: number of layers
        variable_bounds: taken as density for default.
        spacer_height_voxels: height of each spacer in voxels
        spacer_voxels_value: value to set spacer voxels to. should be in the range of the variable_bounds
        NOTE: By default this sets spacers of equal voxel height in between every layer.
        If you want to change that, recalculate the start indices via self.get_layer_start_idxs().
        """
        dim_strs = ['x','y','z']
        if isinstance(dim, str) and dim.lower() in dim_strs:
            dim = dim_strs.index(dim)

        self.dim = dim
        self.num_layers = num_layers
        self.num_spacers = num_spacers
        self.variable_bounds = variable_bounds
        self.layer_height_voxels = layer_height_voxels
        self.spacer_height_voxels = spacer_height_voxels
        self.spacer_voxels_value = spacer_voxels_value

        print(f'Layer, spacer:{self.layer_height_voxels, self.spacer_height_voxels}')

        self.range = self._bounds[1] - self._bounds[0]
        
        self.layer_start_idxs = {} 

    def __eq__(self, __value: object) -> bool:
        """Test equality."""
        if isinstance(__value, Layering):
            return self.num_layers == __value.num_layers & self.dimension == __value.dimension
        return super().__eq__(__value)

    def __repr__(self):
        """Return a string representation of the filter."""
        return f'Layering filter: {self.num_layers}-layer in {self.dimension}.'

    def get_layer_spacer_idxs(self, variable_shape, start_from:str ='bottom',
                       layer_height_voxels = None, layer_type_nums = None,
                       spacer_first:bool = False,
                       mode:str = 'auto'):

        '''Gets the indices of each layer in the dimension that was declared to the Layering object.
         Return list of tuples, indicating start (inclusive) and end (exclusive).'''
        #! This method assumes height of each layer is fixed via self.layer(spacer)_height_voxels
        # Consequently the last layer(spacer) will be thicker to fill out the device size.

        layer_type_names = ['design','spacers']
        if layer_type_nums is None:
            # Default: N design layers, N-1 spacers.
            layer_type_nums = [self.num_layers, self.num_layers-1]
        if layer_height_voxels is None:
            layer_height_voxels = [self.layer_height_voxels, self.spacer_height_voxels]
        if mode=='auto':
            pass    # todo:
        num_voxels_total = int(variable_shape[self.dim])

        # If there are N layers, N-1 spacers, a design layer is on the bottom.
        match layer_type_nums[1] - layer_type_nums[0]:
            case -1:
                spacer_first = False
        # If there are N layers, N+1 spacers, a spacer layer is on the bottom.
            case 1:
                spacer_first = True
        # If there are N layers, N spacers, either a design or spacer layer can be on the bottom.

        layer_start_idxs = dict(zip(layer_type_names, [[] for x in layer_type_names]))
         # -> Assemble list of tuples, indicating start (inclusive) and end (exclusive).
        current_idx = 0
        for n in range(np.sum(layer_type_nums)):
            # Which of the layer types is it? We cycle through
            layer_type_idx = (spacer_first + n) % len(layer_type_names)
            # Get name and height
            layer_type_name = layer_type_names[layer_type_idx]
            height_voxels = layer_height_voxels[layer_type_idx]
            # Append resulting tuple [start, end) to the correct key
            if n == np.sum(layer_type_nums)-1:
                layer_start_idxs[layer_type_name].append((current_idx, num_voxels_total))
            else:
                print(current_idx, height_voxels)
                layer_start_idxs[layer_type_name].append((current_idx, current_idx + height_voxels))
            # Update current index
            current_idx = current_idx + height_voxels

        if not start_from=='bottom':    # Flip
            for k in layer_start_idxs:
                # Superimpose into a reversed array, 1 larger to account for zero-index
                layer_start_idxs[k] = np.arange(num_voxels_total + 1)[::-1][layer_start_idxs[k]]
                # list[list] -> list[tuple]
                layer_start_idxs[k] = [tuple(np.sort(x)) for x in layer_start_idxs[k]]
                # It will be in reversed order, flip it back
                layer_start_idxs[k] = layer_start_idxs[k][::-1]

        return layer_start_idxs
    
    # def get_layer_idxs(self, variable_shape, start_from:str ='bottom'):
    #     '''Gets the indices of each layer in the dimension that was declared to the Layering object'''
    #     num_voxels_total = variable_shape[self.dim]
    #     layer_idxs = []

    #     num_voxels_per_layer = int(num_voxels_total / self.num_layers)
    #     num_voxels_last_layer = num_voxels_total - (self.num_layers - 1) * num_voxels_per_layer

    #     if start_from=='bottom':
    #         # Start from bottom - the top layer will have leftovers (extra thickness).
    #         return list(np.arange(0, self.num_layers*num_voxels_per_layer, num_voxels_per_layer))
    #     else:
    #         # Start from top - the bottom layer will have leftovers (extra thickness).
    #         return [0] + list(np.arange(num_voxels_last_layer, num_voxels_total, num_voxels_per_layer))

    def do_averaging(self, data, dim, output, *args, **kwargs):
        '''Averages out all the voxels within one layer in a specific (usually vertical) direction,
           such that they all have the same n-D profile within that layer.
           
           data: the unperturbed input variable, with slicing already performed.
           dim: the dimension in which to average.
           output: serves as a template, the values of the input variable are averaged and pasted in.'''

        # Takes the average of the data along the dimension declared to the Layering object
        average = np.mean(data, axis=dim)

        for idx in range(0, data.shape[dim]):
            idx_bunch = [slice(None)] * data.ndim
            idx_bunch[dim] = np.array(idx)
            # Populates all entries of this slice with the average value from above
            output[tuple(idx_bunch)] = average

        return output

    def do_constant(self, data, dim, output, const_value):
        '''Sets all the voxels within one spacer in a specific (usually vertical) direction,
           such that they all have the same constant density value within layer.
           
           data: the unperturbed input variable, with slicing already performed.
           dim: the dimension in which to average.
           output: serves as a template, the values of the input variable are averaged and pasted in.'''

        # Constant value to assign
        const = const_value

        for idx in range(0, data.shape[dim]):
            idx_bunch = [slice(None)] * data.ndim
            idx_bunch[dim] = np.array(idx)
            # Populates all entries of this slice with the constant value from above
            output[tuple(idx_bunch)] = const

        return output
    
    def iterate_through_layers(self, layer_start_idxs, variable, func, *args, **kwargs):
        '''Given a design region and start indices for each layer/region, iterates through them
        and performs some function on those regions only.'''
        variable_out = copy.deepcopy(variable)
        
        for layer_range in layer_start_idxs:
            idx_slice = [slice(None)] * variable.ndim
            idx_slice[self.dim] = np.arange(*layer_range)
            idx_slice = tuple(idx_slice)
            region = variable[idx_slice]
            variable_out[idx_slice] = func(region, self.dim, variable[idx_slice], *args, **kwargs)

        return variable_out

    def layer_averaging(self, variable, layer_start_idxs):
        return self.iterate_through_layers(layer_start_idxs, variable, self.do_averaging)
    
    def make_spacer(self, variable, layer_start_idxs, spacer_value):
        return self.iterate_through_layers(layer_start_idxs, variable, self.do_constant, spacer_value)

    @override
    def forward(self, x: npt.NDArray | float, spacer_value=None) -> npt.NDArray | float:
        """Performs normal layer averaging and spacer assignment."""
        if spacer_value is None:
            spacer_value = self.spacer_voxels_value
        
        # Here we set the default number of spacers to be (num_layers - 1).
        self.layer_start_idxs = self.get_layer_spacer_idxs(x.shape, layer_type_nums=[self.num_layers, self.num_spacers],
                                    spacer_first=False, start_from='bottom',
                                    mode='auto')
        
        x_out = self.layer_averaging(x, self.layer_start_idxs['design'])
        x_out = self.make_spacer(x_out, self.layer_start_idxs['spacers'], spacer_value)
        return x_out

    @override  # type: ignore
    def chain_rule(
        self,
        deriv_out: npt.NDArray | float,
        var_out: npt.NDArray | float,  # noqa: ARG002
        var_in: npt.NDArray | float,  # noqa: ARG002
    ) -> npt.NDArray | float:
        """Apply the chain rule and propagate the derivative back one step."""
        return self.forward(deriv_out, 0)

    @override
    def fabricate(self, x: npt.NDArray | float) -> npt.NDArray | float:
        """Calls forward()."""
        return self.forward(x)


# # Blurs can be more general.  We just need to specify a mask and maximum approximation function (and its derivative)
# class SquareBlur(Filter):
#     # TODO: Could combine square_blur.py and square_blur_smooth.py into the same file.
# 	# alpha: strength of blur

# 	def __init__(self, alpha, blur_half_width, variable_bounds=[0, 1]):
# 		super(SquareBlur, self).__init__(variable_bounds)

# 		self.alpha = alpha
# 		self.blur_half_width = [int(element) for element in (blur_half_width)]

# 		# At this point, we can compute which points in a volume will be part of this blurring operation
# 		self.mask_size = [ (1 + 2 * element) for element in self.blur_half_width ]

# 		self.blur_mask = np.zeros((self.mask_size[0], self.mask_size[1], self.mask_size[2]))

# 		for mask_x in range(-self.blur_half_width[0], self.blur_half_width[0] + 1):
# 			for mask_y in range(-self.blur_half_width[1], self.blur_half_width[1] + 1):
# 				for mask_z in range(-self.blur_half_width[2], self.blur_half_width[2] + 1):
# 					self.blur_mask[self.blur_half_width[0] + mask_x, self.blur_half_width[1] + mask_y, self.blur_half_width[2] + mask_z] = 1

# 		self.number_to_blur = sum((self.blur_mask).flatten())


# 	def forward(self, variable_in):
# 		pad_variable_in = np.pad(
# 			variable_in,
# 			((self.blur_half_width[0], self.blur_half_width[0]), (self.blur_half_width[1], self.blur_half_width[1]), (self.blur_half_width[2], self.blur_half_width[2])),
# 			'constant'
# 		)

# 		unpadded_shape = variable_in.shape
# 		padded_shape = pad_variable_in.shape

# 		start_x = self.blur_half_width[0]
# 		start_y = self.blur_half_width[1]
# 		start_z = self.blur_half_width[2]

# 		x_length = unpadded_shape[0]
# 		y_length = unpadded_shape[1]
# 		z_length = unpadded_shape[2]

# 		blurred_variable = np.zeros((x_length, y_length, z_length))
# 		for mask_x in range(-self.blur_half_width[0], self.blur_half_width[0] + 1):
# 			offset_x = start_x + mask_x
# 			x_bounds = [offset_x, (offset_x + x_length)]
# 			for mask_y in range(-self.blur_half_width[1], self.blur_half_width[1] + 1):
# 				offset_y = start_y + mask_y
# 				y_bounds = [offset_y, (offset_y + y_length)]
# 				for mask_z in range(-self.blur_half_width[2], self.blur_half_width[2] + 1):
# 					offset_z = start_z + mask_z
# 					z_bounds = [offset_z, (offset_z + z_length)]

# 					check_mask = self.blur_mask[mask_x + self.blur_half_width[0], mask_y + self.blur_half_width[1], mask_z + self.blur_half_width[2]]

# 					if check_mask == 1:
# 						blurred_variable = np.add(
# 							blurred_variable,
# 							np.exp(
# 								np.multiply(
# 									self.alpha,
# 									pad_variable_in[x_bounds[0]:x_bounds[1], y_bounds[0]:y_bounds[1], z_bounds[0]:z_bounds[1]]
# 									)
# 								)
# 							)

# 		blurred_variable = np.divide(np.log(np.divide(blurred_variable, self.number_to_blur)), self.alpha)

# 		return blurred_variable

# 	def chain_rule(self, derivative_out, variable_out, variable_in):
# 		pad_variable_out = np.pad(
# 			variable_out,
# 			((self.blur_half_width[0], self.blur_half_width[0]), (self.blur_half_width[1], self.blur_half_width[1]), (self.blur_half_width[2], self.blur_half_width[2])),
# 			'constant'
# 		)

# 		pad_derivative_out = np.pad(
# 			derivative_out,
# 			((self.blur_half_width[0], self.blur_half_width[0]), (self.blur_half_width[1], self.blur_half_width[1]), (self.blur_half_width[2], self.blur_half_width[2])),
# 			'constant'
# 		)

# 		start_x = self.blur_half_width[0]
# 		start_y = self.blur_half_width[1]
# 		start_z = self.blur_half_width[2]

# 		unpadded_shape = variable_in.shape
# 		x_length = unpadded_shape[0]
# 		y_length = unpadded_shape[1]
# 		z_length = unpadded_shape[2]

# 		derivative_in = np.zeros(derivative_out.shape)

# 		for mask_x in range(-self.blur_half_width[0], self.blur_half_width[0] + 1):
# 			offset_x = start_x + mask_x
# 			x_bounds = [offset_x, (offset_x + x_length)]
# 			for mask_y in range(-self.blur_half_width[1], self.blur_half_width[1] + 1):
# 				offset_y = start_y + mask_y
# 				y_bounds = [offset_y, (offset_y + y_length)]
# 				for mask_z in range(-self.blur_half_width[2], self.blur_half_width[2] + 1):
# 					offset_z = start_z + mask_z
# 					z_bounds = [offset_z, (offset_z + z_length)]

# 					check_mask = self.blur_mask[mask_x + self.blur_half_width[0], mask_y + self.blur_half_width[1], mask_z + self.blur_half_width[2]]

# 					if check_mask == 1:
# 						derivative_in = np.add(
# 							derivative_in,
# 							np.multiply(
# 								np.exp(
# 									np.multiply(
# 										self.alpha,
# 										np.subtract(
# 											variable_in,
# 											pad_variable_out[x_bounds[0]:x_bounds[1], y_bounds[0]:y_bounds[1], z_bounds[0]:z_bounds[1]]
# 										)
# 									)
# 								),
# 							pad_derivative_out[x_bounds[0]:x_bounds[1], y_bounds[0]:y_bounds[1], z_bounds[0]:z_bounds[1]])
# 						)

# 		derivative_in = np.divide(derivative_in, self.number_to_blur)
# 		return derivative_in

# 	def fabricate(self, variable_in):
# 		variable_shape = variable_in.shape

# 		start_x = self.blur_half_width[0]
# 		start_y = self.blur_half_width[1]
# 		start_z = self.blur_half_width[2]

# 		x_length = variable_shape[0] - 2 * self.blur_half_width[0]
# 		y_length = variable_shape[1] - 2 * self.blur_half_width[1]
# 		z_length = variable_shape[2] - 2 * self.blur_half_width[2]

# 		blurred_variable = np.zeros((variable_in.shape))
# 		blurred_variable_change = np.zeros((x_length, y_length, z_length))

# 		for mask_x in range(-self.blur_half_width[0], self.blur_half_width[0] + 1):
# 			offset_x = start_x + mask_x
# 			x_bounds = [offset_x, (offset_x + x_length)]
# 			for mask_y in range(-self.blur_half_width[1], self.blur_half_width[1] + 1):
# 				offset_y = start_y + mask_y
# 				y_bounds = [offset_y, (offset_y + y_length)]
# 				for mask_z in range(-self.blur_half_width[2], self.blur_half_width[2] + 1):
# 					offset_z = start_z + mask_z
# 					z_bounds = [offset_z, (offset_z + z_length)]

# 					check_mask = self.blur_mask[mask_x + self.blur_half_width[0], mask_y + self.blur_half_width[1], mask_z + self.blur_half_width[2]]

# 					if check_mask == 1:
# 						blurred_variable_change = np.maximum(
# 							blurred_variable_change,
# 							variable_in[x_bounds[0]:x_bounds[1], y_bounds[0]:y_bounds[1], z_bounds[0]:z_bounds[1]])

# 		blurred_variable[
# 			start_x : (start_x + x_length),
# 			start_y : (start_y + y_length),
# 			start_z : (start_z + z_length)] = blurred_variable_change
# 		return blurred_variable


# class SquareBlurSmooth(Filter):
#     # TODO: Could combine square_blur.py and square_blur_smooth.py into the same file.
#     # TODO: Use <from scipy.ndimage import gaussian_filter> to do the smooth blurring.

# 	def __init__(self, blur_half_width, variable_bounds=[0, 1]):
# 		super(SquareBlurSmooth, self).__init__(variable_bounds)

# 		self.blur_half_width = [int(element) for element in (blur_half_width)]

# 		# At this point, we can compute which points in a volume will be part of this blurring operation
# 		self.mask_size = [ (1 + 2 * element) for element in self.blur_half_width ]

# 		self.blur_mask = np.zeros((self.mask_size[0], self.mask_size[1], self.mask_size[2]))

# 		self.blur_radius = np.sqrt( np.sum( np.array( self.blur_half_width )**2 ) )
# 		self.blur_normalization = 0

# 		for mask_x in range(-self.blur_half_width[0], self.blur_half_width[0] + 1):
# 			for mask_y in range(-self.blur_half_width[1], self.blur_half_width[1] + 1):
# 				for mask_z in range(-self.blur_half_width[2], self.blur_half_width[2] + 1):
# 					self.blur_mask[self.blur_half_width[0] + mask_x, self.blur_half_width[1] + mask_y, self.blur_half_width[2] + mask_z] = 1

# 					self.blur_normalization += ( ( self.blur_radius + 1 - np.sqrt( mask_x**2 + mask_y**2 + mask_z**2 ) ) / self.blur_radius )

# 		self.number_to_blur = sum((self.blur_mask).flatten())


# 	def forward(self, variable_in):
# 		pad_variable_in = np.pad(
# 			variable_in,
# 			((self.blur_half_width[0], self.blur_half_width[0]), (self.blur_half_width[1], self.blur_half_width[1]), (self.blur_half_width[2], self.blur_half_width[2])),
# 			'constant'
# 		)

# 		unpadded_shape = variable_in.shape
# 		padded_shape = pad_variable_in.shape

# 		start_x = self.blur_half_width[0]
# 		start_y = self.blur_half_width[1]
# 		start_z = self.blur_half_width[2]

# 		x_length = unpadded_shape[0]
# 		y_length = unpadded_shape[1]
# 		z_length = unpadded_shape[2]

# 		blurred_variable = np.zeros((x_length, y_length, z_length))
# 		for mask_x in range(-self.blur_half_width[0], self.blur_half_width[0] + 1):
# 			offset_x = start_x + mask_x
# 			x_bounds = [offset_x, (offset_x + x_length)]
# 			for mask_y in range(-self.blur_half_width[1], self.blur_half_width[1] + 1):
# 				offset_y = start_y + mask_y
# 				y_bounds = [offset_y, (offset_y + y_length)]
# 				for mask_z in range(-self.blur_half_width[2], self.blur_half_width[2] + 1):
# 					offset_z = start_z + mask_z
# 					z_bounds = [offset_z, (offset_z + z_length)]

# 					check_mask = self.blur_mask[mask_x + self.blur_half_width[0], mask_y + self.blur_half_width[1], mask_z + self.blur_half_width[2]]

# 					if check_mask == 1:
# 						blurred_variable = np.add( blurred_variable,
# 							( 1. / self.blur_normalization ) *
# 							( ( self.blur_radius + 1 - np.sqrt( mask_x**2 + mask_y**2 + mask_z**2 ) ) / self.blur_radius ) *
# 							pad_variable_in[x_bounds[0]:x_bounds[1], y_bounds[0]:y_bounds[1], z_bounds[0]:z_bounds[1]] )


# 		return blurred_variable

# 	def chain_rule(self, derivative_out, variable_out, variable_in):
# 		pad_variable_out = np.pad(
# 			variable_out,
# 			((self.blur_half_width[0], self.blur_half_width[0]), (self.blur_half_width[1], self.blur_half_width[1]), (self.blur_half_width[2], self.blur_half_width[2])),
# 			'constant'
# 		)

# 		pad_derivative_out = np.pad(
# 			derivative_out,
# 			((self.blur_half_width[0], self.blur_half_width[0]), (self.blur_half_width[1], self.blur_half_width[1]), (self.blur_half_width[2], self.blur_half_width[2])),
# 			'constant'
# 		)

# 		start_x = self.blur_half_width[0]
# 		start_y = self.blur_half_width[1]
# 		start_z = self.blur_half_width[2]

# 		unpadded_shape = variable_in.shape
# 		x_length = unpadded_shape[0]
# 		y_length = unpadded_shape[1]
# 		z_length = unpadded_shape[2]

# 		derivative_in = np.zeros(derivative_out.shape)

# 		for mask_x in range(-self.blur_half_width[0], self.blur_half_width[0] + 1):
# 			offset_x = start_x + mask_x
# 			x_bounds = [offset_x, (offset_x + x_length)]
# 			for mask_y in range(-self.blur_half_width[1], self.blur_half_width[1] + 1):
# 				offset_y = start_y + mask_y
# 				y_bounds = [offset_y, (offset_y + y_length)]
# 				for mask_z in range(-self.blur_half_width[2], self.blur_half_width[2] + 1):
# 					offset_z = start_z + mask_z
# 					z_bounds = [offset_z, (offset_z + z_length)]

# 					check_mask = self.blur_mask[mask_x + self.blur_half_width[0], mask_y + self.blur_half_width[1], mask_z + self.blur_half_width[2]]

# 					if check_mask == 1:
# 						derivative_in = np.add( derivative_in,
# 							( 1. / self.blur_normalization ) *
# 							( ( self.blur_radius + 1 - np.sqrt( mask_x**2 + mask_y**2 + mask_z**2 ) ) / self.blur_radius ) *
# 							pad_derivative_out[x_bounds[0]:x_bounds[1], y_bounds[0]:y_bounds[1], z_bounds[0]:z_bounds[1]] )

# 		return derivative_in


# 	def fabricate(self, variable_in):
# 		variable_shape = variable_in.shape

# 		start_x = self.blur_half_width[0]
# 		start_y = self.blur_half_width[1]
# 		start_z = self.blur_half_width[2]

# 		x_length = variable_shape[0] - 2 * self.blur_half_width[0]
# 		y_length = variable_shape[1] - 2 * self.blur_half_width[1]
# 		z_length = variable_shape[2] - 2 * self.blur_half_width[2]

# 		blurred_variable = np.zeros((variable_in.shape))
# 		blurred_variable_change = np.zeros((x_length, y_length, z_length))

# 		for mask_x in range(-self.blur_half_width[0], self.blur_half_width[0] + 1):
# 			offset_x = start_x + mask_x
# 			x_bounds = [offset_x, (offset_x + x_length)]
# 			for mask_y in range(-self.blur_half_width[1], self.blur_half_width[1] + 1):
# 				offset_y = start_y + mask_y
# 				y_bounds = [offset_y, (offset_y + y_length)]
# 				for mask_z in range(-self.blur_half_width[2], self.blur_half_width[2] + 1):
# 					offset_z = start_z + mask_z
# 					z_bounds = [offset_z, (offset_z + z_length)]

# 					check_mask = self.blur_mask[mask_x + self.blur_half_width[0], mask_y + self.blur_half_width[1], mask_z + self.blur_half_width[2]]

# 					if check_mask == 1:
# 						blurred_variable_change = np.add( blurred_variable_change,
# 							( 1. / self.blur_normalization ) *
# 							( ( self.blur_radius + 1 - np.sqrt( mask_x**2 + mask_y**2 + mask_z**2 ) ) / self.blur_radius ) *
# 							variable_in[x_bounds[0]:x_bounds[1], y_bounds[0]:y_bounds[1], z_bounds[0]:z_bounds[1]] )


# 		# todo - unsure about this and same for in the square blur

# 		blurred_variable[
# 			start_x : (start_x + x_length),
# 			start_y : (start_y + y_length),
# 			start_z : (start_z + z_length)] = blurred_variable_change
# 		return blurred_variable