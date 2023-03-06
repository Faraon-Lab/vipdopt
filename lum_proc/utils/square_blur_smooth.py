import filter as filter

import numpy as np

#
# Blurs can be more general.  We just need to specify a mask and maximum approximation function (and its derivative)
#
class SquareBlurSmooth(filter.Filter):
    # TODO: Could combine square_blur.py and square_blur_smooth.py into the same file.
    # TODO: Use <from scipy.ndimage import gaussian_filter> to do the smooth blurring.

	def __init__(self, blur_half_width, variable_bounds=[0, 1]):
		super(SquareBlurSmooth, self).__init__(variable_bounds)

		self.blur_half_width = [int(element) for element in (blur_half_width)]

		# At this point, we can compute which points in a volume will be part of this blurring operation
		self.mask_size = [ (1 + 2 * element) for element in self.blur_half_width ]

		self.blur_mask = np.zeros((self.mask_size[0], self.mask_size[1], self.mask_size[2]))

		self.blur_radius = np.sqrt( np.sum( np.array( self.blur_half_width )**2 ) )
		self.blur_normalization = 0

		for mask_x in range(-self.blur_half_width[0], self.blur_half_width[0] + 1):
			for mask_y in range(-self.blur_half_width[1], self.blur_half_width[1] + 1):
				for mask_z in range(-self.blur_half_width[2], self.blur_half_width[2] + 1):
					self.blur_mask[self.blur_half_width[0] + mask_x, self.blur_half_width[1] + mask_y, self.blur_half_width[2] + mask_z] = 1

					self.blur_normalization += ( ( self.blur_radius + 1 - np.sqrt( mask_x**2 + mask_y**2 + mask_z**2 ) ) / self.blur_radius )

		self.number_to_blur = sum((self.blur_mask).flatten())


	def forward(self, variable_in):
		pad_variable_in = np.pad(
			variable_in,
			((self.blur_half_width[0], self.blur_half_width[0]), (self.blur_half_width[1], self.blur_half_width[1]), (self.blur_half_width[2], self.blur_half_width[2])),
			'constant'
		)

		unpadded_shape = variable_in.shape
		padded_shape = pad_variable_in.shape

		start_x = self.blur_half_width[0]
		start_y = self.blur_half_width[1]
		start_z = self.blur_half_width[2]

		x_length = unpadded_shape[0]
		y_length = unpadded_shape[1]
		z_length = unpadded_shape[2]

		blurred_variable = np.zeros((x_length, y_length, z_length))
		for mask_x in range(-self.blur_half_width[0], self.blur_half_width[0] + 1):
			offset_x = start_x + mask_x
			x_bounds = [offset_x, (offset_x + x_length)]
			for mask_y in range(-self.blur_half_width[1], self.blur_half_width[1] + 1):
				offset_y = start_y + mask_y
				y_bounds = [offset_y, (offset_y + y_length)]
				for mask_z in range(-self.blur_half_width[2], self.blur_half_width[2] + 1):
					offset_z = start_z + mask_z
					z_bounds = [offset_z, (offset_z + z_length)]

					check_mask = self.blur_mask[mask_x + self.blur_half_width[0], mask_y + self.blur_half_width[1], mask_z + self.blur_half_width[2]]

					if check_mask == 1:
						blurred_variable = np.add( blurred_variable,
							( 1. / self.blur_normalization ) *
							( ( self.blur_radius + 1 - np.sqrt( mask_x**2 + mask_y**2 + mask_z**2 ) ) / self.blur_radius ) *
							pad_variable_in[x_bounds[0]:x_bounds[1], y_bounds[0]:y_bounds[1], z_bounds[0]:z_bounds[1]] )


		return blurred_variable

	def chain_rule(self, derivative_out, variable_out, variable_in):
		pad_variable_out = np.pad(
			variable_out,
			((self.blur_half_width[0], self.blur_half_width[0]), (self.blur_half_width[1], self.blur_half_width[1]), (self.blur_half_width[2], self.blur_half_width[2])),
			'constant'
		)

		pad_derivative_out = np.pad(
			derivative_out,
			((self.blur_half_width[0], self.blur_half_width[0]), (self.blur_half_width[1], self.blur_half_width[1]), (self.blur_half_width[2], self.blur_half_width[2])),
			'constant'
		)

		start_x = self.blur_half_width[0]
		start_y = self.blur_half_width[1]
		start_z = self.blur_half_width[2]

		unpadded_shape = variable_in.shape
		x_length = unpadded_shape[0]
		y_length = unpadded_shape[1]
		z_length = unpadded_shape[2]

		derivative_in = np.zeros(derivative_out.shape)

		for mask_x in range(-self.blur_half_width[0], self.blur_half_width[0] + 1):
			offset_x = start_x + mask_x
			x_bounds = [offset_x, (offset_x + x_length)]
			for mask_y in range(-self.blur_half_width[1], self.blur_half_width[1] + 1):
				offset_y = start_y + mask_y
				y_bounds = [offset_y, (offset_y + y_length)]
				for mask_z in range(-self.blur_half_width[2], self.blur_half_width[2] + 1):
					offset_z = start_z + mask_z
					z_bounds = [offset_z, (offset_z + z_length)]

					check_mask = self.blur_mask[mask_x + self.blur_half_width[0], mask_y + self.blur_half_width[1], mask_z + self.blur_half_width[2]]

					if check_mask == 1:
						derivative_in = np.add( derivative_in,
							( 1. / self.blur_normalization ) *
							( ( self.blur_radius + 1 - np.sqrt( mask_x**2 + mask_y**2 + mask_z**2 ) ) / self.blur_radius ) *
							pad_derivative_out[x_bounds[0]:x_bounds[1], y_bounds[0]:y_bounds[1], z_bounds[0]:z_bounds[1]] )

		return derivative_in


	def fabricate(self, variable_in):
		variable_shape = variable_in.shape

		start_x = self.blur_half_width[0]
		start_y = self.blur_half_width[1]
		start_z = self.blur_half_width[2]

		x_length = variable_shape[0] - 2 * self.blur_half_width[0]
		y_length = variable_shape[1] - 2 * self.blur_half_width[1]
		z_length = variable_shape[2] - 2 * self.blur_half_width[2]

		blurred_variable = np.zeros((variable_in.shape))
		blurred_variable_change = np.zeros((x_length, y_length, z_length))

		for mask_x in range(-self.blur_half_width[0], self.blur_half_width[0] + 1):
			offset_x = start_x + mask_x
			x_bounds = [offset_x, (offset_x + x_length)]
			for mask_y in range(-self.blur_half_width[1], self.blur_half_width[1] + 1):
				offset_y = start_y + mask_y
				y_bounds = [offset_y, (offset_y + y_length)]
				for mask_z in range(-self.blur_half_width[2], self.blur_half_width[2] + 1):
					offset_z = start_z + mask_z
					z_bounds = [offset_z, (offset_z + z_length)]

					check_mask = self.blur_mask[mask_x + self.blur_half_width[0], mask_y + self.blur_half_width[1], mask_z + self.blur_half_width[2]]

					if check_mask == 1:
						blurred_variable_change = np.add( blurred_variable_change,
							( 1. / self.blur_normalization ) *
							( ( self.blur_radius + 1 - np.sqrt( mask_x**2 + mask_y**2 + mask_z**2 ) ) / self.blur_radius ) *
							variable_in[x_bounds[0]:x_bounds[1], y_bounds[0]:y_bounds[1], z_bounds[0]:z_bounds[1]] )


		# todo - unsure about this and same for in the square blur

		blurred_variable[
			start_x : (start_x + x_length),
			start_y : (start_y + y_length),
			start_z : (start_z + z_length)] = blurred_variable_change
		return blurred_variable


