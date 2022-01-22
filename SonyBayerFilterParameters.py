#
# Parameter file for the Sony Bayer Filter optimization
#

import numpy as np

#
# Files
#

project_name_init = 'th30'
project_name = 'angInc_sony_th30_phi0'

# project_name_init = 'normal_sony_baseline_1p989x1p989x2p04um_f1p53um'
# project_name = 'normal_sony_baseline_1p989x1p989x2p04um_f1p53um'


#
# Mesh
#
mesh_spacing_um = 0.017
geometry_spacing_um = 0.051
mesh_to_geometry_factor = int( ( geometry_spacing_um + 0.5 * mesh_spacing_um ) / mesh_spacing_um )

#
# Optical
#
background_index = 1.5
min_device_index = 1.5
max_device_index = 2.4

min_device_permittivity = min_device_index**2
max_device_permittivity = max_device_index**2

init_permittivity_0_1_scale = 0.5


focal_length_um = geometry_spacing_um * 30
focal_plane_center_lateral_um = 0
focal_plane_center_vertical_um = -focal_length_um

#
# Device
#

num_vertical_layers = 40

vertical_layer_height_um = 0.051
vertical_layer_height_voxels = int( vertical_layer_height_um / geometry_spacing_um )

device_size_lateral_um = geometry_spacing_um * 40
device_size_vertical_um = vertical_layer_height_um * num_vertical_layers

device_voxels_lateral = int(device_size_lateral_um / geometry_spacing_um)
device_voxels_vertical = int(device_size_vertical_um / geometry_spacing_um)

device_voxels_simulation_mesh_lateral = 1 + int(device_size_lateral_um / mesh_spacing_um)
# device_voxels_simulation_mesh_lateral = 2 + int(device_size_lateral_um / mesh_spacing_um)
device_voxels_simulation_mesh_vertical = 2 + int(device_size_vertical_um / mesh_spacing_um)

device_vertical_maximum_um = device_size_vertical_um
device_vertical_minimum_um = 0

#
# Spectral
#
lambda_min_um = 0.45
lambda_max_um = 0.65


bandwidth_um = lambda_max_um - lambda_min_um

num_bands = 3
num_points_per_band = 10
num_design_frequency_points = num_bands * num_points_per_band

lambda_values_um = np.linspace(lambda_min_um, lambda_max_um, num_design_frequency_points)
max_intensity_by_wavelength = (device_size_lateral_um**2)**2 / (focal_length_um**2 * lambda_values_um**2)


num_dispersive_ranges = 1
dispersive_range_size_um = bandwidth_um / num_dispersive_ranges

dispersive_ranges_um = [
	[ lambda_min_um, lambda_max_um ]
]

#
# FDTD
#
vertical_gap_size_um = geometry_spacing_um * 15
lateral_gap_size_um = geometry_spacing_um * 10

fdtd_region_size_vertical_um = 2 * vertical_gap_size_um + device_size_vertical_um + focal_length_um
fdtd_region_size_lateral_um = 2 * lateral_gap_size_um + device_size_lateral_um
fdtd_region_maximum_vertical_um = device_size_vertical_um + vertical_gap_size_um
fdtd_region_minimum_vertical_um = -focal_length_um - vertical_gap_size_um

fdtd_simulation_time_fs = 3000

#
# Forward Source
#
lateral_aperture_um = 1.1 * device_size_lateral_um
src_maximum_vertical_um = device_size_vertical_um + vertical_gap_size_um * 2. / 3.
src_minimum_vertical_um = -focal_length_um - 0.5 * vertical_gap_size_um

src_beam_rad = device_size_lateral_um/2
src_angle_incidence = 30 # degrees
src_phi_incidence = 0 # degrees

src_hgt_Si = 0
src_hgt_polymer = 0
n_Si = 3.43
n_polymer = max_device_index



weight_by_individual_wavelength = True

#
# Spectral and polarization selectivity information
#
polarizations_focal_plane_map = [ ['x', 'y'], ['x', 'y'], ['x', 'y'], ['x', 'y'] ]

weight_focal_plane_map_performance_weighting = [ 1.0, 1.0, 1.0, 1.0 ]
weight_focal_plane_map = [ 1.0, 1.0, 1.0, 1.0 ]

if weight_by_individual_wavelength:
	weight_focal_plane_map_performance_weighting = [ 1.0, 1.0, 1.0, 1.0 ]
	weight_focal_plane_map = [ 1.0, 1.0, 1.0, 1.0 ]

polarization_name_to_idx = { 'x':0, 'y':1, 'z':2 }
spectral_focal_plane_map = [
	[0, num_points_per_band],
	[num_points_per_band, 2 * num_points_per_band],
	[2 * num_points_per_band, 3 * num_points_per_band],
	[num_points_per_band, 2 * num_points_per_band]
]

do_rejection = False
if do_rejection:
	half_points = num_points_per_band // 2
	quarter_points = num_points_per_band // 4

	spectral_focal_plane_map = [
		[0, num_points_per_band + half_points ],
		[num_points_per_band - half_points, 2 * num_points_per_band + half_points ],
		[2 * num_points_per_band - half_points, 3 * num_points_per_band],
		[num_points_per_band - half_points, 2 * num_points_per_band + half_points ]
	]

	total_sizes = [ spectral_focal_plane_map[ idx ][ 1 ] - spectral_focal_plane_map[ idx ][ 0 ] for idx in range( 0, len( spectral_focal_plane_map ) ) ]

	reject_sizes = [ 2 * quarter_points, 2 * half_points, 2 * quarter_points, 2 * half_points ]
	accept_sizes = [ total_sizes[ idx ] - reject_sizes[ idx ] for idx in range( 0, len( total_sizes ) ) ]


	reject_weights = [ -0.5 for idx in range( 0, len( reject_sizes ) ) ]
	accept_weights = [ 1 for idx in range( 0, len( accept_sizes ) ) ]

	spectral_focal_plane_map_directional_weights = [
		[ accept_weights[ 0 ] for idx in range( 0, num_points_per_band ) ] +
		[ reject_weights[ 0 ] for idx in range( num_points_per_band, num_points_per_band + half_points ) ],
		
		[ reject_weights[ 1 ] for idx in range( num_points_per_band - half_points, num_points_per_band ) ] +
		[ accept_weights[ 1 ] for idx in range( num_points_per_band, 2 * num_points_per_band ) ] +
		[ reject_weights[ 1 ] for idx in range( 2 * num_points_per_band, 2 * num_points_per_band + half_points ) ],

		[ reject_weights[ 2 ] for idx in range( 2 * num_points_per_band - half_points, 2 * num_points_per_band ) ] +
		[ accept_weights[ 2 ] for idx in range( 2 * num_points_per_band, 3 * num_points_per_band ) ],

		[ reject_weights[ 3 ] for idx in range( num_points_per_band - half_points, num_points_per_band ) ] +
		[ accept_weights[ 3 ] for idx in range( num_points_per_band, 2 * num_points_per_band ) ] +
		[ reject_weights[ 3 ] for idx in range( 2 * num_points_per_band, 2 * num_points_per_band + half_points ) ],
	]

#
# Adjoint sources
#
adjoint_vertical_um = -focal_length_um
num_focal_spots = 4
num_adjoint_sources = num_focal_spots
adjoint_x_positions_um = [device_size_lateral_um / 4., -device_size_lateral_um / 4., -device_size_lateral_um / 4., device_size_lateral_um / 4.]
adjoint_y_positions_um = [device_size_lateral_um / 4., device_size_lateral_um / 4., -device_size_lateral_um / 4., -device_size_lateral_um / 4.]


dispersive_range_to_adjoint_src_map = [
	[ 0, 1, 2, 3 ]
]

adjoint_src_to_dispersive_range_map = [
	0,
	0,
	0,
	0
]

#
# Optimization
#

#
# Alternative is to weight by focal intensity - need to check max intensity weighting thing
#
weight_by_quadrant_transmission = True

num_epochs = 10
num_iterations_per_epoch = 30

use_fixed_step_size = False
fixed_step_size = 2.0


epoch_start_design_change_max = 0.15
epoch_end_design_change_max = 0.05
epoch_range_design_change_max = epoch_start_design_change_max - epoch_end_design_change_max

epoch_start_design_change_min = 0.05
epoch_end_design_change_min = 0
epoch_range_design_change_min = epoch_start_design_change_min - epoch_end_design_change_min

