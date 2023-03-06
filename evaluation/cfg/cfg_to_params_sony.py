import os
import shutil
import sys
import logging
import argparse
import yaml
import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))
from utils import *

#* Logging
# Logger is already initialized through the call to utility.py
logging.info(f'Starting up config loader file: {os.path.basename(__file__)}')
logging.debug(f'Current working directory: {os.getcwd()}')

#* Establish folder structure & location
python_src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
python_src_directory = os.path.dirname(python_src_directory)		# Go up one folder

running_on_local_machine = False
slurm_job_env_variable = os.getenv('SLURM_JOB_NODELIST')
if slurm_job_env_variable is None:
	running_on_local_machine = True


#* Debug Options

start_from_step = 0 	    # 0 if running entire file in one shot


#* Input Arguments
parser = argparse.ArgumentParser(description=\
					"Takes a config file and processes it. Should only ever be called from another Python script, not directly - will trigger import cycle chain otherwise.")
for argument in utility.argparse_universal_args:
	parser.add_argument(argument[0], argument[1], **(argument[2]))
args = parser.parse_args()

yaml_filename = args.filename

#* Load YAML file
with open( yaml_filename, 'rb' ) as yaml_file:
	parameters = yaml.safe_load( yaml_file )
logging.debug(f'Read parameters from file: {yaml_filename}')

#* Create project folder to store all files
projects_directory_location = 'ares_' + os.path.basename(python_src_directory)
if args.override is not None:
    projects_directory_location = args.override
if not os.path.isdir(projects_directory_location):
	os.mkdir(projects_directory_location)

# TODO: Move this to the main.py -----------------------------------------------------------------------------
# #* Output / Save Paths
DATA_FOLDER = projects_directory_location
SAVED_SCRIPTS_FOLDER = os.path.join(projects_directory_location, 'saved_scripts')
OPTIMIZATION_INFO_FOLDER = os.path.join(projects_directory_location, 'opt_info')
OPTIMIZATION_PLOTS_FOLDER = os.path.join(OPTIMIZATION_INFO_FOLDER, 'plots')
DEBUG_COMPLETED_JOBS_FOLDER = 'ares_test_dev'
PULL_COMPLETED_JOBS_FOLDER = projects_directory_location
if running_on_local_machine:
	PULL_COMPLETED_JOBS_FOLDER = DEBUG_COMPLETED_JOBS_FOLDER

EVALUATION_FOLDER = os.path.join(DATA_FOLDER, 'lumproc')
EVALUATION_CONFIG_FOLDER = os.path.join(EVALUATION_FOLDER, 'cfg')
EVALUATION_UTILS_FOLDER = os.path.join(EVALUATION_FOLDER, 'utils')

# parameters['MODEL_PATH'] = os.path.join(projects_directory_location, 'model.pth')
# parameters['OPTIMIZER_PATH'] = os.path.join(projects_directory_location, 'optimizer.pth')


# #* Save out the various files that exist right before the optimization runs for debugging purposes.
# # If these files have changed significantly, the optimization should be re-run to compare to anything new.

if not os.path.isdir( SAVED_SCRIPTS_FOLDER ):
	os.mkdir( SAVED_SCRIPTS_FOLDER )
if not os.path.isdir( OPTIMIZATION_INFO_FOLDER ):
	os.mkdir( OPTIMIZATION_INFO_FOLDER )
if not os.path.isdir( OPTIMIZATION_PLOTS_FOLDER ):
	os.mkdir( OPTIMIZATION_PLOTS_FOLDER )

shutil.copy2( python_src_directory + "/SonyBayerFilterOptimization.py", SAVED_SCRIPTS_FOLDER + "/SonyBayerFilterOptimization.py" )
shutil.copy2( os.path.join(python_src_directory, yaml_filename), 
             os.path.join(SAVED_SCRIPTS_FOLDER, os.path.basename(yaml_filename)) )
shutil.copy2( python_src_directory + "/configs/SonyBayerFilterParameters.py", SAVED_SCRIPTS_FOLDER + "/SonyBayerFilterParameters.py" )
# TODO: et cetera... might have to save out various scripts from each folder
# TODO: Rename this script to SonyBayerFilterParameters.py
# todo: ------------------------------------------------------------------------------------------------------

#  Create convenient folder for evaluation code
shutil.copytree(python_src_directory + "/config", EVALUATION_CONFIG_FOLDER)
shutil.copytree(python_src_directory + "/utils", EVALUATION_UTILS_FOLDER)

#
#* Pull out variables from parameters directly
def get_check_none(param_dict, variable_name):
	variable = param_dict.get(variable_name)
	if variable is None:
		logging.warning(f'variable {variable_name} not present in config file "{yaml_filename}".')
	return variable

# NOTE: Use configs/dict_to_globals_metacode.py to generate this next code block. This is the most Pythonic way to do it.
# The if statement is just so the entire block can be collapsed for readability.
if True:
	start_from_step = get_check_none(parameters, "start_from_step")
	shelf_fn = get_check_none(parameters, "shelf_fn")
	restart_epoch = get_check_none(parameters, "restart_epoch")
	restart_iter = get_check_none(parameters, "restart_iter")
	lumapi_filepath_local = get_check_none(parameters, "lumapi_filepath_local")
	lumapi_filepath_hpc = get_check_none(parameters, "lumapi_filepath_hpc")
	yaml_filename = get_check_none(parameters, "yaml_filename")
	convert_existing = get_check_none(parameters, "convert_existing")
	mesh_spacing_um = get_check_none(parameters, "mesh_spacing_um")
	geometry_spacing_minimum_um = get_check_none(parameters, "geometry_spacing_minimum_um")
	geometry_spacing_lateral_um = get_check_none(parameters, "geometry_spacing_lateral_um")
	device_scale_um = get_check_none(parameters, "device_scale_um")
	use_smooth_blur = get_check_none(parameters, "use_smooth_blur")
	min_feature_size_um = get_check_none(parameters, "min_feature_size_um")
	background_index = get_check_none(parameters, "background_index")
	min_device_index = get_check_none(parameters, "min_device_index")
	max_device_index = get_check_none(parameters, "max_device_index")
	init_permittivity_0_1_scale = get_check_none(parameters, "init_permittivity_0_1_scale")
	focal_plane_center_lateral_um = get_check_none(parameters, "focal_plane_center_lateral_um")
	num_vertical_layers = get_check_none(parameters, "num_vertical_layers")
	vertical_layer_height_um = get_check_none(parameters, "vertical_layer_height_um")
	device_size_lateral_um = get_check_none(parameters, "device_size_lateral_um")
	device_vertical_minimum_um = get_check_none(parameters, "device_vertical_minimum_um")
	objects_above_device = get_check_none(parameters, "objects_above_device")
	num_sidewalls = get_check_none(parameters, "num_sidewalls")
	sidewall_thickness_um = get_check_none(parameters, "sidewall_thickness_um")
	sidewall_material = get_check_none(parameters, "sidewall_material")
	sidewall_extend_focalplane = get_check_none(parameters, "sidewall_extend_focalplane")
	sidewall_vertical_minimum_um = get_check_none(parameters, "sidewall_vertical_minimum_um")
	add_infrared = get_check_none(parameters, "add_infrared")
	lambda_min_um = get_check_none(parameters, "lambda_min_um")
	lambda_max_um = get_check_none(parameters, "lambda_max_um")
	lambda_min_eval_um = get_check_none(parameters, "lambda_min_eval_um")
	lambda_max_eval_um = get_check_none(parameters, "lambda_max_eval_um")
	num_bands = get_check_none(parameters, "num_bands")
	num_points_per_band = get_check_none(parameters, "num_points_per_band")
	num_dispersive_ranges = get_check_none(parameters, "num_dispersive_ranges")
	add_pdaf = get_check_none(parameters, "add_pdaf")
	pdaf_angle_phi_degrees = get_check_none(parameters, "pdaf_angle_phi_degrees")
	pdaf_angle_theta_degrees = get_check_none(parameters, "pdaf_angle_theta_degrees")
	pdaf_adj_src_idx = get_check_none(parameters, "pdaf_adj_src_idx")
	fdtd_simulation_time_fs = get_check_none(parameters, "fdtd_simulation_time_fs")
	weight_by_individual_wavelength = get_check_none(parameters, "weight_by_individual_wavelength")
	use_gaussian_sources = get_check_none(parameters, "use_gaussian_sources")
	use_airy_approximation = get_check_none(parameters, "use_airy_approximation")
	f_number = get_check_none(parameters, "f_number")
	airy_correction_factor = get_check_none(parameters, "airy_correction_factor")
	mid_lambda_um = get_check_none(parameters, "mid_lambda_um")
	source_angle_theta_vacuum_deg = get_check_none(parameters, "source_angle_theta_vacuum_deg")
	source_angle_phi_deg = get_check_none(parameters, "source_angle_phi_deg")
	xy_phi_rotations = get_check_none(parameters, "xy_phi_rotations")
	xy_pol_rotations = get_check_none(parameters, "xy_pol_rotations")
	xy_names = get_check_none(parameters, "xy_names")
	num_focal_spots = get_check_none(parameters, "num_focal_spots")
	dispersive_range_to_adjoint_src_map = get_check_none(parameters, "dispersive_range_to_adjoint_src_map")
	adjoint_src_to_dispersive_range_map = get_check_none(parameters, "adjoint_src_to_dispersive_range_map")
	enforce_xy_gradient_symmetry = get_check_none(parameters, "enforce_xy_gradient_symmetry")
	weight_by_quadrant_transmission = get_check_none(parameters, "weight_by_quadrant_transmission")
	num_epochs = get_check_none(parameters, "num_epochs")
	num_iterations_per_epoch = get_check_none(parameters, "num_iterations_per_epoch")
	use_fixed_step_size = get_check_none(parameters, "use_fixed_step_size")
	fixed_step_size = get_check_none(parameters, "fixed_step_size")
	epoch_start_design_change_max = get_check_none(parameters, "epoch_start_design_change_max")
	epoch_end_design_change_max = get_check_none(parameters, "epoch_end_design_change_max")
	epoch_start_design_change_min = get_check_none(parameters, "epoch_start_design_change_min")
	epoch_end_design_change_min = get_check_none(parameters, "epoch_end_design_change_min")
	fom_types = get_check_none(parameters, "fom_types")
	optimization_fom = get_check_none(parameters, "optimization_fom")
	polarizations_focal_plane_map = get_check_none(parameters, "polarizations_focal_plane_map")
	weight_focal_plane_map_performance_weighting = get_check_none(parameters, "weight_focal_plane_map_performance_weighting")
	weight_focal_plane_map = get_check_none(parameters, "weight_focal_plane_map")
	do_green_balance = get_check_none(parameters, "do_green_balance")
	polarization_name_to_idx = get_check_none(parameters, "polarization_name_to_idx")
	desired_peaks_per_band_um = get_check_none(parameters, "desired_peaks_per_band_um")
	explicit_band_centering = get_check_none(parameters, "explicit_band_centering")
	shuffle_green = get_check_none(parameters, "shuffle_green")
	infrared_center_um = get_check_none(parameters, "infrared_center_um")
	do_rejection = get_check_none(parameters, "do_rejection")
	softplus_kappa = get_check_none(parameters, "softplus_kappa")


#* Environment Variables
if running_on_local_machine:	
	lumapi_filepath =  lumapi_filepath_local
else:	
	#! do not include spaces in filepaths passed to linux
	lumapi_filepath = lumapi_filepath_hpc

#* Process variables

# Device Height with Number of Layers
# Creates a lookup table that adds additional vertical mesh cells depending on the number of layers
lookup_additional_vertical_mesh_cells = {1:3, 2:3, 3:2, 4:3, 5:2, 6:2, 8:3, 10:2,
                                         15:2, 20:2, 30:2, 40:2}

# Mesh
# mesh_to_geometry_factor = int( ( geometry_spacing_um + 0.5 * mesh_spacing_um ) / mesh_spacing_um )	# TODO: Do we need this for back-compatibility?

# Meant for square_blur and square_blur_smooth (used when mesh cells and minimum feature cells don't match)
min_feature_size_voxels = min_feature_size_um / geometry_spacing_lateral_um
blur_half_width_voxels = int( np.ceil( ( min_feature_size_voxels - 1 ) / 2. ) )

# Optical
min_device_permittivity = min_device_index**2
max_device_permittivity = max_device_index**2

#! 20230220 - Empirically, keeping focal length at 3/4 of device length seems the most effective.
focal_length_um = device_scale_um * 30#40#10#20#40#50#30	
focal_plane_center_vertical_um = -focal_length_um

# Device
# vertical_layer_height_voxels = int( vertical_layer_height_um / geometry_spacing_minimum_um )
vertical_layer_height_voxels = int( vertical_layer_height_um / device_scale_um )
device_size_vertical_um = vertical_layer_height_um * num_vertical_layers

device_voxels_lateral = int(np.round( device_size_lateral_um / geometry_spacing_lateral_um))
# device_voxels_vertical = int(np.round( device_size_vertical_um / geometry_spacing_minimum_um))
device_voxels_vertical = int(np.round( device_size_vertical_um / device_scale_um))

device_voxels_simulation_mesh_lateral = 1 + int(device_size_lateral_um / mesh_spacing_um)
# device_voxels_simulation_mesh_lateral = 2 + int(device_size_lateral_um / mesh_spacing_um)
# device_voxels_simulation_mesh_vertical = 2 + int(device_size_vertical_um / mesh_spacing_um)
device_voxels_simulation_mesh_vertical = lookup_additional_vertical_mesh_cells[ num_vertical_layers ] + int(device_size_vertical_um / mesh_spacing_um)

device_vertical_maximum_um = device_size_vertical_um	# NOTE: Unused

# Surrounding          
sidewall_x_positions_um = [device_size_lateral_um / 2 + sidewall_thickness_um / 2, 0, -device_size_lateral_um / 2 - sidewall_thickness_um / 2, 0]
sidewall_y_positions_um = [0, device_size_lateral_um / 2 + sidewall_thickness_um / 2, 0, -device_size_lateral_um / 2 - sidewall_thickness_um / 2]
sidewall_xspan_positions_um = [sidewall_thickness_um, device_size_lateral_um + sidewall_thickness_um * 2,
                               sidewall_thickness_um, device_size_lateral_um + sidewall_thickness_um * 2]
sidewall_yspan_positions_um = [device_size_lateral_um + sidewall_thickness_um * 2, sidewall_thickness_um, 
                               device_size_lateral_um + sidewall_thickness_um * 2, sidewall_thickness_um]
# sidewall_vertical_minimum_um = device_vertical_minimum_um

# Spectral

if add_infrared:
	lambad_max_um = 1.0
	lambda_max_eval_um = 1.0
bandwidth_um = lambda_max_um - lambda_min_um

num_design_frequency_points = num_bands * num_points_per_band
num_eval_frequency_points = 6 * num_design_frequency_points

lambda_values_um = np.linspace(lambda_min_um, lambda_max_um, num_design_frequency_points)
# Following derivation is a normalization factor for the intensity depending on frequency, so as to ensure equal weighting.
max_intensity_by_wavelength = (device_size_lateral_um**2)**2 / (focal_length_um**2 * lambda_values_um**2)
# See https://en.wikipedia.org/wiki/Airy_disk#:~:text=at%20the%20center%20of%20the%20diffraction%20pattern

dispersive_range_size_um = bandwidth_um / num_dispersive_ranges
dispersive_ranges_um = [
					[ lambda_min_um, lambda_max_um ]
				]

# PDAF Functionality
if add_pdaf:
	assert device_voxels_lateral % 2 == 0, "Expected an even number here for PDAF implementation ease!"
	assert ( not add_infrared ), "For now, we are just going to look at RGB designs!"

pdaf_focal_spot_x_um = 0.25 * device_size_lateral_um
pdaf_focal_spot_y_um = 0.25 * device_size_lateral_um

pdaf_lambda_min_um = lambda_min_um
pdaf_lambda_max_um = lambda_min_um + bandwidth_um / 3.

pdaf_lambda_values_um = np.linspace(pdaf_lambda_min_um, pdaf_lambda_max_um, num_design_frequency_points)
pdaf_max_intensity_by_wavelength = (device_size_lateral_um**2)**2 / (focal_length_um**2 * pdaf_lambda_values_um**2)

# FDTD
vertical_gap_size_um = geometry_spacing_lateral_um * 15
lateral_gap_size_um = device_scale_um * 10 #25#50
# vertical_gap_size_um = geometry_spacing_minimum_um * 15
# lateral_gap_size_um = geometry_spacing_minimum_um * 10#25#50

fdtd_region_size_vertical_um = 2 * vertical_gap_size_um + device_size_vertical_um + focal_length_um
fdtd_region_size_lateral_um = 2 * lateral_gap_size_um + device_size_lateral_um
fdtd_region_maximum_vertical_um = device_size_vertical_um + vertical_gap_size_um
fdtd_region_minimum_vertical_um = -focal_length_um - vertical_gap_size_um

# Forward (Input) Source
lateral_aperture_um = 1.1 * device_size_lateral_um
src_maximum_vertical_um = device_size_vertical_um + vertical_gap_size_um * 2. / 3.
src_minimum_vertical_um = -focal_length_um - 0.5 * vertical_gap_size_um

mid_lambda_um = (lambda_min_um + lambda_max_um)/2

gaussian_waist_radius_um = 0.5 * ( 0.5 * 2.04 )		# half a quadrant width
# take f number = 2.2, check index calc
# # gaussian_waist_radius_um = airy_correction_factor * mid_lambda_um / ( np.pi * ( 1. / ( 2 * f_number ) ) ) 
# 							 = 0.84 * 0.55 * 2.2 / (np.pi/2)
if use_airy_approximation:
	gaussian_waist_radius_um = airy_correction_factor * mid_lambda_um * f_number
else:
	gaussian_waist_radius_um = mid_lambda_um / ( np.pi * ( 1. / ( 2 * f_number ) ) )
# TODO: Check which one we decided on in the end - Airy disk of Gaussian on input aperture, or f-number determining divergence half-angle?
# TODO: Add frequency dependent profile?

source_angle_theta_rad = np.arcsin( np.sin( source_angle_theta_vacuum_deg * np.pi / 180. ) * 1.0 / background_index )
source_angle_theta_deg = source_angle_theta_rad * 180. / np.pi

# Adjoint Sources
adjoint_vertical_um = -focal_length_um
num_adjoint_sources = num_focal_spots
adjoint_x_positions_um = [device_size_lateral_um / 4., -device_size_lateral_um / 4., -device_size_lateral_um / 4., device_size_lateral_um / 4.]
adjoint_y_positions_um = [device_size_lateral_um / 4., device_size_lateral_um / 4., -device_size_lateral_um / 4., -device_size_lateral_um / 4.]

# Optimization
if enforce_xy_gradient_symmetry:
	assert ( int( np.round( device_size_lateral_um / geometry_spacing_lateral_um ) ) % 2 ) == 1, "We should have an odd number of design voxels across for this operation."
assert not enforce_xy_gradient_symmetry, "20230220 Greg - This feature seems to be making things worse - I think there is a small inherent symmetry in the simulation/import/monitors"

epoch_range_design_change_max = epoch_start_design_change_max - epoch_end_design_change_max
epoch_range_design_change_min = epoch_start_design_change_min - epoch_end_design_change_min

# Spectral and polarization selectivity information
if weight_by_individual_wavelength:
	weight_focal_plane_map_performance_weighting = [ 1.0, 1.0, 1.0, 1.0 ]
	weight_focal_plane_map = [ 1.0, 1.0, 1.0, 1.0 ]
	weight_individual_wavelengths = np.ones( len( lambda_values_um ) )
 
# Determine the wavelengths that will be directed to each focal area
spectral_focal_plane_map = [
	[0, num_points_per_band],
	[num_points_per_band, 2 * num_points_per_band],
	[2 * num_points_per_band, 3 * num_points_per_band],
	[num_points_per_band, 2 * num_points_per_band]
]

# Find indices of desired peak locations in lambda_values_um
desired_peak_location_per_band = [ np.argmin( ( lambda_values_um - desired_peaks_per_band_um[ idx ] )**2 ) for idx in range( 0, len( desired_peaks_per_band_um ) ) ]
# logging.info(f'Desired peak locations per band are: {desired_peak_location_per_band}')

# for idx in range( 0, len( desired_peak_location_per_band ) ):
# 	get_location = desired_peak_location_per_band[ idx ]
# 	half_width = num_points_per_band // 2
# 	assert ( get_location - half_width ) >= 0, "Not enough optimization room in short wavelength region!"
# 	assert ( get_location + half_width + 1 ) <= num_design_frequency_points, "Not enough optimization room in long wavelength region!"

# spectral_focal_plane_map = [
# 	[ desired_peak_location_per_band[ 0 ] - ( num_points_per_band // 2 ), desired_peak_location_per_band[ 0 ] + 1 + ( num_points_per_band // 2 ) ],
# 	[ desired_peak_location_per_band[ 1 ] - ( num_points_per_band // 2 ), desired_peak_location_per_band[ 1 ] + 1 + ( num_points_per_band // 2 ) ],
# 	[ desired_peak_location_per_band[ 2 ] - ( num_points_per_band // 2 ), desired_peak_location_per_band[ 2 ] + 1 + ( num_points_per_band // 2 ) ],
# 	[ desired_peak_location_per_band[ 1 ] - ( num_points_per_band // 2 ), desired_peak_location_per_band[ 1 ] + 1 + ( num_points_per_band // 2 ) ]
# ]


if explicit_band_centering:
    # Determine the wavelengths that will be directed to each focal area
	spectral_focal_plane_map = [
		[0, num_design_frequency_points],
		[0, num_design_frequency_points],
		[0, num_design_frequency_points],
		[0, num_design_frequency_points]
	]


	bandwidth_left_by_quad = [ 
		( desired_peak_location_per_band[ 1 ] - desired_peak_location_per_band[ 0 ] ) // 2,
		( desired_peak_location_per_band[ 1 ] - desired_peak_location_per_band[ 0 ] ) // 2,
		( desired_peak_location_per_band[ 2 ] - desired_peak_location_per_band[ 1 ] ) // 2,
		( desired_peak_location_per_band[ 1 ] - desired_peak_location_per_band[ 0 ] ) // 2
	]

	bandwidth_right_by_quad = [ 
		( desired_peak_location_per_band[ 1 ] - desired_peak_location_per_band[ 0 ] ) // 2,
		( desired_peak_location_per_band[ 2 ] - desired_peak_location_per_band[ 1 ] ) // 2,
		( desired_peak_location_per_band[ 2 ] - desired_peak_location_per_band[ 1 ] ) // 2,
		( desired_peak_location_per_band[ 2 ] - desired_peak_location_per_band[ 1 ] ) // 2
	]

	bandwidth_left_by_quad = [ bandwidth_left_by_quad[ idx ] / 1.75 for idx in range( 0, 4 ) ]
	bandwidth_right_by_quad = [ bandwidth_right_by_quad[ idx ] / 1.75 for idx in range( 0, 4 ) ]

	quad_to_band = [ 0, 1, 2, 1 ]
	if shuffle_green:
		quad_to_band = [ 0, 1, 1, 2 ]

	weight_individual_wavelengths_by_quad = np.zeros((4,num_design_frequency_points))
	for quad_idx in range( 0, 4 ):
		bandwidth_left = bandwidth_left_by_quad[ quad_to_band[ quad_idx ] ]
		bandwidth_right = bandwidth_right_by_quad[ quad_to_band[ quad_idx ] ]

		band_center = desired_peak_location_per_band[ quad_to_band[ quad_idx ] ]

		for wl_idx in range( 0, num_design_frequency_points ):

			choose_width = bandwidth_right
			if wl_idx < band_center:
				choose_width = bandwidth_left


			scaling_exp = -1. / np.log( 0.5 )
			weight_individual_wavelengths_by_quad[ quad_idx, wl_idx ] = np.exp( -( wl_idx - band_center )**2 / ( scaling_exp * choose_width**2 ) )


	# import matplotlib.pyplot as plt
	# colors = [ 'b', 'g', 'r', 'm' ]
	# for quad_idx in range( 0, 4 ):

	# 	plt.plot( lambda_values_um, weight_individual_wavelengths_by_quad[ quad_idx ], color=colors[ quad_idx ] )
	# plt.show()


if add_infrared:
	wl_idx_ir_start = 0
	for wl_idx in range( 0, len( lambda_values_um ) ):
		if lambda_values_um[ wl_idx ] > infrared_center_um:
			wl_idx_ir_start = wl_idx - 1
			break

	# Determine the wavelengths that will be directed to each focal area
	spectral_focal_plane_map = [ 
		[ 0, 2 + ( num_points_per_band // 2 ) ],
		[ 2 + ( num_points_per_band // 2 ), 4 + num_points_per_band ],
		[ 4 + num_points_per_band, 6 + num_points_per_band + ( num_points_per_band // 2 ) ],
		[ wl_idx_ir_start, ( wl_idx_ir_start + 3 ) ]
	]

	ir_band_equalization_weights = [
		1. / ( spectral_focal_plane_map[ idx ][ 1 ] - spectral_focal_plane_map[ idx ][ 0 ] ) for idx in range( 0, len( spectral_focal_plane_map ) ) ]

	weight_individual_wavelengths = np.ones( len( lambda_values_um ) )
	for band_idx in range( 0, len( spectral_focal_plane_map ) ):

		for wl_idx in range( spectral_focal_plane_map[ band_idx ][ 0 ], spectral_focal_plane_map[ band_idx ][ 1 ] ):
			weight_individual_wavelengths[ wl_idx ] = ir_band_equalization_weights[ band_idx ]


if do_rejection:

	# def weight_accept( transmission ):
	# 	return transmission
	# def weight_reject( transmission ):
	# 	return ( 1 - transmission )

	# performance_weighting_functions = [
	# 	[ weight_accept for idx in range( 0, num_points_per_band ) ] + [ weight_reject for idx in range( 0, 2 * num_points_per_band ) ],
	# 	[ weight_reject for idx in range( 0, num_points_per_band ) ] + [ weight_accept for idx in range( 0, num_points_per_band ) ] + [ weight_reject for idx in range( 0, num_points_per_band ) ],
	# 	[ weight_reject for idx in range( 0, 2 * num_points_per_band ) ] + [ weight_accept for idx in range( 0, num_points_per_band ) ],
	# 	[ weight_reject for idx in range( 0, num_points_per_band ) ] + [ weight_accept for idx in range( 0, num_points_per_band ) ] + [ weight_reject for idx in range( 0, num_points_per_band ) ]
	# ]

	# Determine the wavelengths that will be directed to each focal area
	spectral_focal_plane_map = [
		[0, num_design_frequency_points],
		[0, num_design_frequency_points],
		[0, num_design_frequency_points],
		[0, num_design_frequency_points]
	]

	desired_band_width_um = 0.04#0.08#0.12#0.18


	wl_per_step_um = lambda_values_um[ 1 ] - lambda_values_um[ 0 ]

	#
	# weights will be 1 at the center and drop to 0 at the edge of the band
	#

	weighting_by_band = np.zeros( ( num_bands, num_design_frequency_points ) )

	for band_idx in range( 0, num_bands ):
		wl_center_um = desired_peaks_per_band_um[ band_idx ]

		for wl_idx in range( 0, num_design_frequency_points ):
			wl_um = lambda_values_um[ wl_idx ]
			weight = -0.5 + 1.5 * np.exp( -( ( wl_um - wl_center_um )**2 / ( desired_band_width_um**2 ) ) )
			# weight = -1 + 2 * np.exp( -( ( wl_um - wl_center_um )**2 / ( desired_band_width_um**2 ) ) )
			# weight = -2 + 3 * np.exp( -( ( wl_um - wl_center_um )**2 / ( desired_band_width_um**2 ) ) )

			weighting_by_band[ band_idx, wl_idx ] = weight

	#
	# todo: we need to sort out this weighting scheme and how it works with the performance weighting scheme.
	# we have seen this problem before, but the problem is that this weighting in the FoM will end up bumping
	# up the apparent performance (in the FoM calculation), but as this apparent performance increases, the
	# performance weighting scheme will counteract it and give a lower performance weight.  This, then, fights
	# against the weight the gradient gets from these FoM bumps.  
	#
	spectral_focal_plane_map_directional_weights = np.zeros( ( 4, num_design_frequency_points ) )
	spectral_focal_plane_map_directional_weights[ 0, : ] = weighting_by_band[ 0 ]
	spectral_focal_plane_map_directional_weights[ 1, : ] = weighting_by_band[ 1 ]
	spectral_focal_plane_map_directional_weights[ 2, : ] = weighting_by_band[ 2 ]
	spectral_focal_plane_map_directional_weights[ 3, : ] = weighting_by_band[ 1 ]

	# import matplotlib.pyplot as plt
	# plt.plot( lambda_values_um, spectral_focal_plane_map_directional_weights[ 0 ], color='b', linewidth=2 )
	# plt.plot( lambda_values_um, spectral_focal_plane_map_directional_weights[ 1 ], color='g', linewidth=2 )
	# plt.plot( lambda_values_um, spectral_focal_plane_map_directional_weights[ 2 ], color='r', linewidth=2 )
	# plt.plot( lambda_values_um, spectral_focal_plane_map_directional_weights[ 3 ], color='m', linewidth=2, linestyle='--' )
	# plt.axhline( y=0, linestyle=':', color='k' )
	# plt.show()




	# half_points = num_points_per_band // 2
	# quarter_points = num_points_per_band // 4

	# Determine the wavelengths that will be directed to each focal area
	# spectral_focal_plane_map = [
	# 	[0, num_points_per_band + half_points ],
	# 	[num_points_per_band - half_points, 2 * num_points_per_band + half_points ],
	# 	[2 * num_points_per_band - half_points, 3 * num_points_per_band],
	# 	[num_points_per_band - half_points, 2 * num_points_per_band + half_points ]
	# ]

	# total_sizes = [ spectral_focal_plane_map[ idx ][ 1 ] - spectral_focal_plane_map[ idx ][ 0 ] for idx in range( 0, len( spectral_focal_plane_map ) ) ]

	# reject_sizes = [ half_points, 2 * half_points, half_points, 2 * half_points ]
	# accept_sizes = [ total_sizes[ idx ] - reject_sizes[ idx ] for idx in range( 0, len( total_sizes ) ) ]

	# reject_weights = [ -1 for idx in range( 0, len( reject_sizes ) ) ]
	# accept_weights = [ 1 for idx in range( 0, len( accept_sizes ) ) ]

	# spectral_focal_plane_map_directional_weights = [
	# 	[ accept_weights[ 0 ] for idx in range( 0, num_points_per_band ) ] +
	# 	[ reject_weights[ 0 ] for idx in range( num_points_per_band, num_points_per_band + half_points ) ],
		
	# 	[ reject_weights[ 1 ] for idx in range( num_points_per_band - half_points, num_points_per_band ) ] +
	# 	[ accept_weights[ 1 ] for idx in range( num_points_per_band, 2 * num_points_per_band ) ] +
	# 	[ reject_weights[ 1 ] for idx in range( 2 * num_points_per_band, 2 * num_points_per_band + half_points ) ],

	# 	[ reject_weights[ 2 ] for idx in range( 2 * num_points_per_band - half_points, 2 * num_points_per_band ) ] +
	# 	[ accept_weights[ 2 ] for idx in range( 2 * num_points_per_band, 3 * num_points_per_band - 4 ) ] +
	# 	[ reject_weights[ 2 ] for idx in range( 3 * num_points_per_band - 4, 3 * num_points_per_band ) ],

	# 	[ reject_weights[ 3 ] for idx in range( num_points_per_band - half_points, num_points_per_band ) ] +
	# 	[ accept_weights[ 3 ] for idx in range( num_points_per_band, 2 * num_points_per_band ) ] +
	# 	[ reject_weights[ 3 ] for idx in range( 2 * num_points_per_band, 2 * num_points_per_band + half_points ) ],
	# ]


logging.info('Config imported and processed.')