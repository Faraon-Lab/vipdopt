# Copyright © 2023, California Institute of Technology. All rights reserved.
#
# Use in source and binary forms for nonexclusive, nonsublicenseable, commercial purposes with or without modification, is permitted provided that the following conditions are met:
# - Use of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# - Use in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the software.
# - Neither the name of the California Institute of Technology (Caltech) nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import os
import shutil
import sys
import logging
import argparse
import yaml
import numpy as np
from types import SimpleNamespace

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())
from utils import utility

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
# start_from_step = 0 	    #NOTE: Overwritten by config anyway   # 0 if running entire file in one shot


#* Input Arguments
parser = argparse.ArgumentParser(description=\
					"Takes a config file and processes it. Should only ever be called from another Python script, not directly - will trigger import cycle chain otherwise.")
for argument in utility.argparse_universal_args:
	parser.add_argument(argument[0], argument[1], **(argument[2]))
args = parser.parse_args()

#* Load YAML file
yaml_filename = args.filename
with open( yaml_filename, 'rb' ) as yaml_file:
	parameters = yaml.safe_load( yaml_file )
logging.debug(f'Read parameters from file: {yaml_filename}')

#* Create project folder to store all files
projects_directory_location = 'ares_' + os.path.basename(python_src_directory)
if args.override is not None:
	projects_directory_location = args.override
if not os.path.isdir(projects_directory_location):
	os.mkdir(projects_directory_location)


#
#* Pull out variables from parameters directly
# We pull raw parameters from config and store in globals() but also in its own dictionary.
#

def get_check_none(param_dict, variable_name):
	variable = param_dict.get(variable_name)
	if variable is None:
		logging.warning(f'variable {variable_name} not present in config file "{yaml_filename}".')
	return variable

def get_config_vars(param_dict):
	'''Pulls raw config variables from the prescribed config YAML file. Uses dict.get() to make sure that there is a None assignment
 	in case the script asks for a variable not defined in the given config.'''
	# NOTE: Use configs/dict_to_globals_metacode.py to generate this next code block. This is the most Pythonic way to do it.
	
	# Record all the variables that are not pulled from the config
	not_config_keys = set(dir())
	
	start_from_step = get_check_none(param_dict, "start_from_step")
	shelf_fn = get_check_none(param_dict, "shelf_fn")
	restart_epoch = get_check_none(param_dict, "restart_epoch")
	restart_iter = get_check_none(param_dict, "restart_iter")
	lumapi_filepath_local = get_check_none(param_dict, "lumapi_filepath_local")
	lumapi_filepath_hpc = get_check_none(param_dict, "lumapi_filepath_hpc")
	convert_existing = get_check_none(param_dict, "convert_existing")
	mesh_spacing_um = get_check_none(param_dict, "mesh_spacing_um")
	geometry_spacing_lateral_um = get_check_none(param_dict, "geometry_spacing_lateral_um")
	device_scale_um = get_check_none(param_dict, "device_scale_um")
	use_smooth_blur = get_check_none(param_dict, "use_smooth_blur")
	min_feature_size_um = get_check_none(param_dict, "min_feature_size_um")
	background_index = get_check_none(param_dict, "background_index")
	min_device_index = get_check_none(param_dict, "min_device_index")
	max_device_index = get_check_none(param_dict, "max_device_index")
	init_permittivity_0_1_scale = get_check_none(param_dict, "init_permittivity_0_1_scale")
	binarize_3_levels = get_check_none(param_dict, "binarize_3_levels")
	focal_plane_center_lateral_um = get_check_none(param_dict, "focal_plane_center_lateral_um")
	num_vertical_layers = get_check_none(param_dict, "num_vertical_layers")
	vertical_layer_height_um = get_check_none(param_dict, "vertical_layer_height_um")
	device_size_lateral_um = get_check_none(param_dict, "device_size_lateral_um")
	device_vertical_minimum_um = get_check_none(param_dict, "device_vertical_minimum_um")
	objects_above_device = get_check_none(param_dict, "objects_above_device")
	num_sidewalls = get_check_none(param_dict, "num_sidewalls")
	sidewall_material = get_check_none(param_dict, "sidewall_material")
	sidewall_extend_pml = get_check_none(param_dict, "sidewall_extend_pml")
	sidewall_extend_focalplane = get_check_none(param_dict, "sidewall_extend_focalplane")
	sidewall_thickness_um = get_check_none(param_dict, "sidewall_thickness_um")
	sidewall_vertical_minimum_um = get_check_none(param_dict, "sidewall_vertical_minimum_um")
	add_infrared = get_check_none(param_dict, "add_infrared")
	lambda_min_um = get_check_none(param_dict, "lambda_min_um")
	lambda_max_um = get_check_none(param_dict, "lambda_max_um")
	lambda_min_eval_um = get_check_none(param_dict, "lambda_min_eval_um")
	lambda_max_eval_um = get_check_none(param_dict, "lambda_max_eval_um")
	num_bands = get_check_none(param_dict, "num_bands")
	num_points_per_band = get_check_none(param_dict, "num_points_per_band")
	num_dispersive_ranges = get_check_none(param_dict, "num_dispersive_ranges")
	add_pdaf = get_check_none(param_dict, "add_pdaf")
	pdaf_angle_phi_degrees = get_check_none(param_dict, "pdaf_angle_phi_degrees")
	pdaf_angle_theta_degrees = get_check_none(param_dict, "pdaf_angle_theta_degrees")
	pdaf_adj_src_idx = get_check_none(param_dict, "pdaf_adj_src_idx")
	fdtd_simulation_time_fs = get_check_none(param_dict, "fdtd_simulation_time_fs")
	weight_by_individual_wavelength = get_check_none(param_dict, "weight_by_individual_wavelength")
	use_gaussian_sources = get_check_none(param_dict, "use_gaussian_sources")
	use_airy_approximation = get_check_none(param_dict, "use_airy_approximation")
	f_number = get_check_none(param_dict, "f_number")
	airy_correction_factor = get_check_none(param_dict, "airy_correction_factor")
	beam_size_multiplier = get_check_none(param_dict, "beam_size_multiplier")
	source_angle_theta_vacuum_deg = get_check_none(param_dict, "source_angle_theta_vacuum_deg")
	source_angle_phi_deg = get_check_none(param_dict, "source_angle_phi_deg")
	xy_phi_rotations = get_check_none(param_dict, "xy_phi_rotations")
	xy_pol_rotations = get_check_none(param_dict, "xy_pol_rotations")
	xy_names = get_check_none(param_dict, "xy_names")
	use_source_aperture = get_check_none(param_dict, "use_source_aperture")
	num_focal_spots = get_check_none(param_dict, "num_focal_spots")
	dispersive_range_to_adjoint_src_map = get_check_none(param_dict, "dispersive_range_to_adjoint_src_map")
	adjoint_src_to_dispersive_range_map = get_check_none(param_dict, "adjoint_src_to_dispersive_range_map")
	enforce_xy_gradient_symmetry = get_check_none(param_dict, "enforce_xy_gradient_symmetry")
	weight_by_quadrant_transmission = get_check_none(param_dict, "weight_by_quadrant_transmission")
	num_epochs = get_check_none(param_dict, "num_epochs")
	num_iterations_per_epoch = get_check_none(param_dict, "num_iterations_per_epoch")
	optimizer_algorithm = get_check_none(param_dict, "optimizer_algorithm")
	adam_betas = get_check_none(param_dict, "adam_betas")
	use_fixed_step_size = get_check_none(param_dict, "use_fixed_step_size")
	fixed_step_size = get_check_none(param_dict, "fixed_step_size")
	epoch_start_design_change_max = get_check_none(param_dict, "epoch_start_design_change_max")
	epoch_end_design_change_max = get_check_none(param_dict, "epoch_end_design_change_max")
	epoch_start_design_change_min = get_check_none(param_dict, "epoch_start_design_change_min")
	epoch_end_design_change_min = get_check_none(param_dict, "epoch_end_design_change_min")
	fom_types = get_check_none(param_dict, "fom_types")
	optimization_fom = get_check_none(param_dict, "optimization_fom")
	reinterpolate_permittivity = get_check_none(param_dict, "reinterpolate_permittivity")
	reinterpolate_permittivity_factor = get_check_none(param_dict, "reinterpolate_permittivity_factor")
	reinterpolate_type = get_check_none(param_dict, "reinterpolate_type")
	border_optimization = get_check_none(param_dict, "border_optimization")
	layer_gradient = get_check_none(param_dict, "layer_gradient")
	flip_gradient = get_check_none(param_dict, "flip_gradient")
	polarizations_focal_plane_map = get_check_none(param_dict, "polarizations_focal_plane_map")
	weight_focal_plane_map_performance_weighting = get_check_none(param_dict, "weight_focal_plane_map_performance_weighting")
	weight_focal_plane_map = get_check_none(param_dict, "weight_focal_plane_map")
	do_green_balance = get_check_none(param_dict, "do_green_balance")
	polarization_name_to_idx = get_check_none(param_dict, "polarization_name_to_idx")
	desired_peaks_per_band_um = get_check_none(param_dict, "desired_peaks_per_band_um")
	explicit_band_centering = get_check_none(param_dict, "explicit_band_centering")
	shuffle_green = get_check_none(param_dict, "shuffle_green")
	infrared_center_um = get_check_none(param_dict, "infrared_center_um")
	do_rejection = get_check_none(param_dict, "do_rejection")
	softplus_kappa = get_check_none(param_dict, "softplus_kappa")
	evaluate_bordered_extended = get_check_none(param_dict, "evaluate_bordered_extended")
	do_normalize = get_check_none(param_dict, "do_normalize")
	should_fix_layers = get_check_none(param_dict, "should_fix_layers")
	fixed_layers = get_check_none(param_dict, "fixed_layers")
	fixed_layers_density = get_check_none(param_dict, "fixed_layers_density")




	# # Non-Pythonic way:
	# # But don't do this. For back-compatibility, we want there to be a None object if we call a variable that is not defined in raw config
	# locals().update(**param_dict)

	# Grab all current variables and exclude the ones from before - set up a dictionary to use for post-processing
	config_keys = set(dir()) - not_config_keys
	config_vars = {}
	for key in config_keys:
		if key not in ["not_config_keys"]:
			config_vars[key] = eval(key)
   
	return config_vars

# Store all raw config variables in a dictionary.
config_vars = get_config_vars(parameters)
# Assign to a SimpleNamespace for easy code readability.
cv = SimpleNamespace(**config_vars)

#* Environment Variables
if running_on_local_machine:	
	lumapi_filepath =  cv.lumapi_filepath_local
else:	
	#! do not include spaces in filepaths passed to linux
	lumapi_filepath = cv.lumapi_filepath_hpc

#* Process variables
# Here we pass config_vars as a dictionary to a function that does all relevant post-processing.
# This way the post-processing of raw config can be accessed repeatably and in a modular fashion.
# Most common use-case is for a plot-producing script that needs to regenerate the exact same variables -
# Then we don't need to change the same code in two different files.

def post_process_config_vars(**config_dict):		# cd: Config Dictionary
	'''All post-processing of raw config variables should be performed here. This covers anything that cannot be defined in
	a YAML file, such as calculations (e.g. degrees to radians).'''
	cd = SimpleNamespace(**config_dict)

	# Record all variables in locals() that are not yet post-processed, so as to filter them out later
	not_output_keys = set(dir())

	# Device Height with Number of Layers
	# Creates a lookup table that adds additional vertical mesh cells depending on the number of layers
	lookup_additional_vertical_mesh_cells = {1:3, 2:3, 3:2, 4:3, 5:2, 6:2, 8:3, 10:2,
											15:2, 20:2, 30:2, 40:2}

	# Mesh
	# mesh_to_geometry_factor = int( ( geometry_spacing_um + 0.5 * mesh_spacing_um ) / mesh_spacing_um )	# TODO: Do we need this for back-compatibility?

	# Meant for square_blur and square_blur_smooth (used when mesh cells and minimum feature cells don't match)
	min_feature_size_voxels = cd.min_feature_size_um / cd.geometry_spacing_lateral_um
	blur_half_width_voxels = int( np.ceil( ( min_feature_size_voxels - 1 ) / 2. ) )

	# Optical
	min_device_permittivity = cd.min_device_index**2
	max_device_permittivity = cd.max_device_index**2

	#! 20230220 - Empirically, keeping focal length at 3/4 of device length seems the most effective.
	focal_length_um = cd.device_scale_um * 30#40#10#20#40#50#30	
	focal_plane_center_vertical_um = -focal_length_um

	# Device
	# vertical_layer_height_voxels = int( vertical_layer_height_um / geometry_spacing_minimum_um )
	vertical_layer_height_voxels = int( cd.vertical_layer_height_um / cd.device_scale_um )
	device_size_vertical_um = cd.vertical_layer_height_um * cd.num_vertical_layers

	device_voxels_lateral = int(np.round( cd.device_size_lateral_um / cd.geometry_spacing_lateral_um))
	# device_voxels_vertical = int(np.round( device_size_vertical_um / geometry_spacing_minimum_um))
	device_voxels_vertical = int(np.round( device_size_vertical_um / cd.device_scale_um))

	device_voxels_simulation_mesh_lateral = 1 + int(cd.device_size_lateral_um / cd.mesh_spacing_um)
	# device_voxels_simulation_mesh_lateral = 2 + int(device_size_lateral_um / mesh_spacing_um)
	# device_voxels_simulation_mesh_vertical = 2 + int(device_size_vertical_um / mesh_spacing_um)
	device_voxels_simulation_mesh_vertical = lookup_additional_vertical_mesh_cells[ cd.num_vertical_layers ] + \
	 													int(device_size_vertical_um / cd.mesh_spacing_um)
    # TODO: 20230720 Ian - Need to write something that automatically corrects for this. It's taking WAY too much time
    # todo: to manually guess-and-check this array mismatch.

	device_vertical_maximum_um = device_size_vertical_um	# NOTE: Unused

	#! new
	device_size_lateral_bordered_um = cd.device_size_lateral_um
	device_voxels_lateral_bordered = device_voxels_lateral
	device_voxels_simulation_mesh_lateral_bordered = device_voxels_simulation_mesh_lateral
	
	border_size_um = 5 * cd.device_scale_um		# new
	border_size_voxels = int( np.round( border_size_um / cd.geometry_spacing_lateral_um ) )		# new
	assert not ( cd.border_optimization and cd.use_smooth_blur ), "This combination is not supported!"	# new
	if cd.evaluate_bordered_extended:
		assert cd.border_optimization, 'Expecting border optimization to be true for this mode'
		border_size_um = cd.device_size_lateral_um

	if cd.border_optimization:
		device_size_lateral_bordered_um += 2 * border_size_um
		device_voxels_lateral_bordered = int(np.round( device_size_lateral_bordered_um / cd.geometry_spacing_lateral_um))
		device_voxels_simulation_mesh_lateral_bordered = int(device_size_lateral_bordered_um / cd.mesh_spacing_um) + 1 #1 if mesh_spacing_um == 0.017

	# FDTD
	vertical_gap_size_um = cd.geometry_spacing_lateral_um * 15
	lateral_gap_size_um = cd.device_scale_um * 10 #25#50
	# vertical_gap_size_um = geometry_spacing_minimum_um * 15
	# lateral_gap_size_um = geometry_spacing_minimum_um * 10#25#50

	fdtd_region_size_vertical_um = 2 * vertical_gap_size_um + device_size_vertical_um + focal_length_um
	fdtd_region_size_lateral_um = 2 * lateral_gap_size_um + cd.device_size_lateral_um
	fdtd_region_maximum_vertical_um = device_size_vertical_um + vertical_gap_size_um
	fdtd_region_minimum_vertical_um = -focal_length_um - vertical_gap_size_um

	# Surrounding

	pec_aperture_thickness_um = 3 * cd.mesh_spacing_um #! new
	assert not ( cd.border_optimization and cd.num_sidewalls != 0 ), "This combination is not supported!"	#! new
	  
	if cd.sidewall_extend_pml:
		cd.sidewall_thickness_um = (fdtd_region_size_lateral_um - cd.device_size_lateral_um)/2
	sidewall_x_positions_um = [cd.device_size_lateral_um / 2 + cd.sidewall_thickness_um / 2, 0, -cd.device_size_lateral_um / 2 - cd.sidewall_thickness_um / 2, 0]
	sidewall_y_positions_um = [0, cd.device_size_lateral_um / 2 + cd.sidewall_thickness_um / 2, 0, -cd.device_size_lateral_um / 2 - cd.sidewall_thickness_um / 2]
	sidewall_xspan_positions_um = [cd.sidewall_thickness_um, cd.device_size_lateral_um + cd.sidewall_thickness_um * 2,
								cd.sidewall_thickness_um, cd.device_size_lateral_um + cd.sidewall_thickness_um * 2]
	sidewall_yspan_positions_um = [cd.device_size_lateral_um + cd.sidewall_thickness_um * 2, cd.sidewall_thickness_um, 
								cd.device_size_lateral_um + cd.sidewall_thickness_um * 2, cd.sidewall_thickness_um]

	# Spectral

	if cd.add_infrared:
		cd.lambda_max_um = 1.0
		cd.lambda_max_eval_um = 1.0
	bandwidth_um = cd.lambda_max_um - cd.lambda_min_um

	num_design_frequency_points = cd.num_bands * cd.num_points_per_band
	num_eval_frequency_points = 6 * num_design_frequency_points

	lambda_values_um = np.linspace(cd.lambda_min_um, cd.lambda_max_um, num_design_frequency_points)
	# Following derivation is a normalization factor for the intensity depending on frequency, so as to ensure equal weighting.
	max_intensity_by_wavelength = (cd.device_size_lateral_um**2)**2 / (focal_length_um**2 * lambda_values_um**2)
	# See https://en.wikipedia.org/wiki/Airy_disk#:~:text=at%20the%20center%20of%20the%20diffraction%20pattern

	dispersive_range_size_um = bandwidth_um / cd.num_dispersive_ranges
	dispersive_ranges_um = [
						[ cd.lambda_min_um, cd.lambda_max_um ]
					]

	# PDAF Functionality
	if cd.add_pdaf:
		assert cd.device_voxels_lateral % 2 == 0, "Expected an even number here for PDAF implementation ease!"
		assert ( not cd.add_infrared ), "For now, we are just going to look at RGB designs!"

	pdaf_focal_spot_x_um = 0.25 * cd.device_size_lateral_um
	pdaf_focal_spot_y_um = 0.25 * cd.device_size_lateral_um

	pdaf_lambda_min_um = cd.lambda_min_um
	pdaf_lambda_max_um = cd.lambda_min_um + bandwidth_um / 3.

	pdaf_lambda_values_um = np.linspace(pdaf_lambda_min_um, pdaf_lambda_max_um, num_design_frequency_points)
	pdaf_max_intensity_by_wavelength = (cd.device_size_lateral_um**2)**2 / (focal_length_um**2 * pdaf_lambda_values_um**2)

	# Forward (Input) Source
	lateral_aperture_um = 1.1 * cd.device_size_lateral_um
	src_maximum_vertical_um = device_size_vertical_um + vertical_gap_size_um * 2. / 3.
	src_minimum_vertical_um = -focal_length_um - 0.5 * vertical_gap_size_um

	mid_lambda_um = (cd.lambda_min_um + cd.lambda_max_um)/2

	gaussian_waist_radius_um = 0.5 * ( 0.5 * 2.04 )		# half a quadrant width
	# take f number = 2.2, check index calc
	# # gaussian_waist_radius_um = airy_correction_factor * mid_lambda_um / ( np.pi * ( 1. / ( 2 * f_number ) ) ) 
	# 							 = 0.84 * 0.55 * 2.2 / (np.pi/2)
	if cd.use_airy_approximation:
		gaussian_waist_radius_um = cd.airy_correction_factor * mid_lambda_um * cd.f_number
	else:
		gaussian_waist_radius_um = mid_lambda_um / ( np.pi * ( 1. / ( 2 * cd.f_number ) ) )
	# TODO: Check which one we decided on in the end - Airy disk of Gaussian on input aperture, or f-number determining divergence half-angle?
	# TODO: Add frequency dependent profile?
	gaussian_waist_radius_um *= cd.beam_size_multiplier		#! new

	source_angle_theta_rad = np.arcsin( np.sin( cd.source_angle_theta_vacuum_deg * np.pi / 180. ) * 1.0 / cd.background_index )
	source_angle_theta_deg = source_angle_theta_rad * 180. / np.pi

	# Adjoint Sources
	adjoint_vertical_um = -focal_length_um
	num_adjoint_sources = cd.num_focal_spots
	adjoint_x_positions_um = [cd.device_size_lateral_um / 4., -cd.device_size_lateral_um / 4., -cd.device_size_lateral_um / 4., cd.device_size_lateral_um / 4.]
	adjoint_y_positions_um = [cd.device_size_lateral_um / 4., cd.device_size_lateral_um / 4., -cd.device_size_lateral_um / 4., -cd.device_size_lateral_um / 4.]

	# Optimization
	if cd.enforce_xy_gradient_symmetry:
		assert ( int( np.round( cd.device_size_lateral_um / cd.geometry_spacing_lateral_um ) ) % 2 ) == 1, "We should have an odd number of design voxels across for this operation."
	assert not cd.enforce_xy_gradient_symmetry, "20230220 Greg - This feature seems to be making things worse - I think there is a small inherent symmetry in the simulation/import/monitors"

	epoch_range_design_change_max = cd.epoch_start_design_change_max - cd.epoch_end_design_change_max
	epoch_range_design_change_min = cd.epoch_start_design_change_min - cd.epoch_end_design_change_min

	adam_betas = tuple(cd.adam_betas)

	#! new ------------------------------------------------------
	if not cd.reinterpolate_permittivity:	# new
		assert cd.reinterpolate_permittivity_factor == 1, 'Expected this factor to be 1 if not reinterpolating permittivity'

	if cd.layer_gradient:
		# would also like to try thin, thick, thin and then thick, thin, thick
		#is mode matching easier with the thin layers? and then scatter with the thick?
		assert cd.num_vertical_layers == 10, 'Expecting a specific number of layers == 10'

		voxels_per_layer = np.array( [ 1, 2, 4, 4, 4, 4, 5, 5, 5, 6 ] )
		assert np.sum( voxels_per_layer ) == device_voxels_vertical, "Expected the total layer sizes to add up to the total size!"

		# alpha = 3
		# delta = 1

		# device_mesh_voxels_vertical = int(np.round( device_size_vertical_um / mesh_spacing_um ) )


		# voxels_per_layer = []

		# for idx in range( 0, num_vertical_layers - 1 ):
		# 	voxels_per_layer.append( alpha + 2 * idx * delta )

		# total_voxels = np.sum( voxels_per_layer )

		# voxels_per_layer.append( device_mesh_voxels_vertical - total_voxels )


		# voxels_per_layer = np.array( voxels_per_layer )

		# import matplotlib.pyplot as plt
		# plt.plot( voxels_per_layer * 17 )
		# # plt.axhline( y=total_voxels )
		# plt.show()
		# asdf

		if cd.flip_gradient:
			voxels_per_layer = np.flip( voxels_per_layer )
	#! --------------------------------------------------------------------

	# Spectral and polarization selectivity information
	if cd.weight_by_individual_wavelength:
		weight_focal_plane_map_performance_weighting = [ 1.0, 1.0, 1.0, 1.0 ]
		weight_focal_plane_map = [ 1.0, 1.0, 1.0, 1.0 ]
		weight_individual_wavelengths = np.ones( len( lambda_values_um ) )
		weight_individual_wavelengths_by_quad = np.ones((4,num_design_frequency_points))
	
	# Determine the wavelengths that will be directed to each focal area
	spectral_focal_plane_map = [
		[0, cd.num_points_per_band],
		[cd.num_points_per_band, 2 * cd.num_points_per_band],
		[2 * cd.num_points_per_band, 3 * cd.num_points_per_band],
		[cd.num_points_per_band, 2 * cd.num_points_per_band]
	]

	# Find indices of desired peak locations in lambda_values_um
	desired_peak_location_per_band = [ np.argmin( ( lambda_values_um - cd.desired_peaks_per_band_um[ idx ] )**2 ) 
										for idx in range( 0, len( cd.desired_peaks_per_band_um ) ) 
									]
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


	if cd.explicit_band_centering:
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
		if cd.shuffle_green:
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


	if cd.add_infrared:
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


	if cd.do_rejection:

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

	# Filter out all variables that are not processed as a result 
	output_keys = set(dir()) - not_output_keys
	output_vars = {}
	for key in output_keys:
		if key not in ["not_output_keys"]:
			output_vars[key] = eval(key)
	
	return output_vars, vars(cd)

# Store all processed config variables in a dictionary, update raw config variables.
processed_vars, config_vars = post_process_config_vars(**config_vars)
cv = SimpleNamespace(**config_vars)                                 # The rename just makes it easier to code
pv = SimpleNamespace(**processed_vars)

logging.info('Config imported and processed.')