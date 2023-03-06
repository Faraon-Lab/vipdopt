#
# Parameter file for the Lumerical Processing Sweep
#
# This script processes two types of config files.
# 1) The exact config.yaml and cfg_to_params.py that was used for the optimization. Its variables can be replaced or redeclared without 
# issue, and most of them will be replaced by values drawn directly from the .fsp file. But calling this will cover any bases that we have
# potentially missed.
# 2) The sweep_settings.json determines the types of monitors, sweep parameters, and plots that will be generated from the optimization.
#

import os
import sys
import numpy as np
import json

# Add filepath to default path
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())

#! Gets all parameters from config file - explicitly calls it so that the variables pass into this module's globals()
from configs.global_params import *
# These parameters can and will be selectively overwritten e.g. environment variables like folder locations

#* Logging
# Logger is already initialized through the call to utility.py
logging.info(f'Starting up Lumerical processing params file: {os.path.basename(__file__)}')
logging.debug(f'Current working directory: {os.getcwd()}')

#* Debug Options
start_from_step = 0                 # 0 if running entire file in one shot  # NOTE: Overwrites optimization config.

#* Establish folder structure & location
python_src_directory = os.path.dirname(os.path.dirname(__file__))   # Go up one folder

running_on_local_machine = False
slurm_job_env_variable = os.getenv('SLURM_JOB_NODELIST')
if slurm_job_env_variable is None:
	running_on_local_machine = True

accept_params_from_opt_file = True

#* Environment Variables
# Repeat the call to lumapi filepaths in case something has changed
if running_on_local_machine:	
	lumapi_filepath =  lumapi_filepath_local
else:	
	#! do not include spaces in filepaths passed to linux
	lumapi_filepath = lumapi_filepath_hpc


#* Input Arguments
parser = argparse.ArgumentParser(description=\
					"Takes the config files from the previous optimization, as well as the sweep settings config, and processes them for \
                    Lumerical processing.")
for argument in utility.argparse_universal_args:
	parser.add_argument(argument[0], argument[1], **(argument[2]))
parser.add_argument("-s", "--sweepconfig", help="Sweep Settings (JSON) path", default='configs/sweep_settings.json')
args = parser.parse_args()

#* Load sweep JSON
sweep_filename = args.sweepconfig
sweep_settings = json.load(open(sweep_filename))
logging.debug(f'Read parameters from file: {sweep_filename}')

device_filename = 'optimization' # omit '.fsp'

#* Create project folder to store all files
projects_directory_location = 'ares_' + os.path.basename(python_src_directory)
if args.override is not None:
    projects_directory_location = args.override
if not os.path.isdir(projects_directory_location):
	os.mkdir(projects_directory_location)

#
#* Sweep Parameters
#

# The params key in sweep_settings contains both parameters that are being swept and parameters that are not.
# We sort them accordingly and set up a multidimensional array that contains all sweep parameter values.
sweep_parameters = {}
sweep_parameters_shape = []
const_parameters = {}

for val in sweep_settings['params']:
    # Identify parameters that are actually being swept
    if val['iterating']:
        sweep_parameters[val['short_form']] = val
        sweep_parameters_shape.append(len(val['var_values']))
    else:
        const_parameters[val['short_form']] = val

# Create N+1-dim array where the last dimension is of length N.
sweep_parameter_value_array = np.zeros(sweep_parameters_shape + [len(sweep_parameters_shape)])

# At each index it holds the corresponding parameter values that are swept for each parameter's index
for idx, x in np.ndenumerate(np.zeros(sweep_parameters_shape)):
    for t_idx, p_idx in enumerate(idx):
        sweep_parameter_value_array[idx][t_idx] = (list(sweep_parameters.values())[t_idx])['var_values'][p_idx]


#* Plot Types
# and global plot settings

plots = sweep_settings['plots']
cutoff_1d_sweep_offset = [0, 0]

#* Lumerical Monitor Setup

monitors = sweep_settings['monitors']
disable_object_list = ['design_efield_monitor',
                       'src_spill_monitor']		# These objects will be disabled if they exist, in order to save on memory.
#! DO NOT disable the transmission_monitors! There is no reliable way to partition the scatter_plane_monitor in order to retrieve power values. E-field yes, but power no.
# The memory / runtime savings is not that much either.

#* Spectral
change_wavelength_range = False
if change_wavelength_range:
    d_lambda = (lambda_max_um - lambda_min_um)/5        # Just widening out the spectrum a little
    lambda_min_um -= d_lambda
    lambda_max_um += d_lambda

if not accept_params_from_opt_file:
    num_bands = 3
    num_points_per_band = 10
    num_design_frequency_points = num_bands * num_points_per_band
    num_eval_frequency_points = 6 * num_design_frequency_points

# Overwrite some variables from optimization config		
# TODO: What should eventually be done is to separate SonyBayerFilterParameters into two parts. 
# todo: One processes raw parameters from config, stored in a dictionary and in its globals().
# todo: The other is a function that takes the config dictionary as input, does processing of those parameters, 
# todo: and return its locals() as another dictionary.
# todo: Then, this script can just call that function for all the processing that's done below.
bandwidth_um = lambda_max_um - lambda_min_um
# lambda_values_um = np.linspace(lambda_min_um, lambda_max_um, num_design_frequency_points)
lambda_values_um = np.linspace(lambda_min_um, lambda_max_um, num_eval_frequency_points)
dispersive_range_size_um = bandwidth_um / num_dispersive_ranges
dispersive_ranges_um = [
	[ lambda_min_um, lambda_max_um ]
]


#* Mesh

if not accept_params_from_opt_file:
    mesh_spacing_um = 0.017
    geometry_spacing_lateral_um = 0.051
    device_scale_um = 0.051 
# mesh_to_geometry_factor = int( ( geometry_spacing_lateral_um + 0.5 * mesh_spacing_um ) / mesh_spacing_um )	# TODO: Do we need this for back-compatibility?

#* Device

if not accept_params_from_opt_file:
    num_vertical_layers = 40
    vertical_layer_height_um = 0.051
    vertical_layer_height_voxels = int( vertical_layer_height_um / geometry_spacing_lateral_um )

# device_voxels_lateral = int(device_size_lateral_um / geometry_spacing_um)
# device_voxels_vertical = int(device_size_vertical_um / geometry_spacing_um)
# device_size_lateral_um = geometry_spacing_um * 40
device_size_vertical_um = vertical_layer_height_um * num_vertical_layers
if 'device_size_vertical_um' not in globals():
    device_size_vertical_um = device_size_verical_um        # little typo that has to be declared for compatibility

device_voxels_simulation_mesh_lateral = 1 + int(device_size_lateral_um / mesh_spacing_um)
# device_voxels_simulation_mesh_lateral = 2 + int(device_size_lateral_um / mesh_spacing_um)
# device_voxels_simulation_mesh_vertical = 2 + int(device_size_vertical_um / mesh_spacing_um)
device_voxels_simulation_mesh_vertical = lookup_additional_vertical_mesh_cells[ num_vertical_layers ] + int(device_size_vertical_um / mesh_spacing_um)

device_vertical_maximum_um = device_size_vertical_um	# NOTE: Unused

# Structure the following list such that the last entry is the last medium above the device.
objects_above_device = [] #['permittivity_layer_substrate',
                        #'silicon_substrate']

#* Devices in Array

device_array_shape = (1)

#* Optical
if not accept_params_from_opt_file:
    background_index = 1.5
    min_device_index = 1.5
    max_device_index = 2.4

min_device_permittivity = min_device_index**2
max_device_permittivity = max_device_index**2

#! 20230220 - Empirically, keeping focal length at 3/4 of device length seems the most effective.
focal_length_um = device_scale_um * 30#40#10#20#40#50#30	
focal_plane_center_vertical_um = -focal_length_um

#* Sidewall and Side Monitors

num_sidewalls = 0                   #! Remember to turn this off if the optimization actually was done with sidewalls

if num_sidewalls == 0:
    sidewall_thickness_um = 0.0
    sidewall_material = 'etch'
else:
    if not accept_params_from_opt_file:
        sidewall_thickness_um = 0.45
        sidewall_material = 'Cu (Copper) - Palik'

#* FDTD

fdtd_region_size_lateral_um = 2 * lateral_gap_size_um + 3.0 * device_size_lateral_um

if not accept_params_from_opt_file:
    vertical_gap_size_um = geometry_spacing_lateral_um * 15
    lateral_gap_size_um = device_scale_um * 10

    fdtd_region_size_vertical_um = 2 * vertical_gap_size_um + device_size_vertical_um + focal_length_um
    fdtd_region_size_lateral_um = 2 * lateral_gap_size_um + 2.0 * device_size_lateral_um
    fdtd_region_maximum_vertical_um = device_size_vertical_um + vertical_gap_size_um
    fdtd_region_minimum_vertical_um = -focal_length_um - vertical_gap_size_um

#* Adjoint sources

if not accept_params_from_opt_file:
    num_focal_spots = 4

adjoint_vertical_um = -focal_length_um
num_adjoint_sources = num_focal_spots
adjoint_x_positions_um = [device_size_lateral_um / 4., -device_size_lateral_um / 4., -device_size_lateral_um / 4., device_size_lateral_um / 4.]
adjoint_y_positions_um = [device_size_lateral_um / 4., device_size_lateral_um / 4., -device_size_lateral_um / 4., -device_size_lateral_um / 4.]


#* Optimization
# These will always stay at 1 and 1, because the forward simulation only needs to run once to get results.
num_epochs = 1
num_iterations_per_epoch = 1


logging.info("Lumerical plotting parameters read and processed.")