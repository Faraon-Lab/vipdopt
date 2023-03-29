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
import copy
import numpy as np
import json
from types import SimpleNamespace

# Add filepath to default path
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())

#! Gets all parameters from config file - explicitly calls it so that the variables pass into this module's globals()
from configs.global_params import *
# These parameters can and will be selectively overwritten e.g. environment variables like folder locations
# globals().update(config_vars)
# globals().update(processed_vars)
cv = SimpleNamespace(**config_vars)                                 # The rename just makes it easier to code
pv = SimpleNamespace(**processed_vars)

#* Logging
# Logger is already initialized through the call to utility.py
logging.info(f'Starting up Lumerical processing params file: {os.path.basename(__file__)}')
logging.debug(f'Current working directory: {os.getcwd()}')

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
	lumapi_filepath =  cv.lumapi_filepath_local
else:	
	#! do not include spaces in filepaths passed to linux
	lumapi_filepath = cv.lumapi_filepath_hpc


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
current_parameter_values = {}

for val in sweep_settings['params']:
    # Keep a copy of all parameters and their current values
    current_parameter_values[val['var_name']] = val['var_values'][val['peakInd']]
    
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

def create_parameter_filename_string(idx):
    ''' Input: tuple coordinate in the N-D array. Cross-matches against the sweep_parameters dictionary.
    Output: unique identifier string for naming the corresponding job'''
    
    output_string = ''
    
    for t_idx, p_idx in enumerate(idx):
        try:
            variable = list(sweep_parameters.values())[t_idx]
            variable_name = variable['short_form']
            if isinstance(p_idx, slice):
                output_string += variable_name + '_swp_'
            else:
                variable_value = variable['var_values'][p_idx]
                variable_format = variable['formatStr']
                output_string += variable_name + '_' + variable_format%(variable_value) + '_'
        except Exception as err:
            pass
        
    return output_string[:-1]

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

monitors_keyed = {}
for monitor in monitors:
    monitors_keyed[monitor['name']] = monitor
# Can also use this: https://stackoverflow.com/a/8653568


#* Overwrite some variables from optimization config		                                           
# SonyBayerFilterParameters.py is separated into two parts. 
# One processes raw parameters from config, stored in a dictionary and in its globals().
# The other is a function that takes the config dictionary as input, does processing of those parameters, 
# and returns the processed params as another dictionary.
# Here, we perform some editing of the config parameters, and then re-run that function.

# Spectral
change_wavelength_range = False
if change_wavelength_range:
    d_lambda = (cv.lambda_max_um - cv.lambda_min_um)/5        # Just widening out the spectrum a little
    cv.lambda_min_um -= d_lambda
    cv.lambda_max_um += d_lambda

if not accept_params_from_opt_file:
    cv.num_bands = 3
    cv.num_points_per_band = 10

# Mesh

if not accept_params_from_opt_file:
    cv.mesh_spacing_um = 0.017
    cv.geometry_spacing_lateral_um = 0.051
    cv.device_scale_um = 0.051 
    
# Device

if not accept_params_from_opt_file:
    cv.num_vertical_layers = 40
    cv.vertical_layer_height_um = 0.051

# Optical
if not accept_params_from_opt_file:
    cv.background_index = 1.5
    cv.min_device_index = 1.5
    cv.max_device_index = 2.4

# Sidewall and Side Monitors

cv.num_sidewalls = 0                   #! Remember to turn this on if the optimization actually was done with sidewalls

if cv.num_sidewalls == 0:
    cv.sidewall_thickness_um = 0.0
    cv.sidewall_material = 'etch'
else:
    if not accept_params_from_opt_file:
        cv.sidewall_thickness_um = 0.45
        cv.sidewall_material = 'Cu (Copper) - Palik'

# Adjoint sources

if not accept_params_from_opt_file:
    cv.num_focal_spots = 4

#*------------------------------------------------------------------------------------------------------------------------------------



#* Run post-processing and update raw config:
processed_vars, config_vars = post_process_config_vars(**vars(cv))
#* Load all into globals().
# TODO: EXCEPT FOR START_FROM_STEP AND WHATEVER AT THE TOP. We need to preserve them!!!
globals().update(config_vars)
globals().update(processed_vars)


#* Perform evaluation script-specific modifications that differ from SonyBayerFilterParameter.post_process_config_vars()

# Device

# Structure the following list such that the last entry is the last medium above the device.
objects_above_device = [] #['permittivity_layer_substrate',
                        #'silicon_substrate']

# Devices in Array

device_array_shape = (1) #(3,3)    # A tuple describing dimension sizes such as (2,3)

# FDTD

fdtd_region_size_lateral_um = 2 * lateral_gap_size_um + 3.0 * device_size_lateral_um

# Optimization

# These will always stay at 1 and 1, because the forward simulation only needs to run once to get results.
num_epochs = 1
num_iterations_per_epoch = 1

#* Debug Options
start_from_step = 0                 # 0 if running entire file in one shot  # NOTE: Overwrites optimization config.

logging.info("Lumerical plotting parameters read and processed.")