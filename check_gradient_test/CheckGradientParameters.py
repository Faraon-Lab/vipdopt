#
# Parameter file for the Lumerical Processing Sweep
#

# This params file should be the exact one that was used for the optimization. Its variables can be replaced or redeclared without issue, 
# and most of them will be replaced by values drawn directly from the .fsp file. But calling this will cover any bases that we have
# potentially missed.
importing_params_from_opt_file = True
if importing_params_from_opt_file:
    from SonyBayerFilterParameters import *
    # from LayeredMWIRBridgesBayerFilterParameters import *

from os import umask
import os
import numpy as np
import json 

#
#* Debug Options
#

running_on_local_machine = False    # False if running on slurm
slurm_job_env_variable = None
if slurm_job_env_variable is None:
    slurm_job_env_variable = os.getenv('SLURM_JOB_NODELIST')
if slurm_job_env_variable is None:
    running_on_local_machine = True
print('Running on: ' + 'Local Machine' if running_on_local_machine else 'HPC Cluster')
    
if running_on_local_machine:	
    lumapi_filepath =  r"C:\Program Files\Lumerical\v212\api\python\lumapi.py"
else:	
    #! do not include spaces in filepaths passed to linux
    lumapi_filepath = r"/central/home/ifoo/lumerical/2021a_r22/api/python/lumapi.py"
    
start_from_step = 3    # 0 if running entire file in one shot
    
shelf_fn = 'save_state'


#
#* Files
#

python_src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))

if running_on_local_machine:
    # projects_directory_location_init = r"C:\Users\Ian\Dropbox\Caltech\Faraon Group\Simulations\Sony Bayer\[v0] basicopt_setup\Data Processing\v3_nodeviceandcrosssection"
    projects_directory_location_init = python_src_directory
    projects_directory_location = projects_directory_location_init          # Change if necessary to differentiate  
else:
    #! do not include spaces in filepaths passed to linux
    projects_directory_location_init = "/central/groups/Faraon_Computing/ian/data_processing/lum_proc"
    projects_directory_location = "/central/groups/Faraon_Computing/ian/data_processing/lum_proc"

project_name_init = 'mode_overlap_fom_check_gradient'
project_name = 'ares_' + project_name_init
device_filename = 'optimization' # omit '.fsp'

# Initial Project File Directory
projects_directory_location_init += "/" + project_name_init
# Final Output Project File Directory
projects_directory_location += "/" + project_name
# Final Output Project Plot Directory
plot_directory_location = projects_directory_location + "/plots"

sweep_settings = json.load(open('sweep_settings.json'))

#
#* Sweep Parameters
#

def generate_value_array(start,end,step):
    return json.dumps(np.arange(start,end+step,step).tolist())

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

sweep_parameters = {}
sweep_parameters_shape = []
const_parameters = {}

for key, val in sweep_settings['params'].items():
    # Initialize all sweepable parameters with their original value
    globals()[val['var_name']] = val['var_values'][val['peakInd']]
    
    # Identify parameters that are actually being swept
    if val['iterating']:
        sweep_parameters[key] = val
        sweep_parameters_shape.append(len(val['var_values']))
    else:
        const_parameters[key] = val

# Create N+1-dim array where the last dimension is of length N.
sweep_parameter_value_array = np.zeros(sweep_parameters_shape + [len(sweep_parameters_shape)])

# At each index it holds the corresponding parameter values that are swept for each parameter's index
for idx, x in np.ndenumerate(np.zeros(sweep_parameters_shape)):
    for t_idx, p_idx in enumerate(idx):
        sweep_parameter_value_array[idx][t_idx] = (list(sweep_parameters.values())[t_idx])['var_values'][p_idx]
    #print(sweep_parameter_value_array[idx])

#
#* Plot Types
#

plots = sweep_settings['plots']

cutoff_1d_sweep_offset = [0, 0]
#
#* Lumerical Monitor Setup
#

monitors = sweep_settings['monitors']
disable_object_list = [#'design_efield_monitor',
                       #'src_spill_monitor'
                        ]
#! DO NOT disable the transmission_monitors! There is no reliable way to partition the scatter_plane_monitor in order to retrieve power values. E-field yes, but power no.
# The memory / runtime savings is not that much either.

#
#* Spectral
#

change_wavelength_range = True
if change_wavelength_range:
    d_lambda = (lambda_max_um - lambda_min_um)/5        # Just widening out the spectrum a little
    lambda_min_um -= d_lambda
    lambda_max_um += d_lambda

if not importing_params_from_opt_file:
    num_bands = 3
    num_points_per_band = 10
    num_design_frequency_points = num_bands * num_points_per_band

lambda_values_um = np.linspace(lambda_min_um, lambda_max_um, num_design_frequency_points)

bandwidth_um = lambda_max_um - lambda_min_um
num_dispersive_ranges = 1
dispersive_range_size_um = bandwidth_um / num_dispersive_ranges
dispersive_ranges_um = [
	[ lambda_min_um, lambda_max_um ]
]

#
#* Mesh
#

if not importing_params_from_opt_file:
    mesh_spacing_um = 0.017
    geometry_spacing_um = 0.051
mesh_to_geometry_factor = int( ( geometry_spacing_um + 0.5 * mesh_spacing_um ) / mesh_spacing_um )


#* Device
#

if not importing_params_from_opt_file:
    num_vertical_layers = 40

    vertical_layer_height_um = 0.051
    vertical_layer_height_voxels = int( vertical_layer_height_um / geometry_spacing_um )

device_size_lateral_um = geometry_spacing_um * 40
device_size_vertical_um = vertical_layer_height_um * num_vertical_layers

device_voxels_lateral = int(device_size_lateral_um / geometry_spacing_um)
device_voxels_vertical = int(device_size_vertical_um / geometry_spacing_um)

if 'device_size_vertical_um' not in globals():
    device_size_vertical_um = device_size_verical_um        # little typo that has to be declared for compatibility


device_voxels_simulation_mesh_lateral = 1 + int(device_size_lateral_um / mesh_spacing_um)
###### device_voxels_simulation_mesh_lateral = 2 + int(device_size_lateral_um / mesh_spacing_um)
device_voxels_simulation_mesh_vertical = 2 + int(device_size_vertical_um / mesh_spacing_um)

device_vertical_maximum_um = device_size_vertical_um
device_vertical_minimum_um = 0

# Structure the following list such that the last entry is the last medium above the device.
objects_above_device = [] #['permittivity_layer_substrate',
                        #'silicon_substrate']

#
#* Optical
#

if not importing_params_from_opt_file:
    background_index = 1.5
    min_device_index = 1.5
    max_device_index = 2.4

    min_device_permittivity = min_device_index**2
    max_device_permittivity = max_device_index**2

    init_permittivity_0_1_scale = 0.5


    focal_length_um = geometry_spacing_um * 30
    focal_plane_center_lateral_um = 0
    focal_plane_center_vertical_um = -focal_length_um

src_beam_rad = device_size_lateral_um/2
src_div_angle = 13.13            # degrees

#

#* Sidewall and Side Monitors
# num_sidewalls = 4
if not dir().count('num_sidewalls'):        # If num_sidewalls is not defined
    num_sidewalls = 0                       # Then define it
    
if num_sidewalls == 0:
    sidewall_thickness_um = 0.0
    sidewall_material = 'etch'
else:
    if not importing_params_from_opt_file:
        sidewall_thickness_um = 0.45
        sidewall_material = 'Cu (Copper) - Palik'
sidewall_extend_focalplane = False

sidewall_x_positions_um = [device_size_lateral_um / 2 + sidewall_thickness_um / 2, 0, -device_size_lateral_um / 2 - sidewall_thickness_um / 2, 0]
sidewall_y_positions_um = [0, device_size_lateral_um / 2 + sidewall_thickness_um / 2, 0, -device_size_lateral_um / 2 - sidewall_thickness_um / 2]
sidewall_xspan_positions_um = [sidewall_thickness_um, device_size_lateral_um + sidewall_thickness_um * 2,
                               sidewall_thickness_um, device_size_lateral_um + sidewall_thickness_um * 2]
sidewall_yspan_positions_um = [device_size_lateral_um + sidewall_thickness_um * 2, sidewall_thickness_um, 
                               device_size_lateral_um + sidewall_thickness_um * 2, sidewall_thickness_um]


#
#* FDTD
#

fdtd_region_size_lateral_um = 2 * lateral_gap_size_um + 1.0 * device_size_lateral_um

if not importing_params_from_opt_file:
    vertical_gap_size_um = 15 # geometry_spacing_um * 15
    lateral_gap_size_um = 4 # geometry_spacing_um * 10

    fdtd_region_size_vertical_um = 2 * vertical_gap_size_um + device_size_vertical_um + focal_length_um
    fdtd_region_size_lateral_um = 2 * lateral_gap_size_um + 2.0 * device_size_lateral_um
    fdtd_region_maximum_vertical_um = device_size_vertical_um + vertical_gap_size_um
    fdtd_region_minimum_vertical_um = -focal_length_um - vertical_gap_size_um

    fdtd_simulation_time_fs = 3000



#
#* Adjoint sources
#

if not importing_params_from_opt_file:
    num_focal_spots = 4

adjoint_vertical_um = -focal_length_um
num_adjoint_sources = num_focal_spots
adjoint_x_positions_um = [device_size_lateral_um / 4., -device_size_lateral_um / 4., -device_size_lateral_um / 4., device_size_lateral_um / 4.]
adjoint_y_positions_um = [device_size_lateral_um / 4., device_size_lateral_um / 4., -device_size_lateral_um / 4., -device_size_lateral_um / 4.]


#
#* Optimization
#

num_epochs = 1
num_iterations_per_epoch = 1