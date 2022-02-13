#
# Parameter file for the Lumerical Processing Sweep
#

import numpy as np
import json

#
#* Debug Options
#
running_on_local_machine = True	# False if running on slurm
if running_on_local_machine:	
    lumapi_filepath =  r"C:\Program Files\Lumerical\v212\api\python\lumapi.py"
else:	
    #! do not include spaces in filepaths passed to linux
    lumapi_filepath = r"/central/home/ifoo/lumerical/2021a_r22/api/python/lumapi.py"
    
start_from_step = 2		# 0 if running entire file in one shot
    
shelf_fn = 'save_state'


#
#* Files
#
if running_on_local_machine:
    projects_directory_location_init = r"C:\Users\Ian\Dropbox\Caltech\Faraon Group\Simulations\NTT Paper\[v0] Setup and Tutorials\dev\test"
    projects_directory_location = r"C:\Users\Ian\Dropbox\Caltech\Faraon Group\Simulations\NTT Paper\[v0] Setup and Tutorials\dev\test"  
else:
    #! do not include spaces in filepaths passed to linux
    projects_directory_location_init = "/central/groups/Faraon_Computing/ian/data_processing/lum_proc"
    projects_directory_location = "/central/groups/Faraon_Computing/ian/data_processing/lum_proc"

project_name_init = 'Cu_450nm_th0'
project_name = 'ares_' + project_name_init
device_filename = 'optimization' # omit '.fsp'

# Initial Project File Directory
projects_directory_location_init += "/" + project_name_init
# Final Output Project File Directory
projects_directory_location += "/" + project_name

sweep_settings = json.load(open('sweep_settings.json'))

#
#* Sweep Parameters
#

def generate_value_array(start,end,step):
    return json.dumps(np.arange(start,start+end+step,step).tolist())

def create_parameter_filename_string(idx):
    ''' Input: tuple coordinate in the N-D array. Cross-matches against the sweep_parameters dictionary.
    Output: unique identifier string for naming the corresponding job'''
    
    output_string = ''
    
    for t_idx, p_idx in enumerate(idx):
        variable = list(sweep_parameters.values())[t_idx]
        variable_name = variable['short_form']
        variable_value = variable['var_values'][p_idx]
        variable_format = variable['formatStr']
        output_string += variable_name + '_' + variable_format%(variable_value) + '_'
        
    return output_string[:-1]

sweep_parameters = {}
sweep_parameters_shape = []

for key, val in sweep_settings['params'].items():
    # Initialize all sweepable parameters with their original value
    globals()[val['var_name']] = val['var_values'][val['peakInd']]
    
    # Identify parameters that are actually being swept
    if val['iterating']:
        sweep_parameters[key] = val
        sweep_parameters_shape.append(len(val['var_values']))

# Create N+1-dim array where the last dimension is of length N.
sweep_parameter_value_array = np.zeros(sweep_parameters_shape + [len(sweep_parameters_shape)])

# At each index it holds the corresponding parameter values that are swept for each parameter's index
for idx, x in np.ndenumerate(np.zeros(sweep_parameters_shape)):
    for t_idx, p_idx in enumerate(idx):
        sweep_parameter_value_array[idx][t_idx] = (list(sweep_parameters.values())[t_idx])['var_values'][p_idx]
    #print(sweep_parameter_value_array[idx])



#
#* Mesh
#
mesh_spacing_um = 0.017
geometry_spacing_um = 0.051
mesh_to_geometry_factor = int( ( geometry_spacing_um + 0.5 * mesh_spacing_um ) / mesh_spacing_um )

# #
# #* Optical
# #
# background_index = 1.5
# min_device_index = 1.5
# max_device_index = 2.4

# #min_device_permittivity = min_device_index**2
# #max_device_permittivity = max_device_index**2

# #init_permittivity_0_1_scale = 0.5


# focal_length_um = geometry_spacing_um * 30
# focal_plane_center_lateral_um = 0
# focal_plane_center_vertical_um = -focal_length_um

# #
# #* Device
# #

# num_vertical_layers = 40

# vertical_layer_height_um = 0.051
# #vertical_layer_height_voxels = int( vertical_layer_height_um / geometry_spacing_um )

# device_size_lateral_um = geometry_spacing_um * 40
# device_size_vertical_um = vertical_layer_height_um * num_vertical_layers

# device_voxels_lateral = int(device_size_lateral_um / geometry_spacing_um)
# device_voxels_vertical = int(device_size_vertical_um / geometry_spacing_um)

# device_voxels_simulation_mesh_lateral = 1 + int(device_size_lateral_um / mesh_spacing_um)
# # device_voxels_simulation_mesh_lateral = 2 + int(device_size_lateral_um / mesh_spacing_um)
# device_voxels_simulation_mesh_vertical = 2 + int(device_size_vertical_um / mesh_spacing_um)

# device_vertical_maximum_um = device_size_vertical_um
# device_vertical_minimum_um = 0

# ------------------------ LMBBF Params -------------------------------------------
#
#* Optical
#
background_index = 1.0
min_device_index = 1.0
max_device_index = 1.5

min_device_permittivity = min_device_index**2
max_device_permittivity = max_device_index**2

init_permittivity_0_1_scale = 0.5

focal_length_um = 30
focal_plane_center_lateral_um = 0
focal_plane_center_vertical_um = -focal_length_um

#
#* Device
#
mesh_spacing_um = 0.25

device_size_lateral_um = 30#25
device_size_vertical_um = 25

device_voxels_lateral = 1 + int(device_size_lateral_um / mesh_spacing_um)
device_voxels_vertical = 2 + int(device_size_vertical_um / mesh_spacing_um)

device_vertical_maximum_um = device_size_vertical_um
device_vertical_minimum_um = 0

silicon_thickness_um = 3.5

num_sidewalls = 4
sidewall_thickness_um = 0.45
sidewall_material = 'Cu (Copper) - Palik'
sidewall_extend_focalplane = False
sidewall_x_positions_um = [device_size_lateral_um / 2 + sidewall_thickness_um / 2, 0, -device_size_lateral_um / 2 - sidewall_thickness_um / 2, 0]
sidewall_y_positions_um = [0, device_size_lateral_um / 2 + sidewall_thickness_um / 2, 0, -device_size_lateral_um / 2 - sidewall_thickness_um / 2]
sidewall_xspan_positions_um = [sidewall_thickness_um, device_size_lateral_um + sidewall_thickness_um * 2,
                               sidewall_thickness_um, device_size_lateral_um + sidewall_thickness_um * 2]
sidewall_yspan_positions_um = [device_size_lateral_um + sidewall_thickness_um * 2, sidewall_thickness_um, 
                               device_size_lateral_um + sidewall_thickness_um * 2, sidewall_thickness_um]

# Structure the following list such that the last entry is the last medium above the device.
objects_above_device = ['permittivity_layer_substrate',
                        'silicon_substrate']

#-----------------------------------------------------------------------------------------

#
#* FDTD
#
vertical_gap_size_um = 15 # geometry_spacing_um * 15
lateral_gap_size_um = 4 # geometry_spacing_um * 10

fdtd_region_size_vertical_um = 2 * vertical_gap_size_um + device_size_vertical_um + focal_length_um
fdtd_region_size_lateral_um = 2 * lateral_gap_size_um + device_size_lateral_um
fdtd_region_maximum_vertical_um = device_size_vertical_um + vertical_gap_size_um
fdtd_region_minimum_vertical_um = -focal_length_um - vertical_gap_size_um

fdtd_simulation_time_fs = 3000

#
#* Lumerical Monitor Setup
#

monitors = sweep_settings['monitors']

disable_object_list = ['design_efield_monitor']



#
#* Spectral
#
lambda_min_um = 0.400 #3.5
lambda_max_um = 0.700 #5.5

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
#* Adjoint sources
#
adjoint_vertical_um = -focal_length_um
num_focal_spots = 4
num_adjoint_sources = num_focal_spots
adjoint_x_positions_um = [device_size_lateral_um / 4., -device_size_lateral_um / 4., -device_size_lateral_um / 4., device_size_lateral_um / 4.]
adjoint_y_positions_um = [device_size_lateral_um / 4., device_size_lateral_um / 4., -device_size_lateral_um / 4., -device_size_lateral_um / 4.]


dispersive_range_to_adjoint_src_map = [
	[ 0, 1, 2, 3 ]
]



#
#* Optimization
#

num_epochs = 1
num_iterations_per_epoch = 1