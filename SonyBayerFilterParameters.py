#
# Parameter file for the Sony Bayer Filter optimization
#

import numpy as np
from os import umask
import os
import numpy as np
import json

#
#* Debug Options
#

running_on_local_machine = False
slurm_job_env_variable = os.getenv('SLURM_JOB_NODELIST')
if slurm_job_env_variable is None:
    running_on_local_machine = True
    
if running_on_local_machine:	
    lumapi_filepath =  r"C:\Program Files\Lumerical\v212\api\python\lumapi.py"
else:	
    #! do not include spaces in filepaths passed to linux
    lumapi_filepath = r"/central/home/ifoo/lumerical/2021a_r22/api/python/lumapi.py"
    
start_from_step = 0 	# 0 if running entire file in one shot
    
shelf_fn = 'save_state'

#
#* Files
#

python_src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))

if running_on_local_machine:
    # projects_directory_location_init = r"C:\Users\Ian\Dropbox\Caltech\Faraon Group\Simulations\Exploratory Quick Sims\NRL Grant 20220720 Schematic\dev"
    projects_directory_location_init = python_src_directory
    projects_directory_location = projects_directory_location_init          # Change if necessary to differentiate  
else:
    #! do not include spaces in filepaths passed to linux
    projects_directory_location_init = "/central/groups/Faraon_Computing/ian/sony"
    projects_directory_location = "/central/groups/Faraon_Computing/ian/sony"
    
project_name_init = 'fom_optimization_dev'
project_name = 'ares_' + project_name_init

# Initial Project File Directory
# projects_directory_location_init += "/" + project_name_init
# Final Output Project File Directory
projects_directory_location += "/" + project_name


#
#* Mesh
#

mesh_spacing_um = 0.017
geometry_spacing_um = 0.051
mesh_to_geometry_factor = int((geometry_spacing_um/mesh_spacing_um + 0.5 * mesh_spacing_um)/mesh_spacing_um)        # round up 


#
#* Optical
#

background_index = 1.5      # TiO2
min_device_index = 1.5
max_device_index = 2.4      # SiO2

min_device_permittivity = min_device_index**2
max_device_permittivity = max_device_index**2

init_permittivity_0_1_scale = 0.5


focal_length_um = geometry_spacing_um * 30
focal_plane_center_lateral_um = 0
focal_plane_center_vertical_um = -focal_length_um

#
#* Device
#

num_vertical_layers = 40
vertical_layer_height_um = 0.051
vertical_layer_height_voxels = int( vertical_layer_height_um / geometry_spacing_um )

# TODO: Switch back to 41
device_size_lateral_um = geometry_spacing_um * 40 #41
device_size_vertical_um = vertical_layer_height_um * num_vertical_layers

device_voxels_lateral = int(device_size_lateral_um / geometry_spacing_um)
device_voxels_vertical = int(device_size_vertical_um / geometry_spacing_um)

# device_voxels_simulation_mesh_lateral = 1 + int(device_size_lateral_um / mesh_spacing_um)
device_voxels_simulation_mesh_lateral = 2 + int(device_size_lateral_um / mesh_spacing_um)
device_voxels_simulation_mesh_vertical = 2 + int(device_size_vertical_um / mesh_spacing_um)

device_vertical_maximum_um = device_size_vertical_um
device_vertical_minimum_um = 0

# Structure the following list such that the last entry is the last medium above the device.
objects_above_device = [] #['permittivity_layer_substrate',
                        #'silicon_substrate']


#
#* Spectral
#
lambda_min_um = 0.400
lambda_max_um = 0.700
bandwidth_um = lambda_max_um - lambda_min_um

num_bands = 3
num_points_per_band = 30
num_design_frequency_points = num_bands * num_points_per_band
lambda_values_um = np.linspace(lambda_min_um, lambda_max_um, num_design_frequency_points)

# If we are using an intensity FoM, different spectral bands will have different maximum intensities, and so they must be renormalized otherwise
# one band will be prioritized / differently weighted over others.
# See: https://en.wikipedia.org/wiki/Airy_disk#:~:text=.-,The%20intensity,-%7B%5Cdisplaystyle%20I_%7B0 for details
max_intensity_by_wavelength = (device_size_lateral_um**2)**2 / (focal_length_um**2 * lambda_values_um**2)


lambda_min_eval_um = 0.400
lambda_max_eval_um = 0.700
num_eval_frequency_points = 6 * num_design_frequency_points

num_dispersive_ranges = 1
dispersive_range_size_um = bandwidth_um / num_dispersive_ranges

dispersive_ranges_um = [
    [ lambda_min_um, lambda_max_um ]
]


#
#* FDTD
#
vertical_gap_size_um = geometry_spacing_um * 6 #15
lateral_gap_size_um = geometry_spacing_um * 4 #10

fdtd_region_size_vertical_um = 2 * vertical_gap_size_um + device_size_vertical_um + focal_length_um
fdtd_region_size_lateral_um = 2 * lateral_gap_size_um + device_size_lateral_um
fdtd_region_maximum_vertical_um = device_size_vertical_um + vertical_gap_size_um
fdtd_region_minimum_vertical_um = -focal_length_um - vertical_gap_size_um

fdtd_simulation_time_fs = 3000


#
#* Forward Source
#
lateral_aperture_um = 1.1 * device_size_lateral_um
src_maximum_vertical_um = device_size_vertical_um + vertical_gap_size_um * 2. / 3.
src_minimum_vertical_um = -focal_length_um - 0.5 * vertical_gap_size_um

src_beam_rad = 0.5 * device_size_lateral_um
src_angle_incidence = 0 # degrees
src_phi_incidence = 0 # degrees

src_hgt_Si = 0
src_hgt_polymer = 0
n_Si = 3.43
n_polymer = max_device_index

weight_by_individual_wavelength = True


#
#* Adjoint sources
#
adjoint_vertical_um = -focal_length_um
num_focal_spots = 4
num_adjoint_sources = num_focal_spots
adjoint_x_positions_um = [device_size_lateral_um / 4., -device_size_lateral_um / 4., -device_size_lateral_um / 4., device_size_lateral_um / 4.]
adjoint_y_positions_um = [device_size_lateral_um / 4., device_size_lateral_um / 4., -device_size_lateral_um / 4., -device_size_lateral_um / 4.]

adj_src_beam_rad = 0.5 * (0.5*device_size_lateral_um)

# An array of dispersive ranges, that maps each dispersive range to one or more adjoint sources. 
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
#* Optimization
#

enforce_xy_gradient_symmetry = False
if enforce_xy_gradient_symmetry:
    assert ( int( device_size_lateral_um / geometry_spacing_um ) % 2 ) == 1, "We should have an odd number of deisgn voxels across for this operation"


num_epochs = 1 #10
num_iterations_per_epoch = 1 #30

use_fixed_step_size = False
fixed_step_size = 2.0

#
# Alternative is to weight by focal intensity - need to check max intensity weighting thing
#
weight_by_quadrant_transmission = True


epoch_start_design_change_max = 0.15
epoch_end_design_change_max = 0.05
epoch_range_design_change_max = epoch_start_design_change_max - epoch_end_design_change_max

epoch_start_design_change_min = 0.05
epoch_end_design_change_min = 0
epoch_range_design_change_min = epoch_start_design_change_min - epoch_end_design_change_min


#
#* Spectral and polarization selectivity information
#
polarizations_focal_plane_map = [ ['x', 'y'], ['x', 'y'], ['x', 'y'], ['x', 'y'] ]

weight_focal_plane_map_performance_weighting = [ 1.0, 1.0, 1.0, 1.0 ]
weight_focal_plane_map = [ 1.0, 1.0, 1.0, 1.0 ]
if weight_by_individual_wavelength:
    weight_focal_plane_map_performance_weighting = [ 1.0, 1.0, 1.0, 1.0 ]
    weight_focal_plane_map = [ 1.0, 1.0, 1.0, 1.0 ]

desired_peaks_per_band_um = [ 0.46, 0.52, 0.61 ]
desired_peak_location_per_band = [ 0 for idx in range( 0, len( desired_peaks_per_band_um ) ) ]

for idx in range( 0, len( desired_peaks_per_band_um ) ):
    peak_um = desired_peaks_per_band_um[ idx ]
    found = False

    # Populate desired_peak_location_per_band
    for wl_idx in range( 0, num_design_frequency_points ):
        get_wl_um = lambda_values_um[ wl_idx ]

        if get_wl_um > peak_um:
            desired_peak_location_per_band[ idx ] = wl_idx
            found = True
            break

    assert found, "Peak location doesn't make sense"

for idx in range( 0, len( desired_peak_location_per_band ) ):
    get_location = desired_peak_location_per_band[ idx ]

    half_width = num_points_per_band // 2

    assert ( get_location - half_width ) >= 0, "Not enough optimization room in short wavelength region!"
    assert ( get_location + half_width + 1 ) <= num_design_frequency_points, "Not enough optimization room in long wavelength region!"


polarization_name_to_idx = { 'x':0, 'y':1, 'z':2 }
# Array that maps a band of wavelength (indices) to each adjoint source on the focal plane. Hard-coded.
spectral_focal_plane_map = [
    [0, num_points_per_band],
    [num_points_per_band, 2 * num_points_per_band],
    [2 * num_points_per_band, 3 * num_points_per_band],
    [num_points_per_band, 2 * num_points_per_band]
]


# spectral_focal_plane_map = [
# 	[ desired_peak_location_per_band[ 0 ] - ( num_points_per_band // 2 ), desired_peak_location_per_band[ 0 ] + 1 + ( num_points_per_band // 2 ) ],
# 	[ desired_peak_location_per_band[ 1 ] - ( num_points_per_band // 2 ), desired_peak_location_per_band[ 1 ] + 1 + ( num_points_per_band // 2 ) ],
# 	[ desired_peak_location_per_band[ 2 ] - ( num_points_per_band // 2 ), desired_peak_location_per_band[ 2 ] + 1 + ( num_points_per_band // 2 ) ],
# 	[ desired_peak_location_per_band[ 1 ] - ( num_points_per_band // 2 ), desired_peak_location_per_band[ 1 ] + 1 + ( num_points_per_band // 2 ) ]
# ]



do_rejection = False
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


    softplus_kappa = 2

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

print("Parameters file loaded successfully.")