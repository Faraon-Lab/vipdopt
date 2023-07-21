# Copyright © 2023, California Institute of Technology. All rights reserved.
#
# Use in source and binary forms for nonexclusive, nonsublicenseable, commercial purposes with or without modification, is permitted provided that the following conditions are met:
# - Use of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# - Use in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the software.
# - Neither the name of the California Institute of Technology (Caltech) nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


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

importing_params_from_opt_file = True
if importing_params_from_opt_file:
    from SonyBayerFilterParameters import *
    # from LayeredMWIRBridgesBayerFilterParameters import *

running_on_local_machine = False
slurm_job_env_variable = os.getenv('SLURM_JOB_NODELIST')
if slurm_job_env_variable is None:
    running_on_local_machine = True
    
if running_on_local_machine:	
    lumapi_filepath =  r"C:\Program Files\Lumerical\v212\api\python\lumapi.py"
else:	
    #! do not include spaces in filepaths passed to linux
    lumapi_filepath = r"/central/home/ifoo/lumerical/2021a_r22/api/python/lumapi.py"

convert_existing = False     # Are we converting a previous file that didn't have the mode overlap FoM?
start_from_step = 0 	    # 0 if running entire file in one shot
    
shelf_fn = 'save_state'

#
#* Files
#

# project_name_init = 'normal_sony_baseline_2p04x2p04x2p04um_f1p53um'
# project_name = 'normal_sony_baseline_2p04x2p04x2p04um_f1p53um'


#
# First attempt band rejection START
#

# project_name_init = 'normal_sony_reject_v2_2p04x2p04x2p04um_f1p53um'
# project_name = 'normal_sony_reject_v2_2p04x2p04x2p04um_f1p53um'

# project_name_init = 'normal_sony_reject_v3_2p04x2p04x2p04um_f1p53um'
# project_name = 'normal_sony_reject_v3_2p04x2p04x2p04um_f1p53um'

# project_name_init = 'normal_sony_reject_v4_2p04x2p04x2p04um_f1p53um'
# project_name = 'normal_sony_reject_v4_2p04x2p04x2p04um_f1p53um'

# project_name_init = 'normal_sony_reject_v5_2p04x2p04x2p04um_f1p53um'
# project_name = 'normal_sony_reject_v5_2p04x2p04x2p04um_f1p53um'

# project_name_init = 'normal_sony_reject_v6_2p04x2p04x2p04um_f1p53um'
# project_name = 'normal_sony_reject_v6_2p04x2p04x2p04um_f1p53um'

# project_name_init = 'normal_sony_reject_v7_2p04x2p04x2p04um_f1p53um'
# project_name = 'normal_sony_reject_v7_2p04x2p04x2p04um_f1p53um'


# project_name_init = 'normal_sony_reject_v8_2p04x2p04x2p04um_f1p53um'
# project_name = 'normal_sony_reject_v8_2p04x2p04x2p04um_f1p53um'


# project_name_init = 'normal_sony_reject_v8_2p091x2p091x2p04um_f1p53um'
# project_name = 'normal_sony_reject_v8_2p091x2p091x2p04um_f1p53um'

#
# First attempt band rejection END
#


#
# Layer Resolution START
#

# project_name_init = 'normal_sony_baseline_2p091x2p091x2p04um_40layer_f1p53um'
# project_name = 'normal_sony_baseline_2p091x2p091x2p04um_40layer_f1p53um'

# project_name_init = 'normal_sony_baseline_2p091x2p091x2p04um_20layer_f1p53um'
# project_name = 'normal_sony_baseline_2p091x2p091x2p04um_20layer_f1p53um'

# project_name_init = 'normal_sony_baseline_2p091x2p091x2p04um_10layer_f1p53um'
# project_name = 'normal_sony_baseline_2p091x2p091x2p04um_10layer_f1p53um'

# project_name_init = 'normal_sony_baseline_2p091x2p091x2p04um_5layer_f1p53um'
# project_name = 'normal_sony_baseline_2p091x2p091x2p04um_5layer_f1p53um'

# project_name_init = 'normal_sony_baseline_2p091x2p091x2p04um_30layer_f1p53um'
# project_name = 'normal_sony_baseline_2p091x2p091x2p04um_30layer_f1p53um'

# project_name_init = 'normal_sony_baseline_2p091x2p091x2p04um_15layer_f1p53um'
# project_name = 'normal_sony_baseline_2p091x2p091x2p04um_15layer_f1p53um'


# project_name_init = 'normal_sony_baseline_2p091x2p091x2p04um_10layer_shift_centers_f1p53um'
# project_name = 'normal_sony_baseline_2p091x2p091x2p04um_10layer_shift_centers_f1p53um'

#
# Layer Resolution END


#
# Device Height with Number of Layers START
#

# project_name_init = 'normal_sony_baseline_2p04x2p04x2p04um_10layer_f1p53um'
# project_name = 'normal_sony_baseline_2p04x2p04x2p04um_10layer_f1p53um'

# project_name_init = 'normal_sony_baseline_2p04x2p04xp204um_1layer_f1p53um'
# project_name = 'normal_sony_baseline_2p04x2p04xp204um_1layer_f1p53um'

# project_name_init = 'normal_sony_baseline_2p04xp408xp408um_2layer_f1p53um'
# project_name = 'normal_sony_baseline_2p04xp408xp408um_2layer_f1p53um'

# project_name_init = 'normal_sony_baseline_2p04xp408xp816um_4layer_f1p53um'
# project_name = 'normal_sony_baseline_2p04xp408xp816um_4layer_f1p53um'

# project_name_init = 'normal_sony_baseline_2p04xp408x1p224um_6layer_f1p53um'
# project_name = 'normal_sony_baseline_2p04xp408x1p224um_6layer_f1p53um'

lookup_additional_vertical_mesh_cells = {}
lookup_additional_vertical_mesh_cells[ 1 ] = 3
lookup_additional_vertical_mesh_cells[ 2 ] = 3
lookup_additional_vertical_mesh_cells[ 3 ] = 2
lookup_additional_vertical_mesh_cells[ 4 ] = 3
lookup_additional_vertical_mesh_cells[ 5 ] = 2
lookup_additional_vertical_mesh_cells[ 6 ] = 2
lookup_additional_vertical_mesh_cells[ 8 ] = 3
lookup_additional_vertical_mesh_cells[ 10 ] = 2
lookup_additional_vertical_mesh_cells[ 15 ] = 2
lookup_additional_vertical_mesh_cells[ 20 ] = 2
lookup_additional_vertical_mesh_cells[ 30 ] = 2
lookup_additional_vertical_mesh_cells[ 40 ] = 2


#
# Device Height with Number of Layers END

#
# Feature Sizes START
#

# project_name_init = 'normal_sony_baseline_2p04x2p04x2p04um_40layer_feature_p051um_f1p53um'
# project_name = 'normal_sony_baseline_2p04x2p04x2p04um_40layer_feature_p051um_f1p53um'

# project_name_init = 'normal_sony_baseline_2p04x2p04x2p04um_40layer_feature_p068um_f1p53um'
# project_name = 'normal_sony_baseline_2p04x2p04x2p04um_40layer_feature_p068um_f1p53um'

# project_name_init = 'normal_sony_baseline_2p04x2p04x2p04um_40layer_feature_p085um_f1p53um'
# project_name = 'normal_sony_baseline_2p04x2p04x2p04um_40layer_feature_p085um_f1p53um'

# project_name_init = 'normal_sony_baseline_2p04x2p04x2p04um_40layer_feature_p102um_f1p53um'
# project_name = 'normal_sony_baseline_2p04x2p04x2p04um_40layer_feature_p102um_f1p53um'

#
# Feature Sizes END
#


#
# Focal Length START
#

# project_name_init = 'normal_sony_baseline_2p04x2p04x2p04um_40layer_fp51um'
# project_name = 'normal_sony_baseline_2p04x2p04x2p04um_40layer_fp51um'

# project_name_init = 'normal_sony_baseline_2p04x2p04x2p04um_40layer_f1p02um'
# project_name = 'normal_sony_baseline_2p04x2p04x2p04um_40layer_f1p02um'

# project_name_init = 'normal_sony_baseline_2p04x2p04x2p04um_40layer_f2p04um'
# project_name = 'normal_sony_baseline_2p04x2p04x2p04um_40layer_f2p04um'

# project_name_init = 'normal_sony_baseline_2p04x2p04x2p04um_40layer_f2p55um'
# project_name = 'normal_sony_baseline_2p04x2p04x2p04um_40layer_f2p55um'


#
# Focal Length END
#


#
# RGBI Baseline START
#

# project_name_init = 'normal_sony_baseline_rgbi_2p04x2p04x2p04um_f1p53um'
# project_name = 'normal_sony_baseline_rgbi_2p04x2p04x2p04um_f1p53um'


#
# RGBI Baseline END
#



#
# 102nm features all dimensions START
#

# project_name_init = 'normal_sony_102nm_all_rgb_2p04x2p04x2p04um_f1p53um'
# project_name = 'normal_sony_102nm_all_rgb_2p04x2p04x2p04um_f1p53um'

# project_name_init = 'normal_sony_102nm_all_rgbi_2p04x2p04x2p04um_f1p53um'
# project_name = 'normal_sony_102nm_all_rgbi_2p04x2p04x2p04um_f1p53um'


#
# 102nm features all dimensions END
#


#
# PDAF seed START
#

# project_name_init = 'normal_sony_pdaf_seed_rgb_2p04x2p04x2p04um_f1p53um'
# project_name = 'normal_sony_pdaf_seed_rgb_2p04x2p04x2p04um_f1p53um'


# project_name_init = 'ten_deg_sony_pdaf_seed_rgb_2p04x2p04x2p04um_f1p53um'
# project_name = 'ten_deg_sony_pdaf_seed_rgb_2p04x2p04x2p04um_f1p53um'


#
# PDAF seed END
#


#
# Green balance START
#

# project_name_init = 'ten_deg_sony_green_balance_v1_rgb_2p04x2p04x2p04um_f1p53um'
# project_name = 'ten_deg_sony_green_balance_v1_rgb_2p04x2p04x2p04um_f1p53um'

# project_name_init = 'ten_deg_sony_green_balance_v2_rgb_2p04x2p04x2p04um_f1p53um'
# project_name = 'ten_deg_sony_green_balance_v2_rgb_2p04x2p04x2p04um_f1p53um'

# project_name_init = 'zero_deg_sony_green_balance_v2_rgb_2p04x2p04x2p04um_f1p53um'
# project_name = 'zero_deg_sony_green_balance_v2_rgb_2p04x2p04x2p04um_f1p53um'

# project_name_init = 'zero_deg_sony_green_balance_v3_enforce_rgb_2p091x2p091x2p04um_f1p53um'
# project_name = 'zero_deg_sony_green_balance_v3_enforce_rgb_2p091x2p091x2p04um_f1p53um'

# project_name_init = 'zero_deg_sony_green_balance_v3_enforce_rgb_2p091x2p091x2p04um_f1p53um_debug'
# project_name = 'zero_deg_sony_green_balance_v3_enforce_rgb_2p091x2p091x2p04um_f1p53um_debug'


#
# Green balance END
#


#
# PDAF opt START
#


# project_name_init = 'normal_sony_pdaf_opt_v1_rgb_2p04x2p04x2p04um_f1p53um'
# project_name = 'normal_sony_pdaf_opt_v1_rgb_2p04x2p04x2p04um_f1p53um'


# project_name_init = 'normal_sony_pdaf_opt_v1_rgb_2p04x2p04x2p04um_f1p53um_test'
# project_name = 'normal_sony_pdaf_opt_v1_rgb_2p04x2p04x2p04um_f1p53um_test'

# project_name_init = 'normal_sony_pdaf_opt_v1_rgb_2p04x2p04x2p04um_f1p53um_weight'
# project_name = 'normal_sony_pdaf_opt_v1_rgb_2p04x2p04x2p04um_f1p53um_weight'


#
# PDAF opt END
#



#
# YEAR 2 BASELINE opt START
#

# project_name_init = 'sony_year2_baseline_10layer_p085um_rgb_2p04x2p04x2p04um_f1p53um'
# project_name = 'sony_year2_baseline_10layer_p085um_rgb_2p04x2p04x2p04um_f1p53um'

# project_name_init = 'sony_year2_baseline_10layer_p085um_rgb_2p125x2p125x2p04um_f1p53um'
# project_name = 'sony_year2_baseline_10layer_p085um_rgb_2p125x2p125x2p04um_f1p53um'

# project_name_init = 'sony_year2_v2_baseline_10layer_p085um_rgb_2p04x2p04x2p04um_f1p53um'
# project_name = 'sony_year2_v2_baseline_10layer_p085um_rgb_2p04x2p04x2p04um_f1p53um'

# project_name_init = 'sony_year2_v2_baseline_plane_wave_10layer_p085um_rgb_2p04x2p04x2p04um_f1p53um'
# project_name = 'sony_year2_v2_baseline_plane_wave_10layer_p085um_rgb_2p04x2p04x2p04um_f1p53um'

# project_name_init = 'sony_year2_v2_baseline_no_gb_10layer_p085um_rgb_2p04x2p04x2p04um_f1p53um'
# project_name = 'sony_year2_v2_baseline_no_gb_10layer_p085um_rgb_2p04x2p04x2p04um_f1p53um'

# project_name_init = 'sony_year2_v2_baseline_plane_wave_no_gb_10layer_p085um_rgb_2p04x2p04x2p04um_f1p53um'
# project_name = 'sony_year2_v2_baseline_plane_wave_no_gb_10layer_p085um_rgb_2p04x2p04x2p04um_f1p53um'

# project_name_init = 'baseline_2022'
# project_name = 'baseline_2022'

python_src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))

if running_on_local_machine:
    # projects_directory_location_init = r"C:\Users\Ian\Dropbox\Caltech\Faraon Group\Simulations\Exploratory Quick Sims\NRL Grant 20220720 Schematic\dev"
    projects_directory_location_init = python_src_directory
    projects_directory_location = projects_directory_location_init          # Change if necessary to differentiate  
else:
    #! do not include spaces in filepaths passed to linux
    projects_directory_location_init = "/central/groups/Faraon_Computing/ian/sony"
    projects_directory_location = "/central/groups/Faraon_Computing/ian/sony"
    
project_name_init = os.path.basename(python_src_directory)
project_name = 'ares_' + project_name_init

# Initial Project File Directory
# projects_directory_location_init += "/" + project_name_init
# Final Output Project File Directory
projects_directory_location += "/" + project_name

#
# YEAR 2 BASELINE opt END
#

#
# YEAR 2 BAND CENTERING opt START
#

# project_name_init = 'sony_year2_v2_gaussian_no_gb_band_centers_10layer_p085um_rgb_2p04x2p04x2p04um_f1p53um'
# project_name = 'sony_year2_v2_gaussian_no_gb_band_centers_10layer_p085um_rgb_2p04x2p04x2p04um_f1p53um'

# project_name_init = 'sony_year2_v2_gaussian_no_gb_band_centers_8layer_p085um_rgb_2p04x2p04x2p04um_f1p53um'
# project_name = 'sony_year2_v2_gaussian_no_gb_band_centers_8layer_p085um_rgb_2p04x2p04x2p04um_f1p53um'

# project_name_init = 'sony_year2_v2_gaussian_no_gb_band_centers_5layer_p085um_rgb_2p04x2p04x2p04um_f1p53um'
# project_name = 'sony_year2_v2_gaussian_no_gb_band_centers_5layer_p085um_rgb_2p04x2p04x2p04um_f1p53um'

# project_name_init = 'sony_year2_v2_gaussian_no_gb_band_centers_3layer_p085um_rgb_2p04x2p04x2p04um_f1p53um'
# project_name = 'sony_year2_v2_gaussian_no_gb_band_centers_3layer_p085um_rgb_2p04x2p04x2p04um_f1p53um'

# project_name_init = 'sony_year2_v3_gaussian_no_gb_band_centers_10layer_p085um_rgb_2p04x2p04x2p04um_f1p53um'
# project_name = 'sony_year2_v3_gaussian_no_gb_band_centers_10layer_p085um_rgb_2p04x2p04x2p04um_f1p53um'


# project_name_init = 'sony_year2_v4_gaussian_no_gb_band_centers_10layer_p085um_rgb_2p04x2p04x2p04um_f1p53um'
# project_name = 'sony_year2_v4_gaussian_no_gb_band_centers_10layer_p085um_rgb_2p04x2p04x2p04um_f1p53um'

# project_name_init = 'sony_year2_v4_gaussian_no_gb_band_centers_8layer_p085um_rgb_2p04x2p04x2p04um_f1p53um'
# project_name = 'sony_year2_v4_gaussian_no_gb_band_centers_8layer_p085um_rgb_2p04x2p04x2p04um_f1p53um'


# project_name_init = 'sony_year2_v4_gaussian_no_gb_band_centers_5layer_p085um_rgb_2p04x2p04x2p04um_f1p53um'
# project_name = 'sony_year2_v4_gaussian_no_gb_band_centers_5layer_p085um_rgb_2p04x2p04x2p04um_f1p53um'


#
# YEAR 2 BAND CENTERING opt END
#


#
# YEAR 2 FEATURE BLUR START
#

# project_name_init = 'sony_year2_v4_gaussian_band_centers_10layer_p085um_rgb_feat_blur_2p04x2p04x2p04um_f1p53um'
# project_name = 'sony_year2_v4_gaussian_band_centers_10layer_p085um_rgb_feat_blur_2p04x2p04x2p04um_f1p53um'

# project_name_init = 'sony_year2_v4_gaussian_band_centers_10layer_p085um_rgb_feat_blur_recenter_2p04x2p04x2p04um_f1p53um'
# project_name = 'sony_year2_v4_gaussian_band_centers_10layer_p085um_rgb_feat_blur_recenter_2p04x2p04x2p04um_f1p53um'

# project_name_init = 'sony_year2_v4_gaussian_band_centers_5layer_p085um_rgb_feat_blur_recenter_2p04x2p04x2p04um_f1p53um'
# project_name = 'sony_year2_v4_gaussian_band_centers_5layer_p085um_rgb_feat_blur_recenter_2p04x2p04x2p04um_f1p53um'

# project_name_init = 'sony_year2_v4_gaussian_band_centers_3layer_p085um_rgb_feat_blur_recenter_2p04x2p04x2p04um_f1p53um'
# project_name = 'sony_year2_v4_gaussian_band_centers_3layer_p085um_rgb_feat_blur_recenter_2p04x2p04x2p04um_f1p53um'


#
# YEAR 2 FEATURE BLUR END
#

#
# YEAR 2 EXPLORE 85nm START
#


# project_name_init = 'sony_year2_v4_gaussian_band_centers_10layer_f_number_p085um_rgb_2p04x2p04x2p04um_f1p53um'
# project_name = 'sony_year2_v4_gaussian_band_centers_10layer_f_number_p085um_rgb_2p04x2p04x2p04um_f1p53um'

# project_name_init = 'sony_year2_v4_gaussian_band_centers_5layer_f_number_p085um_rgb_2p04x2p04x2p04um_f1p53um'
# project_name = 'sony_year2_v4_gaussian_band_centers_5layer_f_number_p085um_rgb_2p04x2p04x2p04um_f1p53um'

# project_name_init = 'sony_year2_v4_gaussian_band_centers_3layer_f_number_p085um_rgb_2p04x2p04x2p04um_f1p53um'
# project_name = 'sony_year2_v4_gaussian_band_centers_3layer_f_number_p085um_rgb_2p04x2p04x2p04um_f1p53um'


# project_name_init = 'sony_year2_v4_gaussian_band_centers_10layer_f_number_p085um_rgb_shuffle_green_2p04x2p04x2p04um_f1p53um'
# project_name = 'sony_year2_v4_gaussian_band_centers_10layer_f_number_p085um_rgb_shuffle_green_2p04x2p04x2p04um_f1p53um'


#
# YEAR 2 EXPLORE 85nm END
#


#
#* Mesh
#
mesh_spacing_um = 0.017
geometry_spacing_minimum_um = 0.051#0.017#0.051
geometry_spacing_lateral_um = 0.085#0.051#0.085#0.051#0.085#0.051#0.102#0.085#0.068#0.051#0.102#0.068#0.102#0.085#0.068#0.051
# mesh_to_geometry_factor = int( ( geometry_spacing_um + 0.5 * mesh_spacing_um ) / mesh_spacing_um )

device_scale_um = 0.051

use_smooth_blur = False#True
min_feature_size_um = 0.085
min_feature_size_voxels = min_feature_size_um / geometry_spacing_lateral_um
blur_half_width_voxels = int( np.ceil( ( min_feature_size_voxels - 1 ) / 2. ) )


#
#* Optical
#
background_index = 1.5		# TiO2
min_device_index = 1.5
max_device_index = 2.4		# SiO2

min_device_permittivity = min_device_index**2
max_device_permittivity = max_device_index**2

init_permittivity_0_1_scale = 0.5


# focal_length_um = geometry_spacing_minimum_um * 30
focal_length_um = device_scale_um * 30#40#10#20#40#50#30
focal_plane_center_lateral_um = 0
focal_plane_center_vertical_um = -focal_length_um

#
#* Device
#

num_vertical_layers = 10#40#10#40#20#40#10#1#2#6#4#10#2#1#10#40#10#40#15#15#30#40#15#30#10#20#30#15#5#10#20#40

vertical_layer_height_um = 0.051 * 4# * 4# * 2# * 8#0.051 * 4# * 8# * 8
# vertical_layer_height_um = 0.017 * 8

# vertical_layer_height_voxels = int( vertical_layer_height_um / geometry_spacing_minimum_um )
vertical_layer_height_voxels = int( vertical_layer_height_um / device_scale_um )

device_size_lateral_um = 2.04#2.125#2.091#2.04#2.091#2.04#geometry_spacing_lateral_um * 40
device_size_vertical_um = vertical_layer_height_um * num_vertical_layers

device_voxels_lateral = int(np.round( device_size_lateral_um / geometry_spacing_lateral_um))
# device_voxels_vertical = int(np.round( device_size_vertical_um / geometry_spacing_minimum_um))
device_voxels_vertical = int(np.round( device_size_vertical_um / device_scale_um))

device_voxels_simulation_mesh_lateral = 1 + int(device_size_lateral_um / mesh_spacing_um)
# device_voxels_simulation_mesh_lateral = 2 + int(device_size_lateral_um / mesh_spacing_um)
# device_voxels_simulation_mesh_vertical = 2 + int(device_size_vertical_um / mesh_spacing_um)
device_voxels_simulation_mesh_vertical = lookup_additional_vertical_mesh_cells[ num_vertical_layers ] + int(device_size_vertical_um / mesh_spacing_um)

device_vertical_maximum_um = device_size_vertical_um
device_vertical_minimum_um = 0


#
#* Spectral
#
add_infrared = False#True

lambda_min_um = 0.375
lambda_max_um = 0.725

if add_infrared:
	lambda_min_um = 0.375
	lambda_max_um = 1.0

lambda_min_eval_um = 0.375
lambda_max_eval_um = 0.725#0.850

if add_infrared:
	lambda_min_eval_um = 0.375
	lambda_max_eval_um = 1.0


bandwidth_um = lambda_max_um - lambda_min_um

num_bands = 3
num_points_per_band = 20#15
num_design_frequency_points = num_bands * num_points_per_band
num_eval_frequency_points = 6 * num_design_frequency_points

lambda_values_um = np.linspace(lambda_min_um, lambda_max_um, num_design_frequency_points)
max_intensity_by_wavelength = (device_size_lateral_um**2)**2 / (focal_length_um**2 * lambda_values_um**2)


num_dispersive_ranges = 1
dispersive_range_size_um = bandwidth_um / num_dispersive_ranges

dispersive_ranges_um = [
	[ lambda_min_um, lambda_max_um ]
]



#
# Add PDAF functionality
#
add_pdaf = False
if add_pdaf:
	assert device_voxels_lateral % 2 == 0, "Expected an even number here for PDAF implementation ease!"
	assert ( not add_infrared ), "For now, we are just going to look at RGB designs!"

pdaf_angle_phi_degrees = 45
pdaf_angle_theta_degrees = -8

pdaf_focal_spot_x_um = 0.25 * device_size_lateral_um
pdaf_focal_spot_y_um = 0.25 * device_size_lateral_um

pdaf_lambda_min_um = lambda_min_um
pdaf_lambda_max_um = lambda_min_um + bandwidth_um / 3.

pdaf_adj_src_idx = 0

pdaf_lambda_values_um = np.linspace(pdaf_lambda_min_um, pdaf_lambda_max_um, num_design_frequency_points)
pdaf_max_intensity_by_wavelength = (device_size_lateral_um**2)**2 / (focal_length_um**2 * pdaf_lambda_values_um**2)



#
#* FDTD
#
vertical_gap_size_um = device_scale_um * 15
lateral_gap_size_um = device_scale_um * 10#25
# lateral_gap_size_um = device_scale_um * 25
# lateral_gap_size_um = device_scale_um * 50

# vertical_gap_size_um = geometry_spacing_minimum_um * 15
# lateral_gap_size_um = geometry_spacing_minimum_um * 10#25
# # lateral_gap_size_um = geometry_spacing_minimum_um * 25
# # lateral_gap_size_um = geometry_spacing_minimum_um * 50

fdtd_region_size_vertical_um = 2 * vertical_gap_size_um + device_size_vertical_um + focal_length_um
fdtd_region_size_lateral_um = 2 * lateral_gap_size_um + device_size_lateral_um
fdtd_region_maximum_vertical_um = device_size_vertical_um + vertical_gap_size_um
fdtd_region_minimum_vertical_um = -focal_length_um - vertical_gap_size_um

fdtd_simulation_time_fs = 600#3000

#
#* Forward Source
#
lateral_aperture_um = 1.1 * device_size_lateral_um
src_maximum_vertical_um = device_size_vertical_um + vertical_gap_size_um * 2. / 3.
src_minimum_vertical_um = -focal_length_um - 0.5 * vertical_gap_size_um


weight_by_individual_wavelength = True

#
#* Input Sources
#

#
# todo: add frequency dependent profile
#
use_gaussian_sources = True
gaussian_waist_radius_um = 0.5 * ( 0.5 * 2.04 )

#
# take f number = 2.2, check index calc
#
use_airy_approximation = True
f_number = 2.2
airy_correction_factor = 0.84
mid_lambda_um = 0.55		# todo: make this depend on config rather than hardcoding it
# gaussian_waist_radius_um = airy_correction_factor * mid_lambda_um / ( np.pi * ( 1. / ( 2 * f_number ) ) )

if use_airy_approximation:
	gaussian_waist_radius_um = airy_correction_factor * mid_lambda_um * f_number
else:
	gaussian_waist_radius_um = mid_lambda_um / ( np.pi * ( 1. / ( 2 * f_number ) ) )

gaussian_waist_radius_um = 0.84 * 0.55 * 2.2 / 1.5

source_angle_theta_vacuum_deg = 0 #63
source_angle_theta_rad = np.arcsin( np.sin( source_angle_theta_vacuum_deg * np.pi / 180. ) * 1.0 / background_index )
source_angle_theta_deg = source_angle_theta_rad * 180. / np.pi

source_angle_phi_deg = 0


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

adjoint_src_to_dispersive_range_map = [
	0,
	0,
	0,
	0
]

#
#* Optimization
#

enforce_xy_gradient_symmetry = False#True#False#True
if enforce_xy_gradient_symmetry:
	assert ( int( np.round( device_size_lateral_um / geometry_spacing_lateral_um ) ) % 2 ) == 1, "We should have an odd number of design voxels across for this operation"

assert not enforce_xy_gradient_symmetry, "This feature seems to be making things worse - I think there is a small inherent symmetry in the simulation/import/monitors"

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


#
#* Spectral and polarization selectivity information
#
polarizations_focal_plane_map = [ ['x', 'y'], ['x', 'y'], ['x', 'y'], ['x', 'y'] ]

weight_focal_plane_map_performance_weighting = [ 1.0, 1.0, 1.0, 1.0 ]
weight_focal_plane_map = [ 1.0, 1.0, 1.0, 1.0 ]
weight_individual_wavelengths = np.ones( len( lambda_values_um ) )

if weight_by_individual_wavelength:
	weight_focal_plane_map_performance_weighting = [ 1.0, 1.0, 1.0, 1.0 ]
	weight_focal_plane_map = [ 1.0, 1.0, 1.0, 1.0 ]
	weight_individual_wavelengths = np.ones( len( lambda_values_um ) )


do_green_balance = False#True

polarization_name_to_idx = { 'x':0, 'y':1, 'z':2 }
spectral_focal_plane_map = [
	[0, num_points_per_band],
	[num_points_per_band, 2 * num_points_per_band],
	[2 * num_points_per_band, 3 * num_points_per_band],
	[num_points_per_band, 2 * num_points_per_band]
]



# desired_peaks_per_band_um = [ 0.46, 0.52, 0.61 ]
desired_peaks_per_band_um = [ 0.48, 0.52, 0.59 ]
# desired_peaks_per_band_um = [ 0.46, 0.52, 0.59 ]
desired_peak_location_per_band = [ np.argmin( ( lambda_values_um - desired_peaks_per_band_um[ idx ] )**2 ) for idx in range( 0, len( desired_peaks_per_band_um ) ) ]

print( desired_peak_location_per_band )

explicit_band_centering = True
shuffle_green = False#True#False#True#False

if explicit_band_centering:
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

# desired_peak_location_per_band = [ 0 for idx in range( 0, len( desired_peaks_per_band_um ) ) ]

# for idx in range( 0, len( desired_peaks_per_band_um ) ):

# 	peak_um = desired_peaks_per_band_um[ idx ]
# 	found = False

# 	for wl_idx in range( 0, num_design_frequency_points ):
# 		get_wl_um = lambda_values_um[ wl_idx ]

# 		if get_wl_um > peak_um:
# 			desired_peak_location_per_band[ idx ] = wl_idx
# 			found = True
# 			break

# 	assert found, "Peak location doesn't make sense"

# for idx in range( 0, len( desired_peak_location_per_band ) ):
# 	get_location = desired_peak_location_per_band[ idx ]

# 	half_width = num_points_per_band // 2

# 	assert ( get_location - half_width ) >= 0, "Not enough optimization room in short wavelength region!"
# 	assert ( get_location + half_width + 1 ) <= num_design_frequency_points, "Not enough optimization room in long wavelength region!"




if add_infrared:
	infrared_center_um = 0.94
	wl_idx_ir_start = 0
	for wl_idx in range( 0, len( lambda_values_um ) ):
		if lambda_values_um[ wl_idx ] > infrared_center_um:
			wl_idx_ir_start = wl_idx - 1
			break


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