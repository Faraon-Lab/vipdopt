from faulthandler import disable
import os
import shutil
import sys
import shelve
import gc

# Add filepath to default path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from SonyBayerFilterParameters import *
import SonyBayerFilter

from scipy import interpolate

# Lumerical hook Python library API
import imp
imp.load_source( "lumapi", lumapi_filepath)
import lumapi

import functools
import copy
import h5py
import numpy as np
import time

import queue
import subprocess
import platform
import re

print("All modules loaded.")
sys.stdout.flush()

#! Utility Functions
def backup_var(variable_name, savefile_name=None):
    '''Performs the equivalent of MATLAB's save(), albeit with some workarounds. Lumapi SimObjects cannot be saved this way.'''
    
    if savefile_name is None:
        savefile_name = shelf_fn
    
    global my_shelf
    my_shelf = shelve.open(savefile_name,'n')
    if variable_name != "my_shelf":
        try:
            my_shelf[variable_name] = globals()[variable_name]
            print(variable_name)
        except Exception as ex:
            # __builtins__, my_shelf, and imported modules can not be shelved.
            print('ERROR shelving: {0}'.format(variable_name))
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            # pass
    my_shelf.close()
    
    print("Successfully backed up variable: " + variable_name)
    sys.stdout.flush()

def backup_all_vars(savefile_name=None, global_enabled=True):
    '''Performs the equivalent of MATLAB's save(), albeit with some workarounds. Lumapi SimObjects cannot be saved this way.'''
    
    if savefile_name is None:
        savefile_name = shelf_fn
    
    if global_enabled:
        global my_shelf
        my_shelf = shelve.open(savefile_name,'n')
        for key in sorted(globals()):
            if key not in ['my_shelf', 'forward_e_fields',
                           'focal_e_data', 'focal_h_data',
                           'mode_e_fields', 'mode_h_fields'
                           'get_focal_e_data', 'get_focal_h_data',
                           'get_mode_e_fields', 'get_mode_h_fields']:
                try:
                    my_shelf[key] = globals()[key]
                    #print(key)
                except Exception as ex:
                    # __builtins__, my_shelf, and imported modules can not be shelved.
                    print('ERROR shelving: {0}'.format(key))
                    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                    message = template.format(type(ex).__name__, ex.args)
                    print(message)
                    #pass
        my_shelf.close()
        
        
        print("Backup complete.")
    else:
        print("Backup disabled.")
        
        sys.stdout.flush()
    
def load_backup_vars(loadfile_name = None):
    '''Restores session variables from a save_state file. Debug purposes.'''
    #! This will overwrite all current session variables!
    # except for the following:
    do_not_overwrite = ['running_on_local_machine',
                        'lumapi_filepath',
                        'start_from_step',
                        'projects_directory_location',
                        'projects_directory_location_init',
                        'plot_directory_location',
                        'start_epoch', 'num_epochs', 'start_iter', 
                        'num_iterations_per_epoch']
    
    if loadfile_name is None:
        loadfile_name = shelf_fn
    
    global my_shelf
    my_shelf = shelve.open(shelf_fn, flag='r')
    for key in my_shelf:
        if key not in do_not_overwrite:
            try:
                globals()[key]=my_shelf[key]
            except Exception as ex:
                print('ERROR retrieving shelf: {0}'.format(key))
                print("An exception of type {0} occurred. Arguments:\n{1!r}".format(type(ex).__name__, ex.args))
                #pass
    my_shelf.close()
    
    # Restore lumapi SimObjects as dictionaries
    for key,val in fdtd_objects.items():
        globals()[key]=val
    
    print("Successfully loaded backup.")
    sys.stdout.flush()

def isolate_filename(address):
    return os.path.basename(address)

def convert_root_folder(address, root_folder_new):
    new_address = os.path.join(root_folder_new, isolate_filename(address))
    return new_address

def create_dict_nx(dict_name):
    '''Checks if dictionary already exists in globals(). If not, creates it as empty.'''
    if dict_name in globals():
            print(dict_name +' already exists in globals().')
    else:
        globals()[dict_name] = {}

def get_slurm_node_list( slurm_job_env_variable=None ):
    if slurm_job_env_variable is None:
        slurm_job_env_variable = os.getenv('SLURM_JOB_NODELIST')
    if slurm_job_env_variable is None:
        raise ValueError('Environment variable does not exist.')

    solo_node_pattern = r'hpc-\d\d-[\w]+'
    cluster_node_pattern = r'hpc-\d\d-\[.*?\]'
    solo_nodes = re.findall(solo_node_pattern, slurm_job_env_variable)
    cluster_nodes = re.findall(cluster_node_pattern, slurm_job_env_variable)
    inner_bracket_pattern = r'\[(.*?)\]'

    output_arr = solo_nodes
    for cluster_node in cluster_nodes:
        prefix = cluster_node.split('[')[0]
        inside_brackets = re.findall(inner_bracket_pattern, cluster_node)[0]
        # Split at commas and iterate through results
        for group in inside_brackets.split(','):
            # Split at hyphen. Get first and last number. Create string in range
            # from first to last.
            node_clump_split = group.split('-')
            starting_number = int(node_clump_split[0])
            try:
                ending_number = int(node_clump_split[1])
            except IndexError:
                ending_number = starting_number
            for i in range(starting_number, ending_number+1):
                # Can use print("{:02d}".format(1)) to turn a 1 into a '01'
                # string. 111 -> 111 still, in case nodes hit triple-digits.
                output_arr.append(prefix + "{:02d}".format(i))
    return output_arr

if not running_on_local_machine:
    num_nodes_available = int( sys.argv[ 1 ] )	# should equal --nodes in batch script. Could also get this from len(cluster_hostnames)
    num_cpus_per_node = 8
    cluster_hostnames = get_slurm_node_list()
    
    print(f'There are {len(cluster_hostnames)} nodes available.')
    sys.stdout.flush()

class DevicePermittivityModel():
    '''Simple dispersion model where dispersion is ignored.'''
    # If used in multiple places, it could be called something like "NoDispersion" and be initialized with a give permittivity to return
    # Used to provide a dispersion model object that supports the function average_permittivity() that computes permittivity based on a 
    # wavelength range, when there is an actual dispersion model to consider

    def average_permittivity(self, wl_range_um_):
        '''Computes permittivity based on a wavelength range and the dispersion model of this object.'''
        return max_device_permittivity

def index_from_permittivity(permittivity_):
    '''Checks all permittivity values are real and then takes square root to give index.'''
    assert np.all(np.imag(permittivity_) == 0), 'Not expecting complex index values right now!'

    return np.sqrt(permittivity_)

def softplus( x_in ):
    '''Softplus is a smooth approximation to the ReLu function.
    It equals the sigmoid function when kappa = 1, and approaches the ReLu function as kappa -> infinity.
    See: https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html and https://pat.chormai.org/blog/2020-relu-softplus'''
   # return np.log( 1 + np.exp( x_in ) )
    return np.log( 1 + np.exp( softplus_kappa * x_in ) ) / softplus_kappa

def softplus_prime( x_in ):
    '''Derivative of the softplus function w.r.t. x, as defined in softplus()'''
    # return ( 1. / ( 1 + np.exp( -x_in ) ) )
    return ( 1. / ( 1 + np.exp( -softplus_kappa * x_in ) ) )



#
# Create project folder and save out the parameter file for documentation for this optimization
#
python_src_directory = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '.'))

if not os.path.isdir(projects_directory_location):
    os.mkdir(projects_directory_location)

log_file = open( projects_directory_location + "/log.txt", 'w' )
log_file.write( "Log\n" )
log_file.close()


#
# Create FDTD hook
#
fdtd_hook = lumapi.FDTD()


# Create new file called optimization.fsp and save
fdtd_hook.newproject()
if start_from_step != 0:
    fdtd_hook.load(projects_directory_location + "/optimization")
fdtd_hook.save(projects_directory_location + "/optimization")
# Copy Python code files to the output project directory
shutil.copy2(python_src_directory + "/SonyBayerFilterParameters.py",
             projects_directory_location + "/SonyBayerFilterParameters.py")
shutil.copy2(python_src_directory + "/SonyBayerFilter.py",
             projects_directory_location + "/SonyBayerFilter.py")
shutil.copy2(python_src_directory + "/SonyBayerFilterOptimization.py",
             projects_directory_location + "/SonyBayerFilterOptimization.py")

#! Step 0 Functions

def disable_all_sources():
    '''Disable all sources in the simulation, so that we can selectively turn single sources on at a time'''
    lumapi.evalScript(fdtd_hook.handle, 'switchtolayout;')

    for xy_idx in range(0, 2):
        fdtd_hook.select( forward_sources[xy_idx]['name'] )
        fdtd_hook.set( 'enabled', 0 )

    for adj_src_idx in range(0, num_adjoint_sources):
        for xy_idx in range(0, 2):
            fdtd_hook.select( adjoint_sources[adj_src_idx][xy_idx]['name'] )
            fdtd_hook.set( 'enabled', 0 )

def disable_objects(object_list):
    '''Disable selected monitors in the simulation, to save on memory and runtime'''
    lumapi.evalScript(fdtd_hook.handle, 'switchtolayout;')
    
    for object_name in object_list:
        try:
            fdtd_hook.select(object_name)
            fdtd_hook.set('enabled', 0)
        except Exception as err:
            pass

#*---- ADD TO FDTD ENVIRONMENT AND OBJECTS ----

if start_from_step == 0:
    #
    # Step 0: Set up structures, regions and monitors from which E-fields, Poynting fields, and transmission data will be drawn.
    # Also any other necessary editing of the Lumerical environment and objects.
    #
    
    print("Beginning Step 0: Lumerical Environment Setup")
    sys.stdout.flush()

    # Create dictionary to duplicate FDTD SimObjects as dictionaries for saving out
    fdtd_objects = {}
    def fdtd_simobject_to_dict( simObj ):
        ''' Duplicates a Lumapi SimObject (that cannot be shelved) to a Python dictionary equivalent.
         Works for dictionary objects and lists of dictionaries.'''
        
        if isinstance(simObj, lumapi.SimObject):
            duplicate_dict = {}
            for key in simObj._nameMap:
                duplicate_dict[key] = simObj[key]
            return duplicate_dict
        
        elif isinstance(simObj, list):
            duplicate_list = []
            for object in simObj:
                duplicate_list.append(fdtd_simobject_to_dict(object))
            return duplicate_list
        
        else:   
            raise ValueError('Tried to convert Lumapi SimObject that is not a list or dictionary.')

    #
    # Set up the FDTD region and mesh
    # We are using dict-like access: see https://support.lumerical.com/hc/en-us/articles/360041873053-Session-management-Python-API
    #
    fdtd = fdtd_hook.addfdtd()
    fdtd['x span'] = fdtd_region_size_lateral_um * 1e-6
    fdtd['y span'] = fdtd_region_size_lateral_um * 1e-6
    fdtd['z max'] = fdtd_region_maximum_vertical_um * 1e-6
    fdtd['z min'] = fdtd_region_minimum_vertical_um * 1e-6
    fdtd['simulation time'] = fdtd_simulation_time_fs * 1e-15
    fdtd['index'] = background_index
    fdtd_objects['FDTD'] = fdtd_simobject_to_dict(fdtd)

    device_mesh = fdtd_hook.addmesh()
    device_mesh['name'] = 'device_mesh'
    device_mesh['x'] = 0
    device_mesh['x span'] = fdtd_region_size_lateral_um * 1e-6
    device_mesh['y'] = 0
    device_mesh['y span'] = fdtd_region_size_lateral_um * 1e-6
    device_mesh['z max'] = ( device_vertical_maximum_um + 0.5 ) * 1e-6
    device_mesh['z min'] = ( device_vertical_minimum_um - 0.5 ) * 1e-6
    device_mesh['dx'] = mesh_spacing_um * 1e-6
    device_mesh['dy'] = mesh_spacing_um * 1e-6
    device_mesh['dz'] = mesh_spacing_um * 1e-6
    fdtd_objects['device_mesh'] = fdtd_simobject_to_dict(device_mesh)

    #
    # General polarized source information
    #
    xy_pol_rotations = [0, 90]
    xy_names = ['x', 'y']

    #
    #* Add a Gaussian wave forward source at angled incidence
    # (Deprecated) Add a TFSF plane wave forward source at normal incidence
    #
    forward_sources = []
    def snellRefrOffset(src_angle_incidence, object_list):
        ''' Takes a list of objects in between source and device, retrieves heights and indices,
        and calculates injection angle so as to achieve the desired angle of incidence upon entering the device.'''
        # structured such that n0 is the input interface and n_end is the last above the device
        # height_0 should be 0
        
        input_theta = 0
        r_offset = 0

        heights = [0.]
        indices = [1.]
        total_height = device_vertical_maximum_um
        max_height = src_maximum_vertical_um + src_hgt_Si
        for object_name in object_list:
            hgt = fdtd_hook.getnamed(object_name, 'z span')/1e-6
            total_height += hgt
            indices.append(float(fdtd_hook.getnamed(object_name, 'index')))
            
            if total_height >= max_height:
                heights.append(hgt-(total_height - max_height))
                break
            else:
                heights.append(hgt)
    
        # print(f'Device ends at z = {device_vertical_maximum_um}; Source is located at z = {max_height}')
        # print(f'Heights are {heights} with corresponding indices {indices}')
        # sys.stdout.flush()

        src_angle_incidence = np.radians(src_angle_incidence)
        snellConst = float(indices[-1])*np.sin(src_angle_incidence)
        input_theta = np.degrees(np.arcsin(snellConst/float(indices[0])))

        r_offset = 0
        for cnt, h_i in enumerate(heights):
            s_i = snellConst/indices[cnt]
            r_offset = r_offset + h_i*s_i/np.sqrt(1-s_i**2)

        return [input_theta, r_offset]

    for xy_idx in range(0, 2):
        # forward_src = fdtd_hook.addtfsf()
        # forward_src['name'] = 'forward_src_' + xy_names[xy_idx]
        # forward_src['angle phi'] = xy_phi_rotations[xy_idx]
        # forward_src['direction'] = 'Backward'
        # forward_src['x span'] = lateral_aperture_um * 1e-6
        # forward_src['y span'] = lateral_aperture_um * 1e-6
        # forward_src['z max'] = src_maximum_vertical_um * 1e-6
        # forward_src['z min'] = src_minimum_vertical_um * 1e-6
        # forward_src['wavelength start'] = lambda_min_um * 1e-6
        # forward_src['wavelength stop'] = lambda_max_um * 1e-6

        input_theta, r_offset = snellRefrOffset(src_angle_incidence, objects_above_device)

        forward_src = fdtd_hook.addgaussian()
        forward_src['name'] = 'forward_src_'+xy_names[xy_idx]
        forward_src['injection axis'] = 'z-axis'
        forward_src['direction'] = 'Backward'
        forward_src['polarization angle'] = xy_pol_rotations[xy_idx]  # degrees
        forward_src['angle theta'] = input_theta  # degrees
        forward_src['angle phi'] = src_phi_incidence    # degrees

        forward_src['x'] = r_offset*np.cos(np.radians(src_phi_incidence)) * 1e-6 * 1e-6
        forward_src['x span'] = lateral_aperture_um * 1e-6 * (4/3)
        forward_src['y'] = r_offset*np.sin(np.radians(src_phi_incidence)) * 1e-6
        forward_src['y span'] = lateral_aperture_um * 1e-6 * (4/3)
        forward_src['z'] = (src_maximum_vertical_um + src_hgt_Si) * 1e-6
        forward_src['wavelength start'] = lambda_min_um * 1e-6
        forward_src['wavelength stop'] = lambda_max_um * 1e-6

        forward_src["frequency dependent profile"] = 1
        forward_src["number of field profile samples"] = 150
        forward_src['beam parameters'] = 'Waist size and position'
        forward_src['waist radius w0'] = src_beam_rad * 1e-6
        forward_src['distance from waist'] = 0
        # forward_src['beam parameters'] = 'Beam size and divergence angle'
        # forward_src['beam radius wz'] = 100*1e-6
        # forward_src['divergence angle'] = 3.42365 # degrees
        # forward_src['beam radius wz'] = src_beam_rad*1e-6
        # forward_src['divergence angle'] = 0.5 * \
        #     (lambda_min_um+lambda_max_um)*1e-6/(np.pi*src_beam_rad)*(180/np.pi)
        # forward_src['beam radius wz'] = src_beam_rad

        forward_sources.append(forward_src)
    fdtd_objects['forward_sources'] = fdtd_simobject_to_dict(forward_sources)
    
    #
    # Place dipole adjoint sources at the focal plane that can ring in both
    # x-axis and y-axis
    #
    adjoint_sources = []

    for adj_src_idx in range(0, num_adjoint_sources):
        adjoint_sources.append([])
        for xy_idx in range(0, 2):
            # adj_src = fdtd_hook.adddipole()
            # adj_src['name'] = 'adj_src_' + str(adj_src_idx) + xy_names[xy_idx]
            # adj_src['x'] = adjoint_x_positions_um[adj_src_idx] * 1e-6
            # adj_src['y'] = adjoint_y_positions_um[adj_src_idx] * 1e-6
            # adj_src['z'] = adjoint_vertical_um * 1e-6
            # adj_src['theta'] = 90
            # adj_src['phi'] = xy_pol_rotations[xy_idx]
            # adj_src['wavelength start'] = lambda_min_um * 1e-6
            # adj_src['wavelength stop'] = lambda_max_um * 1e-6
            
            adj_src = fdtd_hook.addgaussian()
            adj_src['name'] = 'adj_src_' + str(adj_src_idx) + xy_names[xy_idx]
            adj_src['injection axis'] = 'z-axis'
            adj_src['direction'] = 'Forward'
            adj_src['polarization angle'] = xy_pol_rotations[xy_idx]  # degrees
            adj_src['wavelength start'] = lambda_min_um * 1e-6
            adj_src['wavelength stop'] = lambda_max_um * 1e-6
            
            adj_src['x'] = adjoint_x_positions_um[adj_src_idx] * 1e-6
            adj_src['y'] = adjoint_y_positions_um[adj_src_idx] * 1e-6
            adj_src['z'] = adjoint_vertical_um * 1e-6
            adj_src['x span'] = 0.5 * device_size_lateral_um * 1e-6
            adj_src['y span'] = 0.5 * device_size_lateral_um * 1e-6

            adj_src["frequency dependent profile"] = 1
            adj_src["number of field profile samples"] = 150
            adj_src['beam parameters'] = 'Waist size and position'
            adj_src['waist radius w0'] = adj_src_beam_rad * 1e-6
            adj_src['distance from waist'] = 0

            adjoint_sources[adj_src_idx].append(adj_src)
    fdtd_objects['adjoint_sources'] = fdtd_simobject_to_dict(adjoint_sources)
    
    #
    # Set up the volumetric electric field monitor inside the design region.  We will need this to compute
    # the adjoint gradient
    #
    design_efield_monitor = fdtd_hook.addprofile()
    design_efield_monitor['name'] = 'design_efield_monitor'
    design_efield_monitor['monitor type'] = '3D'
    design_efield_monitor['x span'] = device_size_lateral_um * 1e-6
    design_efield_monitor['y span'] = device_size_lateral_um * 1e-6
    design_efield_monitor['z max'] = device_vertical_maximum_um * 1e-6
    design_efield_monitor['z min'] = device_vertical_minimum_um * 1e-6
    design_efield_monitor['override global monitor settings'] = 1
    design_efield_monitor['use wavelength spacing'] = 1
    design_efield_monitor['use source limits'] = 1
    design_efield_monitor['frequency points'] = num_design_frequency_points
    design_efield_monitor['output Hx'] = 0
    design_efield_monitor['output Hy'] = 0
    design_efield_monitor['output Hz'] = 0
    fdtd_objects['design_efield_monitor'] = fdtd_simobject_to_dict(design_efield_monitor)

    #
    # Set up adjoint point monitors to get electric field strength at focus spots.  This will allow us to
    # compute the figure of merit as well as weight the adjoint simulations properly in calculation of the
    # gradient.
    #
    focal_monitors = []

    for adj_src in range(0, num_adjoint_sources):
        focal_monitor = fdtd_hook.addpower()
        focal_monitor['name'] = 'focal_monitor_' + str(adj_src)
        focal_monitor['monitor type'] = 'point'
        focal_monitor['x'] = adjoint_x_positions_um[adj_src] * 1e-6
        focal_monitor['y'] = adjoint_y_positions_um[adj_src] * 1e-6
        focal_monitor['z'] = adjoint_vertical_um * 1e-6
        focal_monitor['override global monitor settings'] = 1
        focal_monitor['use wavelength spacing'] = 1
        focal_monitor['use source limits'] = 1
        focal_monitor['frequency points'] = num_design_frequency_points

        focal_monitors.append(focal_monitor)
    fdtd_objects['focal_monitors'] = fdtd_simobject_to_dict(focal_monitors)
    
    transmission_monitors = []

    for adj_src in range(0, num_adjoint_sources):
        transmission_monitor = fdtd_hook.addpower()
        transmission_monitor['name'] = 'transmission_monitor_' + str(adj_src)
        transmission_monitor['monitor type'] = '2D Z-normal'
        transmission_monitor['x'] = adjoint_x_positions_um[adj_src] * 1e-6
        transmission_monitor['x span'] = 0.5 * device_size_lateral_um * 1e-6
        transmission_monitor['y'] = adjoint_y_positions_um[adj_src] * 1e-6
        transmission_monitor['y span'] = 0.5 * device_size_lateral_um * 1e-6
        transmission_monitor['z'] = adjoint_vertical_um * 1e-6
        transmission_monitor['override global monitor settings'] = 1
        transmission_monitor['use wavelength spacing'] = 1
        transmission_monitor['use source limits'] = 1
        transmission_monitor['frequency points'] = num_design_frequency_points

        transmission_monitors.append(transmission_monitor)
    fdtd_objects['transmission_monitors'] = fdtd_simobject_to_dict(transmission_monitors)

    transmission_monitor_focal = fdtd_hook.addpower()
    transmission_monitor_focal['name'] = 'transmission_focal_monitor_'
    transmission_monitor_focal['monitor type'] = '2D Z-normal'
    transmission_monitor_focal['x'] = 0 * 1e-6
    transmission_monitor_focal['x span'] = device_size_lateral_um * 1e-6
    transmission_monitor_focal['y'] = 0 * 1e-6
    transmission_monitor_focal['y span'] = device_size_lateral_um * 1e-6
    transmission_monitor_focal['z'] = adjoint_vertical_um * 1e-6
    transmission_monitor_focal['override global monitor settings'] = 1
    transmission_monitor_focal['use wavelength spacing'] = 1
    transmission_monitor_focal['use source limits'] = 1
    transmission_monitor_focal['frequency points'] = num_design_frequency_points
    fdtd_objects['transmission_monitor_focal'] = fdtd_simobject_to_dict(transmission_monitor_focal)


    #
    # Add device region and create device permittivity
    #
    design_import = fdtd_hook.addimport()
    design_import['name'] = 'design_import'
    design_import['x span'] = device_size_lateral_um * 1e-6
    design_import['y span'] = device_size_lateral_um * 1e-6
    design_import['z max'] = device_vertical_maximum_um * 1e-6
    design_import['z min'] = device_vertical_minimum_um * 1e-6
    fdtd_objects['design_import'] = fdtd_simobject_to_dict(design_import)

    bayer_filter_size_voxels = np.array(
        [device_voxels_lateral, device_voxels_lateral, device_voxels_vertical])
    bayer_filter = SonyBayerFilter.SonyBayerFilter(
        bayer_filter_size_voxels,
        [ 0.0, 1.0 ],
        init_permittivity_0_1_scale,
        num_vertical_layers,
        vertical_layer_height_voxels)

    bayer_filter_region_x = 1e-6 * \
        np.linspace(-0.5 * device_size_lateral_um, 0.5 * 
                    device_size_lateral_um, device_voxels_lateral)
    bayer_filter_region_y = 1e-6 * \
        np.linspace(-0.5 * device_size_lateral_um, 0.5 * 
                    device_size_lateral_um, device_voxels_lateral)
    bayer_filter_region_z = 1e-6 * \
        np.linspace(device_vertical_minimum_um, 
                    device_vertical_maximum_um, device_voxels_vertical)

    # Populate a 3+1-D array at each xyz-position with the values of x, y, and z
    bayer_filter_region = np.zeros( 
        ( device_voxels_lateral, device_voxels_lateral, device_voxels_vertical, 3 ) )
    for x_idx in range( 0, device_voxels_lateral ):
        for y_idx in range( 0, device_voxels_lateral ):
            for z_idx in range( 0, device_voxels_vertical ):
                bayer_filter_region[ x_idx, y_idx, z_idx, : ] = [
                        bayer_filter_region_x[ x_idx ], bayer_filter_region_y[ y_idx ], bayer_filter_region_z[ z_idx ] ]

    gradient_region_x = 1e-6 * np.linspace(-0.5 * device_size_lateral_um, 
                                        0.5 * device_size_lateral_um, device_voxels_simulation_mesh_lateral)
    gradient_region_y = 1e-6 * np.linspace(-0.5 * device_size_lateral_um, 
                                        0.5 * device_size_lateral_um, device_voxels_simulation_mesh_lateral)
    gradient_region_z = 1e-6 * np.linspace(device_vertical_minimum_um, 
                                        device_vertical_maximum_um, device_voxels_simulation_mesh_vertical)

    disable_all_sources()

    print("Completed Step 0: Lumerical environment setup complete.")
    sys.stdout.flush()
    
    # backup_all_vars()


#! Step 2 Functions 

#
#* Set up queue for parallel jobs
#
jobs_queue = queue.Queue()

def add_job( job_name, queue_in ):
    full_name = projects_directory_location + "/" + job_name
    fdtd_hook.save( full_name )
    queue_in.put( full_name )

    return full_name

def run_jobs( queue_in ):
    small_queue = queue.Queue()

    while not queue_in.empty():
        for node_idx in range( 0, num_nodes_available ):
            # this scheduler schedules jobs in blocks of num_nodes_available, waits for them ALL to finish, and then schedules the next block
            # TODO: write a scheduler that enqueues a new job as soon as one finishes
            # not urgent because for the most part, all jobs take roughly the same time
            if queue_in.qsize() > 0:
                small_queue.put( queue_in.get() )

        run_jobs_inner( small_queue )

def run_jobs_inner( queue_in ):
    processes = []
    job_idx = 0
    while not queue_in.empty():
        get_job_path = queue_in.get()

        print(f'Node {cluster_hostnames[job_idx]} is processing file at {get_job_path}.')
        sys.stdout.flush()
        
        process = subprocess.Popen(
            [
                '/home/gdrobert/Develompent/adjoint_lumerical/inverse_design/run_proc.sh',
                cluster_hostnames[ job_idx ],
                get_job_path
            ]
        )
        processes.append( process )

        job_idx += 1
    
    completed_jobs = [ 0 for i in range( 0, len( processes ) ) ]
    while np.sum( completed_jobs ) < len( processes ):
        for job_idx in range( 0, len( processes ) ):
            if completed_jobs[ job_idx ] == 0:

                poll_result = processes[ job_idx ].poll()
                if not( poll_result is None ):
                    completed_jobs[ job_idx ] = 1

        time.sleep( 1 )

#
#* Functions for obtaining information from Lumerical monitors
#
def get_afield( monitor_name, field_indicator ):
    field_polarizations = [ field_indicator + 'x',
                       field_indicator + 'y', field_indicator + 'z' ]
    data_xfer_size_MB = 0

    start = time.time()

    field_pol_0 = fdtd_hook.getdata( monitor_name, field_polarizations[ 0 ] )
    data_xfer_size_MB += field_pol_0.nbytes / ( 1024. * 1024. )

    total_field = np.zeros( [ len (field_polarizations ) ] + 
                        list( field_pol_0.shape ), dtype=np.complex )
    total_field[ 0 ] = field_pol_0

    for pol_idx in range( 1, len( field_polarizations ) ):
        field_pol = fdtd_hook.getdata( monitor_name, field_polarizations[ pol_idx ] )
        data_xfer_size_MB += field_pol.nbytes / ( 1024. * 1024. )

        total_field[ pol_idx ] = field_pol

    elapsed = time.time() - start

    date_xfer_rate_MB_sec = data_xfer_size_MB / elapsed
    log_file = open( projects_directory_location + "/log.txt", 'a' )
    log_file.write( "Transferred " + str( data_xfer_size_MB ) + " MB\n" )
    log_file.write( "Data rate = " + str( date_xfer_rate_MB_sec ) + " MB/sec\n\n" )
    log_file.close()

    return total_field

def get_hfield( monitor_name ):
    return get_afield( monitor_name, 'H' )

def get_efield( monitor_name ):
    return get_afield( monitor_name, 'E' )

def get_transmission_magnitude( monitor_name ):
    read_T = fdtd_hook.getresult( monitor_name, 'T' )
    return np.abs( read_T[ 'T' ] )

def get_beamdata(src_name, field_indicator):
    field_polarizations = [ field_indicator + 'x',
                       field_indicator + 'y', field_indicator + 'z' ]
    data_xfer_size_MB = 0
    
    def extract_lambda_matches(field_pol_old):
        '''Wavelength matching - Getting the beam data in layout mode means that the wavelengths don't line up with what we've specified in params
        e.g. Layout Mode vector has 150 λ values, new vector has 45 λ values'''
        new_shape = list(field_pol_old.shape)[:-1] + [num_design_frequency_points]
        field_pol_new = np.zeros(new_shape, dtype=np.complex)
        
        for j, wl in enumerate(lambda_values_um):
            old_spectral_idx = min(range(len(field_dataset['lambda'])), 
                                           key=lambda i: abs(field_dataset['lambda'][i] - lambda_values_um[j]*1e-6))
            field_pol_new[:,:,:,j] = field_pol_old[:,:,:,old_spectral_idx]
            
        return field_pol_new
    

    start = time.time()
    
    field_dataset = fdtd_hook.getresult( src_name, 'fields' )        # dictionary with keys ['lambda', 'f', 'x', 'y', 'z', 'E', 'H', 'Lumerical_dataset']
                                                                     # the 'E' item has shape (nx, ny, nz, nλ, 3)
    field_pol_0 = field_dataset[field_indicator][:,:,:,:, 0]
    field_pol_0 = extract_lambda_matches(field_pol_0)           # Correct wavelength mismatch 
    data_xfer_size_MB += field_pol_0.nbytes / ( 1024. * 1024. )

    total_field = np.zeros( [ len (field_polarizations ) ] + 
                        list( field_pol_0.shape ), dtype=np.complex )
    total_field[ 0 ] = field_pol_0

    for pol_idx in range( 1, len( field_polarizations ) ):
        field_pol = field_dataset[field_indicator][:,:,:,:, pol_idx]
        field_pol = extract_lambda_matches(field_pol)
        data_xfer_size_MB += field_pol.nbytes / ( 1024. * 1024. )

        total_field[ pol_idx ] = field_pol

    elapsed = time.time() - start

    date_xfer_rate_MB_sec = data_xfer_size_MB / elapsed
    log_file = open( projects_directory_location + "/log.txt", 'a' )
    log_file.write( "Transferred " + str( data_xfer_size_MB ) + " MB\n" )
    log_file.write( "Data rate = " + str( date_xfer_rate_MB_sec ) + " MB/sec\n\n" )
    log_file.close()
    
    return total_field

def get_hfield_mode(src_name):
    return get_beamdata( src_name, 'H' )

def get_efield_mode(src_name):
    return get_beamdata( src_name, 'E' )


def poynting_flux(E, H, total_surface_area, real_e_h = True, dot_dS = True, pauseDebug=False):
    '''Calculates the expression \int (E x H).dS which, for E: Electric field; H: Magnetic field, is the Poynting flux.
    Assumes a regular and uniform mesh size throughout.'''
    
    # E, H should have shape (3, nx, ny, nz, nλ）
    
    if pauseDebug:
        print('3')
    
    # Get the cross product of E and H along the axis that defines the last axis of the cross product
    integral = np.cross(E, H, axis=0)            # shape: (3, nx, ny, nz)
    if real_e_h:
        integral = np.real(integral)
    # Do the dot product, i.e. take the z (normal)-component; 
    if dot_dS:
        integral = integral[2]                       # shape: (nx, ny, nz)
    # then integrate by summing everything and multiplying by S
    integral = np.sum(integral, (0,1,2)) * total_surface_area
    
    return integral

#
#* Set up some numpy arrays to handle all the data we will pull out of the simulation.
#
forward_e_fields = {}                       # Records E-fields throughout device volume
focal_e_data = {}                           # Records E-fields at focal plane quadrants (locations of adjoint sources) for forward source
focal_h_data = {}                           # Records H-fields at focal plane quadrants (locations of adjoint sources) for forward source
quadrant_transmission_data = {}             # Records transmission through each quadrant
mode_e_fields = {}                          # Records E-field profiles at focal plane quadrants 
mode_h_fields = {}                          # Records H-field profiles at focal plane quadrants 

# Stores various types of information for each epoch and iteration:
figure_of_merit_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
figure_of_merit_by_wl_evolution = np.zeros(
    (num_epochs, num_iterations_per_epoch, 4))
step_size_evolution = np.zeros((num_epochs, num_iterations_per_epoch))                  # alpha in Eq. S2, Optica paper supplement: https://doi.org/10.1364/OPTICA.384228
average_design_variable_change_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
max_design_variable_change_evolution = np.zeros((num_epochs, num_iterations_per_epoch))

transmission_by_focal_spot_evolution = np.zeros((num_epochs, num_iterations_per_epoch, 4))
transmission_by_wl_evolution = np.zeros((num_epochs, num_iterations_per_epoch, num_design_frequency_points))
intensity_fom_by_wavelength_evolution = np.zeros((num_epochs, num_iterations_per_epoch, num_design_frequency_points))
mode_overlap_fom_by_wavelength_evolution = np.zeros((num_epochs, num_iterations_per_epoch, num_design_frequency_points))

step_size_start = 1.

# Choose the dispersion model for this optimization
dispersion_model = DevicePermittivityModel()

restart = False                 #!! Set to true if restarting from a particular iteration - be sure to change start_epoch and start_iter
evaluate = False                # Set to true if evaluating device performance at this specific iteration - exports out transmission at focal plane and at focal spots
if restart or evaluate:
    cur_design_variable = np.load( 
            projects_directory_location + "/cur_design_variable.npy" )
    bayer_filter.w[0] = cur_design_variable
    bayer_filter.update_permittivity()

    step_size_evolution = np.load(
             projects_directory_location + "/step_size_evolution.npy")
    average_design_variable_change_evolution = np.load(
             projects_directory_location + "/average_design_change_evolution.npy")
    max_design_variable_change_evolution = np.load(
             projects_directory_location + "/max_design_change_evolution.npy")
    figure_of_merit_evolution = np.load(
             projects_directory_location + "/figure_of_merit.npy")
    figure_of_merit_by_wl_evolution = np.load(
             projects_directory_location + "/figure_of_merit_by_wl.npy" )
    transmission_by_focal_spot_evolution = np.load(
             projects_directory_location + "/transmission_by_focal_spot.npy")
    transmission_by_wl_evolution = np.load(
             projects_directory_location + "/transmission_by_wl.npy")

if evaluate:
    # Set sigmoid strength to most recent epoch and get (filtered) permittivity
    bayer_filter.update_filters( num_epochs - 1 )
    bayer_filter.update_permittivity()
    cur_density = bayer_filter.get_permittivity()

    fdtd_hook.switchtolayout()

    job_names = {}

    fdtd_hook.switchtolayout()
    
    #
    # Binarize the current permittivity
    #
    cur_density = 1.0 * np.greater_equal( cur_density, 0.5 )        # TODO: Why not greater_equal(cur_density, 0.0)?

    np.save( projects_directory_location + '/density_eval.npy', cur_density )

    dispersive_range_idx = 0

    dispersive_max_permittivity = dispersion_model.average_permittivity( dispersive_ranges_um[ dispersive_range_idx ] )
    disperesive_max_index = index_from_permittivity( dispersive_max_permittivity )

    fdtd_hook.switchtolayout()

    for xy_idx in range(0, 2):
        forward_sources[ xy_idx ]['wavelength start'] = lambda_min_eval_um * 1e-6
        forward_sources[ xy_idx ]['wavelength stop'] = lambda_max_eval_um * 1e-6

    design_efield_monitor[ 'enabled' ] = 0
    transmission_monitor_focal[ 'frequency points' ] = num_eval_frequency_points
    for adj_src in range(0, num_adjoint_sources):
        transmission_monitors[ adj_src ][ 'frequency points' ] = num_eval_frequency_points

    cur_permittivity = min_device_permittivity + ( dispersive_max_permittivity - min_device_permittivity ) * cur_density
    cur_index = index_from_permittivity( cur_permittivity )

    fdtd_hook.select( 'design_import' )
    fdtd_hook.importnk2( cur_index, bayer_filter_region_x, bayer_filter_region_y, bayer_filter_region_z )

    for xy_idx in range(0, 2):
        disable_all_sources()

        fdtd_hook.select( forward_sources[xy_idx]['name'] )
        fdtd_hook.set( 'enabled', 1 )

        job_name = 'forward_job_' + str( xy_idx ) + "_" + str( dispersive_range_idx ) + '.fsp'
        fdtd_hook.save( projects_directory_location + "/optimization.fsp" )
        job_names[ ( 'forward', xy_idx, dispersive_range_idx ) ] = add_job( job_name, jobs_queue )

    run_jobs( jobs_queue )


    eval_quadrant_transmission_data = np.zeros( ( 2, num_adjoint_sources, num_eval_frequency_points ) )
    eval_focal_transmission_data = np.zeros( ( 2, num_eval_frequency_points ) )
    for xy_idx in range(0, 2):
        fdtd_hook.load( job_names[ ( 'forward', xy_idx, dispersive_range_idx ) ] )

        eval_focal_transmission_data[ xy_idx ] = get_transmission_magnitude( transmission_monitor_focal[ 'name' ] )

        for adj_src_idx in dispersive_range_to_adjoint_src_map[ dispersive_range_idx ]:					
            eval_quadrant_transmission_data[ xy_idx ][ adj_src_idx ] = get_transmission_magnitude( transmission_monitors[ adj_src_idx ][ 'name' ] )

    np.save( projects_directory_location + '/eval_quadrant_transmission_data.npy', eval_quadrant_transmission_data )
    np.save( projects_directory_location + '/eval_focal_transmission_data.npy', eval_focal_transmission_data )

else:
        
    if start_from_step != 0:
        load_backup_vars()
        #pass

    #
    #* Run the optimization
    #
    
    start_epoch = 0
    for epoch in range(start_epoch, num_epochs):
        bayer_filter.update_filters(epoch)
        bayer_filter.update_permittivity()

        fdtd_hook.switchtolayout()

        start_iter = 0
        if epoch == start_epoch:
            start_iter = 0
            bayer_filter.current_iteration = start_epoch * num_iterations_per_epoch + start_iter
            
        for iteration in range(start_iter, num_iterations_per_epoch):
            print("Working on epoch " + str(epoch) + " and iteration " + str(iteration))
            sys.stdout.flush()

            job_names = {}

            fdtd_hook.switchtolayout()
            cur_density = bayer_filter.get_permittivity()

            #
            # Step 1: Run the forward optimization for both x- and y-polarized plane waves.  Run a different optimization for each color band so
            # that you can input a different index into each of those cubes.  Further note that the polymer slab needs to also change index when
            # the cube index changes.
            #

            if start_from_step == 0 or start_from_step == 1:
                if start_from_step != 0:
                    load_backup_vars()
            
                print("Beginning Step 1: Forward and Adjoint Optimizations")
                sys.stdout.flush()
    
                for dispersive_range_idx in range( 0, num_dispersive_ranges ):
                    dispersive_max_permittivity = dispersion_model.average_permittivity( dispersive_ranges_um[ dispersive_range_idx ] )
                    disperesive_max_index = index_from_permittivity( dispersive_max_permittivity )

                    fdtd_hook.switchtolayout()

                    cur_permittivity = min_device_permittivity + ( dispersive_max_permittivity - min_device_permittivity ) * cur_density
                    cur_index = index_from_permittivity( cur_permittivity )

                    fdtd_hook.select( 'design_import' )
                    fdtd_hook.importnk2( cur_index, bayer_filter_region_x, 
                                        bayer_filter_region_y, bayer_filter_region_z )

                    for xy_idx in range(0, 2):
                        disable_all_sources()

                        fdtd_hook.select( forward_sources[xy_idx]['name'] )
                        fdtd_hook.set( 'enabled', 1 )

                        job_name = 'forward_job_' + \
                                str( xy_idx ) + "_" + str( dispersive_range_idx ) + '.fsp'
                        fdtd_hook.save( projects_directory_location + "/optimization.fsp" )
                        job_names[ ('forward', xy_idx, dispersive_range_idx)] = add_job( job_name, jobs_queue )
                    
                    for xy_idx in range(0,2):
                        # Easier to initialize in a separate loop
                        mode_e_fields[xy_names[xy_idx]] = [
                            None for idx in range( 0, num_focal_spots ) ]
                        mode_h_fields[xy_names[xy_idx]] = [
                            None for idx in range( 0, num_focal_spots ) ]
                    
                    for adj_src_idx in range(0, num_adjoint_sources):
                        for xy_idx in range(0, 2):
                            disable_all_sources()

                            for not_adj_src_idx in range(0, num_adjoint_sources):
                                if not_adj_src_idx != adj_src_idx:
                                    fdtd_hook.select( adjoint_sources[not_adj_src_idx][xy_idx]['name'] )
                                    fdtd_hook.set( 'enabled', 1 )
                                    
                                    # Aim down and push above focal plane a bit
                                    fdtd_hook.set('z', -focal_length_um + mesh_spacing_um*5)
                                    fdtd_hook.set('direction', 'Backward')

                            job_name = 'adjoint_job_' + str( adj_src_idx ) + '_' + str( xy_idx ) + '_' + str( dispersive_range_idx ) + '.fsp'
                            fdtd_hook.save( projects_directory_location + "/optimization.fsp" )
                            job_names[ ( 'adjoint', adj_src_idx, xy_idx, dispersive_range_idx ) ] = add_job( job_name, jobs_queue )
                            
                            # Pull out adjoint source, user-defined mode data: Em, Hm
                            mode_e_fields[xy_names[xy_idx]][adj_src_idx] = get_efield_mode(adjoint_sources[adj_src_idx][xy_idx]['name'])
                            print(f'Accessed mode E-field data over focal quadrant {adj_src_idx}.')
                            
                            mode_h_fields[xy_names[xy_idx]][adj_src_idx] = get_hfield_mode(adjoint_sources[adj_src_idx][xy_idx]['name'])
                            print(f'Accessed mode E-field data over focal quadrant {adj_src_idx}.')
                            
                            # Set direction back to upwards, restore z position to focal plane
                            for not_adj_src_idx in range(0, num_adjoint_sources):
                                if not_adj_src_idx != adj_src_idx:
                                    fdtd_hook.select( adjoint_sources[not_adj_src_idx][xy_idx]['name'] )
                                    fdtd_hook.set('z', adjoint_vertical_um)
                                    fdtd_hook.set('direction', 'Forward')
                            


                run_jobs( jobs_queue )
                
                print("Completed Step 1: Forward and Adjoint Optimization Jobs Completed.")
                sys.stdout.flush()
                
                backup_all_vars()

            #
            # Step 2: Compute the figure of merit
            #

            if start_from_step == 0 or start_from_step == 2:
                if start_from_step != 0:
                    load_backup_vars()
            
                print("Beginning Step 2: Computing Figure of Merit")
                sys.stdout.flush()
                starttime=time.time()
                
                for xy_idx in range(0, 2):
                    focal_e_data[xy_names[xy_idx]] = [
                            None for idx in range( 0, num_focal_spots ) ]
                    focal_h_data[xy_names[xy_idx]] = [
                            None for idx in range( 0, num_focal_spots ) ]
                    quadrant_transmission_data[xy_names[xy_idx]] = [ 
                            None for idx in range( 0, num_focal_spots ) ]
                    mode_e_fields[xy_names[xy_idx]] = [
                            None for idx in range( 0, num_focal_spots ) ]
                    mode_h_fields[xy_names[xy_idx]] = [
                            None for idx in range( 0, num_focal_spots ) ]

                for dispersive_range_idx in range( 0, num_dispersive_ranges ):
                    for xy_idx in range(0, 2):
                        # TODO: Undo this bit when no longer build-testing
                        actual_directory = r'C:\Users\Ian\Dropbox\Caltech\Faraon Group\Simulations\Mode Overlap FoM\fom_dev\angInc_sony_test_fom_dev'
                        actual_filepath = convert_root_folder(job_names[ ( 'forward', xy_idx, dispersive_range_idx )], actual_directory)
                        #fdtd_hook.load( job_names[ ( 'forward', xy_idx, dispersive_range_idx ) ] )
                        print(f'Loading: {isolate_filename(actual_filepath)}')
                        sys.stdout.flush()
                        fdtd_hook.load(actual_filepath)

                        fwd_e_fields = get_efield(design_efield_monitor['name'])
                        print('Accessed design E-field monitor.')

                        # Populate forward_e_fields with device E-field info
                        if ( dispersive_range_idx == 0 ):
                            forward_e_fields[xy_names[xy_idx]] = fwd_e_fields
                        else:
                            get_dispersive_range_buckets = dispersive_range_to_adjoint_src_map[ dispersive_range_idx ]

                            for bucket_idx in range( 0, len( get_dispersive_range_buckets ) ):
                                # Each wavelength bucket says which wavelength (indices) correspond to which dispersive range,
                                # which itself contains a map for how the wavelength buckets correspond to which specific adjoint 
                                # sources on the focal plane.
                                get_wavelength_bucket = spectral_focal_plane_map[ get_dispersive_range_buckets[ bucket_idx ] ]

                                for spectral_idx in range( get_wavelength_bucket[ 0 ], get_wavelength_bucket[ 1 ] ):
                                    # fwd_e_fields has shape (3, nx, ny, nz, nλ)
                                    forward_e_fields[xy_names[xy_idx]][ :, :, :, :, spectral_idx ] = fwd_e_fields[ :, :, :, :, spectral_idx ]

                        # Populate arrays with adjoint source E-field, adjoint source H-field and quadrant transmission info
                        for adj_src_idx in dispersive_range_to_adjoint_src_map[ dispersive_range_idx ]:					
                            focal_e_data[xy_names[xy_idx]][ adj_src_idx ] = get_efield(transmission_monitors[adj_src_idx]['name'])
                            print(f'Accessed adjoint E-field data over focal quadrant {adj_src_idx}.')
                            
                            # NOTE: If you want focal_h_data['x'] to be an ndarray, it must be initialized as np.empty above.
                            focal_h_data[xy_names[xy_idx]][ adj_src_idx ] = get_hfield(transmission_monitors[adj_src_idx]['name'])
                            print(f'Accessed adjoint H-field data over focal quadrant {adj_src_idx}.')

                            quadrant_transmission_data[xy_names[xy_idx]][ adj_src_idx ] = get_transmission_magnitude(transmission_monitors[adj_src_idx]['name'])
                            print(f'Accessed transmission data for quadrant {adj_src_idx}.')

                            # # * OLD METHOD: Accessing source mode data by extracting from completed adjoint simulations
                            # # for adj_src_idx in range(0, num_adjoint_sources):
                            # lookup_dispersive_range_idx = adjoint_src_to_dispersive_range_map[ adj_src_idx ]
                            # actual_filepath = convert_root_folder(job_names[ ( 'adjoint', adj_src_idx, xy_idx, lookup_dispersive_range_idx ) ], actual_directory)
                            # #fdtd_hook.load( job_names[ ( 'forward', xy_idx, dispersive_range_idx ) ] )
                            # print(f'Loading: {isolate_filename(actual_filepath)}')
                            # fdtd_hook.load(actual_filepath)
                            # #fdtd_hook.load( job_names[ ( 'adjoint', adj_src_idx, xy_idx, lookup_dispersive_range_idx ) ] )
                            # #print('Loading .fsp file: '+job_names[('adjoint', adj_src_idx, xy_idx, lookup_dispersive_range_idx)])
                            # #sys.stdout.flush()
                            
                            
                            # mode_e_fields[xy_names[xy_idx]][adj_src_idx] = get_efield(transmission_monitors[adj_src_idx]['name'])
                            # print(f'Accessed mode E-field data over focal quadrant {adj_src_idx}.')
                            
                            # mode_h_fields[xy_names[xy_idx]][adj_src_idx] = get_hfield(transmission_monitors[adj_src_idx]['name'])
                            # print(f'Accessed mode E-field data over focal quadrant {adj_src_idx}.')
                            
                endtime=time.time()
                print(f'Loading data took: {(endtime-starttime)/60} min.')               


                # These arrays record intensity and transmission FoMs for all focal spots
                figure_of_merit_per_focal_spot = []
                transmission_per_quadrant = []
                # These arrays record the same, but for each wavelength
                transmission_by_wavelength = np.zeros(num_design_frequency_points)
                intensity_fom_by_wavelength = np.zeros(num_design_frequency_points)
                mode_overlap_fom_by_wavelength = np.zeros(num_design_frequency_points)

                for focal_idx in range(0, num_focal_spots):             # For each focal spot:
                    compute_fom = 0
                    compute_transmission_fom = 0

                    polarizations = polarizations_focal_plane_map[focal_idx]    # choose which polarization(s) are directed to this focal spot

                    for polarization_idx in range(0, len(polarizations)):
                        get_focal_e_data = focal_e_data[polarizations[polarization_idx]]                                # shape: (#focalspots, 3, nx, ny, nz, nλ)
                        get_focal_h_data = focal_h_data[polarizations[polarization_idx]]                                # shape: (#focalspots, 3, nx, ny, nz, nλ)
                        get_quad_transmission_data = quadrant_transmission_data[polarizations[polarization_idx]]        # shape: (#focalspots, nλ)
                        get_mode_e_fields = mode_e_fields[polarizations[polarization_idx]]                              # shape: (#focalspots, 3, nx, ny, nz, nλ)
                        get_mode_h_fields = mode_h_fields[polarizations[polarization_idx]]                              # shape: (#focalspots, 3, nx, ny, nz, nλ)

                        max_intensity_weighting = max_intensity_by_wavelength[spectral_focal_plane_map[focal_idx][0] : spectral_focal_plane_map[focal_idx][1] : 1]
                        # Use the max_intensity_weighting to renormalize the intensity distribution used to calculate the FoM
                        # and insert the custom weighting schemes here as well.
                        total_weighting = max_intensity_weighting / weight_focal_plane_map_performance_weighting[focal_idx]
                        one_over_weight_focal_plane_map_performance_weighting = 1. / weight_focal_plane_map_performance_weighting[focal_idx]

                        for spectral_idx in range(0, total_weighting.shape[0]):

                            weight_spectral_rejection = 1.0
                            if do_rejection:
                                weight_spectral_rejection = spectral_focal_plane_map_directional_weights[ focal_idx ][ spectral_idx ]

                            #* Transmission FoM Calculation: FoM = T
                            # \text{FoM}_{\text{transmission}} = \sum_i^{} \left{ \sum_\lambda^{} w_{\text{rejection}}(\lambda) w_{f_i} T_{f_i}(\lambda) \right}
                            # https://i.imgur.com/L414ybG.png  
                            compute_transmission_fom += weight_spectral_rejection * \
                                        get_quad_transmission_data[focal_idx][spectral_focal_plane_map[focal_idx][0] + spectral_idx] / \
                                                one_over_weight_focal_plane_map_performance_weighting
                            
                            # # Intensity FoM Calculation: FoM = |E(x_0)|^2                 
                            # # \text{FoM}_{\text{intensity}} = \sum_i^{} \left{ \sum_\lambda^{} w_{\text{rejection}}(\lambda) w_{f_i}
                            # # \frac{\left|E_{f_i}(\lambda)\right|^2}{I_{\text{norm, Airy}}}
                            # # https://i.imgur.com/5jcN01k.png
                            # compute_fom += weight_spectral_rejection * np.sum(
                            #     (
                            #         np.abs(get_focal_e_data[focal_idx][:, 0, 0, 0, spectral_focal_plane_map[focal_idx][0] + spectral_idx])**2 /
                            #         total_weighting[spectral_idx]
                            #     )
                            # )
                            
                            #* Mode Overlap FoM Calculation:                 
                            # \text{FoM}_{\text{m.o.}} = \sum_i^{} \left{ \sum_\lambda^{} w_{\text{rejection}}(\lambda) w_{f_i} 
                            # \frac{1/8}{I_{\text{norm, Airy}}} 
                            # \frac{\left| \int \textbf{E} \times\overline{\textbf{H}_m} \cdot d\textbf{S} + \int \overline{\textbf{E}_m} \times \textbf{H} \cdot d\textbf{S} \right|^2}
                            # {\int Re\left( \textbf{E}_m \times\overline{\textbf{H}_m} \right)\cdot d\textbf{S}}
                            # https://i.imgur.com/fCI56r1.png
                            
                            
                            
                            fom_integral_1 = poynting_flux(get_focal_e_data[focal_idx][:,:,:,:,spectral_focal_plane_map[focal_idx][0] + spectral_idx],
                                                  np.conj(get_mode_h_fields[focal_idx][:,:,:,:,spectral_focal_plane_map[focal_idx][0] + spectral_idx]),
                                                  (0.5*device_size_lateral_um)**2)
                            fom_integral_2 = poynting_flux(np.conj(get_mode_e_fields[focal_idx][:,:,:,:,spectral_focal_plane_map[focal_idx][0] + spectral_idx]),
                                                  get_focal_h_data[focal_idx][:,:,:,:,spectral_focal_plane_map[focal_idx][0] + spectral_idx],
                                                  (0.5*device_size_lateral_um)**2)
                            fom_integral_3 = poynting_flux(get_mode_e_fields[focal_idx][:,:,:,:,spectral_focal_plane_map[focal_idx][0] + spectral_idx],
                                                  np.conj(get_mode_h_fields[focal_idx][:,:,:,:,spectral_focal_plane_map[focal_idx][0] + spectral_idx]),
                                                  (0.5*device_size_lateral_um)**2, real_e_h = True)
                            
                            compute_fom += weight_spectral_rejection * np.sum(
                                    1/8 * (np.abs(fom_integral_1 + fom_integral_2)**2 / fom_integral_3
                                           ) / total_weighting[spectral_idx]
                            )

                            # Records intensity FOM per wavelength
                            intensity_fom_by_wavelength[ spectral_focal_plane_map[focal_idx][0] + spectral_idx ] += weight_spectral_rejection * np.sum(
                                (
                                    np.abs(get_focal_e_data[focal_idx][:, 0, 0, 0, spectral_focal_plane_map[focal_idx][0] + spectral_idx])**2 /
                                    total_weighting[spectral_idx]
                                )
                            )
                            # Records transmission FOM per wavelength
                            transmission_by_wavelength[ spectral_focal_plane_map[focal_idx][0] + spectral_idx ] += weight_spectral_rejection * \
                                    get_quad_transmission_data[focal_idx][spectral_focal_plane_map[focal_idx][0] + spectral_idx] / \
                                        one_over_weight_focal_plane_map_performance_weighting
                            
                            # Records mode overlap FOM per wavelength
                            mode_overlap_fom_by_wavelength[ spectral_focal_plane_map[focal_idx][0] + spectral_idx ] += weight_spectral_rejection * np.sum(
                                    1/8 * (np.abs(fom_integral_1 + fom_integral_2)**2 / fom_integral_3
                                           ) / total_weighting[spectral_idx]
                            )

                    figure_of_merit_per_focal_spot.append(compute_fom)
                    transmission_per_quadrant.append(compute_transmission_fom)


                figure_of_merit_per_focal_spot = np.array(figure_of_merit_per_focal_spot)
                transmission_per_quadrant = np.array(transmission_per_quadrant)

                # Copy everything so as not to disturb the original data
                process_fom_per_focal_spot = figure_of_merit_per_focal_spot.copy()
                process_transmission_per_quadrant = transmission_per_quadrant.copy()
                # process_fom_by_wavelength = intensity_fom_by_wavelength.copy()
                process_fom_by_wavelength = mode_overlap_fom_by_wavelength.copy()
                process_transmission_by_wavelength = transmission_by_wavelength.copy()
                if do_rejection:        # softplus each array
                    for idx in range( 0, len( figure_of_merit_per_focal_spot ) ):
                        process_fom_per_focal_spot[ idx ] = softplus( figure_of_merit_per_focal_spot[ idx ] )
                    for idx in range( 0, len( transmission_per_quadrant ) ):
                        process_transmission_per_quadrant[ idx ] = softplus( transmission_per_quadrant[ idx ] )
                    for idx in range( 0, len( intensity_fom_by_wavelength ) ):
                        process_fom_by_wavelength[ idx ] = softplus( intensity_fom_by_wavelength[ idx ] )
                    for idx in range( 0, len( transmission_by_wavelength ) ):
                        process_transmission_by_wavelength[ idx ] = softplus( transmission_by_wavelength[ idx ] )

                # TODO: Why this metric?
                performance_weighting = (2. / num_focal_spots) - process_fom_per_focal_spot**2 / np.sum(process_fom_per_focal_spot**2)
                if weight_by_quadrant_transmission:
                    performance_weighting = (2. / num_focal_spots) - process_transmission_per_quadrant**2 / np.sum(process_transmission_per_quadrant**2)

                if weight_by_individual_wavelength:
                    performance_weighting = (2. / num_design_frequency_points) - process_fom_by_wavelength**2 / np.sum( process_fom_by_wavelength**2 )
                    if weight_by_quadrant_transmission:
                        performance_weighting = (2. / num_design_frequency_points) - process_transmission_by_wavelength**2 / np.sum( process_transmission_by_wavelength**2 )

                # Zero-shift and renormalize
                if np.min( performance_weighting ) < 0:
                    performance_weighting -= np.min(performance_weighting)
                    performance_weighting /= np.sum(performance_weighting)


                figure_of_merit = np.sum(figure_of_merit_per_focal_spot)
                figure_of_merit_evolution[epoch, iteration] = figure_of_merit
                figure_of_merit_by_wl_evolution[epoch, iteration] = figure_of_merit_per_focal_spot

                transmission_by_focal_spot_evolution[epoch, iteration] = transmission_per_quadrant
                transmission_by_wl_evolution[epoch, iteration] = transmission_by_wavelength
                intensity_fom_by_wavelength_evolution[epoch, iteration] = intensity_fom_by_wavelength
                mode_overlap_fom_by_wavelength_evolution[epoch, iteration] = mode_overlap_fom_by_wavelength

                np.save(projects_directory_location + "/figure_of_merit.npy", figure_of_merit_evolution)
                np.save(projects_directory_location + "/figure_of_merit_by_wl.npy", figure_of_merit_by_wl_evolution)
                np.save(projects_directory_location + "/transmission_by_focal_spot.npy", transmission_by_focal_spot_evolution)
                np.save(projects_directory_location + "/transmission_by_wl.npy", transmission_by_wl_evolution)
                np.save(projects_directory_location + "/intensity_fom_by_wavelength.npy", intensity_fom_by_wavelength_evolution)

                print("Figure of Merit So Far:")
                print(figure_of_merit_evolution)

                print("Completed Step 2: Saved out figures of merit.")
                sys.stdout.flush()
                
                # starttime=time.time()
                # gc.collect()
                # np.save(projects_directory_location + '/fwd_e_x.npy', forward_e_fields['x'])
                # np.save(projects_directory_location + '/fwd_e_y.npy', forward_e_fields['y'])
                # np.save(projects_directory_location + '/focal_e_x.npy', focal_e_data['x'])
                # np.save(projects_directory_location + '/focal_e_x.npy', focal_e_data['y'])
                # np.save(projects_directory_location + '/focal_h_x.npy', focal_h_data['x'])
                # np.save(projects_directory_location + '/focal_h_x.npy', focal_h_data['y'])
                # endtime=time.time()
                # print(f'Saving data took: {(endtime-starttime)/60} min.')    
                # gc.collect()
                # backup_all_vars()
                # gc.collect()
                # endtime=time.time()
                # print(f'Saving data took: {(endtime-starttime)/60} min.')    
                
            #
            # Step 3: Run all the adjoint optimizations for both x- and y-polarized adjoint sources and use the results to compute the
            # gradients for x- and y-polarized forward sources.
            #
            
            if True:# start_from_step == 0 or start_from_step == 3:
                # if start_from_step != 0:
                #     load_backup_vars()
                #     forward_e_fields = {}
                #     forward_e_fields['x'] = np.load(projects_directory_location + '/fwd_e_x.npy')
                #     forward_e_fields['y'] = np.load(projects_directory_location + '/fwd_e_y.npy')
                #     focal_e_data = {}
                #     focal_e_data['x'] = np.load(projects_directory_location + '/focal_e_x.npy')
                #     focal_e_data['y'] = np.load(projects_directory_location + '/focal_e_y.npy')
                #     focal_h_data = {}
                #     focal_h_data['x'] = np.load(projects_directory_location + '/focal_h_x.npy')
                #     focal_h_data['y'] = np.load(projects_directory_location + '/focal_h_y.npy')
            
                print("Beginning Step 3: Adjoint Optimization Jobs")
                sys.stdout.flush()
            
                # Get the device / current permittivity's dimensions in terms of (geometry) voxels       NOTE: UNUSED
                cur_permittivity_shape = cur_permittivity.shape
                # Get the device's dimensions in terms of (MESH) voxels
                field_shape = [device_voxels_simulation_mesh_lateral, device_voxels_simulation_mesh_lateral, device_voxels_simulation_mesh_vertical]

                # Initialize arrays to store the gradients of each polarization, each the shape of the voxel mesh
                xy_polarized_gradients = [ np.zeros(field_shape, dtype=np.complex), np.zeros(field_shape, dtype=np.complex) ]

                for adj_src_idx in range(0, num_adjoint_sources):
                    polarizations = polarizations_focal_plane_map[adj_src_idx]
                    spectral_indices = spectral_focal_plane_map[adj_src_idx]

                    gradient_performance_weight = 0
                    if not weight_by_individual_wavelength:
                        gradient_performance_weight = performance_weighting[adj_src_idx]

                        if do_rejection:
                            if weight_by_quadrant_transmission:
                                gradient_performance_weight *= softplus_prime( transmission_per_quadrant[ adj_src_idx ] )
                            else:
                                gradient_performance_weight *= softplus_prime( figure_of_merit_per_focal_spot[ adj_src_idx ] )

                    lookup_dispersive_range_idx = adjoint_src_to_dispersive_range_map[ adj_src_idx ]

                    # Store the design E-field for the adjoint simulations
                    adjoint_e_fields = []
                    starttime = time.time()
                    for xy_idx in range(0, 2):
                        #
                        # This is ok because we are only going to be using the spectral idx corresponding to this range in the summation below
                        #
                        
                        actual_filepath = convert_root_folder(job_names[ ( 'adjoint', adj_src_idx, xy_idx, lookup_dispersive_range_idx ) ], actual_directory)
                        #fdtd_hook.load( job_names[ ( 'forward', xy_idx, dispersive_range_idx ) ] )
                        print(f'Loading: {isolate_filename(actual_filepath)}')
                        fdtd_hook.load(actual_filepath)
                        #fdtd_hook.load( job_names[ ( 'adjoint', adj_src_idx, xy_idx, lookup_dispersive_range_idx ) ] )
                        #print('Loading .fsp file: '+job_names[('adjoint', adj_src_idx, xy_idx, lookup_dispersive_range_idx)])
                        #sys.stdout.flush()

                        adjoint_e_fields.append(
                            get_efield(design_efield_monitor['name']))
                    endtime=time.time()
                    print(f'Loading data took: {(endtime-starttime)/60} min.')
                    
                    for pol_idx in range(0, len(polarizations)):
                        pol_name = polarizations[pol_idx]
                        get_focal_e_data = focal_e_data[pol_name]                           # Get E-field at adjoint source for each polarization
                        get_focal_h_data = focal_h_data[pol_name]                           # Get H-field at adjoint source for each polarization
                        get_mode_e_fields = mode_e_fields[pol_name]                         # Get E-field at device for each polarization, given adjoint source
                        get_mode_h_fields = mode_h_fields[pol_name]                         # Get H-field at device for each polarization, given adjoint source
                        pol_name_to_idx = polarization_name_to_idx[pol_name]

                        for xy_idx in range(0, 2):
                            #* Conjugate of E_{old}(x_0) -field at the adjoint source of interest, with direction along the polarization
                            # This is going to be the amplitude of the dipole-adjoint source driven at the focal plane
                            
                            # Create a normal z-vector
                            # normal_vector = [0,0,1]     #NOTE: This is fine, this works in np.cross()
                            #normal_vector = np.ones(np.shape(get_focal_h_data[adj_src_idx]))
                            normal_vector = np.zeros(np.shape(get_focal_h_data[adj_src_idx]))
                            normal_vector[2] = np.ones(np.shape(normal_vector[2]))
                            mu_0 = 1.25663706 * 1e-6
                            
                            adjoint_integral_1 = poynting_flux(get_focal_e_data[adj_src_idx][:,:,:,:, spectral_indices[0] : spectral_indices[1] : 1],
                                                               np.conj(get_mode_h_fields[adj_src_idx][:,:,:,:, spectral_indices[0] : spectral_indices[1] : 1]),
                                                               (0.5*device_size_lateral_um)**2)
                            adjoint_integral_2 = poynting_flux(np.conj(get_mode_e_fields[adj_src_idx][:,:,:,:, spectral_indices[0] : spectral_indices[1] : 1]),
                                                               get_focal_h_data[adj_src_idx][:,:,:,:, spectral_indices[0] : spectral_indices[1] : 1],
                                                               (0.5*device_size_lateral_um)**2)
                            adjoint_integral_3 = poynting_flux(get_mode_e_fields[adj_src_idx][:,:,:,:, spectral_indices[0] : spectral_indices[1] : 1],
                                                               np.conj(get_mode_h_fields[adj_src_idx][:,:,:,:, spectral_indices[0] : spectral_indices[1] : 1]),
                                                               (0.5*device_size_lateral_um)**2, real_e_h=True)
                            # TODO: 
                            adjoint_A_factor = 1/4 * np.conj(adjoint_integral_1 + adjoint_integral_2) / (adjoint_integral_3)
                            print("Calculated adjoint A factor.")
                            sys.stdout.flush()
                                        
                            
                            source_weight =  np.squeeze(adjoint_A_factor) 
                                # * (
                                # poynting_flux(np.conj(get_focal_h_data[adj_src_idx][:, :,:,:, spectral_indices[0] : spectral_indices[1] : 1]),
                                #               normal_vector[:, :,:,:, spectral_indices[0] : spectral_indices[1] : 1], 
                                #               (0.5*device_size_lateral_um)**2, dot_dS = False, pauseDebug=False) - 
                                # poynting_flux(normal_vector[:, :,:,:, spectral_indices[0] : spectral_indices[1] : 1],
                                #               np.conj(get_focal_e_data[adj_src_idx][:, :,:,:, spectral_indices[0] : spectral_indices[1] : 1]),
                                #               (0.5*device_size_lateral_um)**2, dot_dS = False, pauseDebug=False) / mu_0
                                # ))


                            #
                            # todo: do you need to change this if you weight by transmission? the figure of merit still has this weight, the transmission is just
                            # guiding additional weights on top of this
                            #
                            max_intensity_weighting = max_intensity_by_wavelength[spectral_indices[0] : spectral_indices[1] : 1]
                            total_weighting = max_intensity_weighting / weight_focal_plane_map[adj_src_idx]

                            #
                            # We need to properly account here for the current real and imaginary index
                            # because they both contribute in the end to the real part of the gradient ΔFoM/Δε_r
                            # in Eq. 5 of Lalau-Keraly paper https://doi.org/10.1364/OE.21.021693
                            #
                            dispersive_max_permittivity = dispersion_model.average_permittivity(dispersive_ranges_um[lookup_dispersive_range_idx])
                            delta_real_permittivity = np.real(dispersive_max_permittivity - min_device_permittivity )
                            delta_imag_permittivity = np.imag(dispersive_max_permittivity - min_device_permittivity )

                            for spectral_idx in range(0, len( source_weight ) ):
                                # TODO: Performance weighting is likely the thing that tunes the FoM based on the performance thus far in the optimization
                                if weight_by_individual_wavelength:
                                    gradient_performance_weight = performance_weighting[ spectral_indices[0] + spectral_idx ]

                                    if do_rejection:
                                        if weight_by_quadrant_transmission:
                                            gradient_performance_weight *= softplus_prime( transmission_by_wavelength[ spectral_idx ] )
                                        else:
                                            gradient_performance_weight *= softplus_prime( intensity_fom_by_wavelength[ spectral_idx ] )
                                        
                                        # gradient_performance_weight *= softplus_prime( intensity_fom_by_wavelength[ spectral_idx ] )

                                if do_rejection:
                                    gradient_performance_weight *= spectral_focal_plane_map_directional_weights[ adj_src_idx ][ spectral_idx ]

                                # See Eqs. (5), (6) of Lalau-Keraly et al: https://doi.org/10.1364/OE.21.021693
                                print(f'Spectral idx is: {spectral_idx}')
                                sys.stdout.flush()
                                gradient_component = np.sum(
                                    (source_weight[spectral_idx] *                                                      # Amplitude of electric dipole at x_0
                                     gradient_performance_weight / total_weighting[spectral_idx]) *                     # Performance weighting, Manual override weighting, and intensity normalization factors
                                    adjoint_e_fields[xy_idx][:, :, :, :, spectral_indices[0] + spectral_idx] *          # E-field induced at x from electric dipole at x_0
                                    forward_e_fields[pol_name][:, :, :, :, spectral_indices[0] + spectral_idx],         # E_old(x)
                                    axis=0)                                                                             # Dot product

                                # Permittivity factor in amplitude of electric dipole at x_0
                                # Explanation: For two complex numbers, Re(z1*z2) = Re(z1)*Re(z2) + [-Im(z1)]*Im(z2)
                                real_part_gradient = np.real( gradient_component )
                                imag_part_gradient = -np.imag( gradient_component )
                                get_grad_density = delta_real_permittivity * real_part_gradient + delta_imag_permittivity * imag_part_gradient

                                xy_polarized_gradients[pol_name_to_idx] += get_grad_density

                print("Completed Step 3: Retrieved Adjoint Optimization results and calculated gradient.")
                sys.stdout.flush()
                
                # backup_all_vars()
                
                
            #
            # Step 4: Step the design variable.
            #

            # if start_from_step == 0 or start_from_step == 4:
            #     if start_from_step != 0:
            #         load_backup_vars()
            
                print("Beginning Step 4: Stepping Design Variable.")
                sys.stdout.flush()


                # Get the full device gradient by summing the x,y polarization components 
                device_gradient = 2 * ( xy_polarized_gradients[0] + xy_polarized_gradients[1] )
                # Project / interpolate the device_gradient, the values of which we have at each (mesh) voxel point, and obtain it at each (geometry) voxel point
                device_gradient_interpolated = interpolate.interpn( ( gradient_region_x, gradient_region_y, gradient_region_z ), device_gradient, bayer_filter_region, method='linear' )


                if enforce_xy_gradient_symmetry:
                    device_gradient_interpolated = 0.5 * ( device_gradient_interpolated + np.swapaxes( device_gradient_interpolated, 0, 1 ) )

                device_gradient = device_gradient_interpolated.copy()
                
                #
                #* Begin scaling of step size so that the design change stays within epoch_design_change limits in parameters.py
                #
                
                # Backpropagate the design_gradient through the various filters so it is applied to the (0-1) scaled permittivity
                design_gradient = bayer_filter.backpropagate( device_gradient )

                # Control the maximum that the design is allowed to change per epoch. This will increase according to the limits set, for each iteration
                max_change_design = epoch_start_design_change_max
                min_change_design = epoch_start_design_change_min

                if num_iterations_per_epoch > 1:
                    max_change_design = (
                        epoch_end_design_change_max +
                        (num_iterations_per_epoch - 1 - iteration) * (epoch_range_design_change_max / (num_iterations_per_epoch - 1))
                    )

                    min_change_design = (
                        epoch_end_design_change_min +
                        (num_iterations_per_epoch - 1 - iteration) * (epoch_range_design_change_min / (num_iterations_per_epoch - 1))
                    )

                cur_design_variable = bayer_filter.get_design_variable()
                if use_fixed_step_size:
                    step_size = fixed_step_size
                else:
                    step_size = step_size_start

                    check_last = False
                    last = 0

                    while True:
                        # Gets proposed design variable according to Eq. S2, OPTICA Paper Supplement: https://doi.org/10.1364/OPTICA.384228
                        # Divides step size by 2 until the difference in the design variable is within the ranges set.
                        
                        proposed_design_variable = cur_design_variable + step_size * design_gradient
                        # proposed_design_variable = cur_design_variable - step_size * design_gradient
                        proposed_design_variable = np.maximum(                                  # Makes sure that it's between 0 and 1
                                                    np.minimum(
                                                        proposed_design_variable,
                                                        1.0),
                                                    0.0)

                        difference = np.abs(proposed_design_variable - cur_design_variable)
                        max_difference = np.max(difference)

                        # TODO: Try and understand this control loop
                        if (max_difference <= max_change_design) and (max_difference >= min_change_design):
                            break
                        elif (max_difference <= max_change_design):
                            step_size *= 2
                            if (last ^ 1) and check_last:
                                break
                            check_last = True
                            last = 1
                        else:
                            step_size /= 2
                            if (last ^ 0) and check_last:
                                break
                            check_last = True
                            last = 0

                    step_size_start = step_size

                #
                #* Now that we have obtained the right step size, and the (backpropagated) design gradient,
                #* we will finalize everything and save it out.
                #
                last_design_variable = cur_design_variable.copy()
                # Feed proposed design variable into bayer filter, run it through filters and update permittivity
                # -device gradient because the step() function has a - sign instead of a +
                bayer_filter.step(-device_gradient, step_size)
                # bayer_filter.step(+device_gradient, step_size)
                cur_design_variable = bayer_filter.get_design_variable()
                cur_design = bayer_filter.get_permittivity()

                average_design_variable_change = np.mean( np.abs(cur_design_variable - last_design_variable) )
                max_design_variable_change = np.max( np.abs(cur_design_variable - last_design_variable) )

                step_size_evolution[epoch][iteration] = step_size
                average_design_variable_change_evolution[epoch][iteration] = average_design_variable_change
                max_design_variable_change_evolution[epoch][iteration] = max_design_variable_change

                np.save(projects_directory_location + '/device_gradient.npy', device_gradient)
                np.save(projects_directory_location + '/design_gradient.npy', design_gradient)
                np.save(projects_directory_location + "/step_size_evolution.npy", step_size_evolution)
                np.save(projects_directory_location + "/average_design_change_evolution.npy", average_design_variable_change_evolution)
                np.save(projects_directory_location + "/max_design_change_evolution.npy", max_design_variable_change_evolution)
                np.save(projects_directory_location + "/cur_design_variable.npy", cur_design_variable)
                np.save(projects_directory_location + "/cur_design.npy", cur_design)
                np.save(projects_directory_location + "/cur_design_variable_epoch_" + str( epoch ) + ".npy", cur_design_variable)
                # np.save(projects_directory_location_init +
                    # "/cur_design_variable.npy", cur_design_variable)
                
                print("Completed Step 4: Saved out cur_design_variable.npy and other files.")
                sys.stdout.flush()
                
                backup_all_vars()
                





print('Reached end of code. Operation completed.')