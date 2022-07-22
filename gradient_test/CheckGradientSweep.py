import os
import shutil
import sys
import shelve
from threading import activeCount

# Add filepath to default path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from CheckGradientParameters import *
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
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from mpl_toolkits.axes_grid1 import make_axes_locatable


import time

import queue
import subprocess
import platform
import re
import gc

print("All modules loaded.")
sys.stdout.flush()

def backup_var(variable_name, savefile_name=None):
    '''Performs the equivalent of MATLAB's save(), albeit with some workarounds. Lumapi SimObjects cannot be saved this way.'''
    
    if savefile_name is None:
        savefile_name = shelf_fn
    
    global my_shelf
    my_shelf = shelve.open(savefile_name,'n')
    if variable_name != "my_shelf":
        try:
            my_shelf[variable_name] = globals()[variable_name]
            #print(key)
        except Exception as ex:
            # __builtins__, my_shelf, and imported modules can not be shelved.
            # print('ERROR shelving: {0}'.format(key))
            # template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            # message = template.format(type(ex).__name__, ex.args)
            # print(message)
            pass
    my_shelf.close()
    
    print("Successfully backed up variable: " + variable_name)
    sys.stdout.flush()

def backup_all_vars(savefile_name=None):
    '''Performs the equivalent of MATLAB's save(), albeit with some workarounds. Lumapi SimObjects cannot be saved this way.'''
    
    if savefile_name is None:
        savefile_name = shelf_fn
    
    global my_shelf
    my_shelf = shelve.open(savefile_name,'n')
    for key in sorted(globals()):
        if key != "my_shelf":
            try:
                my_shelf[key] = globals()[key]
                #print(key)
            except Exception as ex:
                # __builtins__, my_shelf, and imported modules can not be shelved.
                # print('ERROR shelving: {0}'.format(key))
                # template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                # message = template.format(type(ex).__name__, ex.args)
                # print(message)
                pass
    my_shelf.close()
    
    print("Backup complete.")
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
                        'monitors',
                        'plots']
    
    if loadfile_name is None:
        loadfile_name = shelf_fn
    
    global my_shelf
    my_shelf = shelve.open(shelf_fn, flag='r')
    for key in my_shelf:
        if key not in do_not_overwrite:
            try:
                globals()[key]=my_shelf[key]
            except Exception as ex:
                #print('ERROR retrieving shelf: {0}'.format(key))
                #print("An exception of type {0} occurred. Arguments:\n{1!r}".format(type(ex).__name__, ex.args))
                pass
    my_shelf.close()
    
    # Restore lumapi SimObjects as dictionaries
    for key,val in fdtd_objects.items():
        globals()[key]=val
    
    print("Successfully loaded backup.")
    sys.stdout.flush()

def isolate_filename(address):
    return address.rsplit('/',1)[1]

def convert_root_folder(address, root_folder_new):
    new_address = root_folder_new + '/' + isolate_filename(address)
    return new_address

def create_dict_nx(dict_name):
    '''Checks if dictionary already exists in globals(). If not, creates it as empty.'''
    if dict_name in globals():
            print(dict_name +' already exists in globals().')
    else:
        globals()[dict_name] = {}

def recursive_get(d, *keys):
    '''Accesses multi-level dictionaries of arbitrary depth.'''
    return functools.reduce(lambda c, k: c.get(k, {}), keys, d)
        
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
			# Split at hypen. Get first and last number. Create string in range
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
python_src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))

if not os.path.isdir(projects_directory_location):
	os.mkdir(projects_directory_location)

log_file = open( projects_directory_location + "/log.txt", 'w' )
log_file.write( "---Log for Lumerical Processing Sweep---\n" )
log_file.close()

#
# Create FDTD hook
#
fdtd_hook = lumapi.FDTD()


# Load saved session from adjoint optimization
fdtd_hook.newproject()
fdtd_hook.load(python_src_directory + "/" + device_filename)
fdtd_hook.save(projects_directory_location + "/" + device_filename)
# shutil.copy2(python_src_directory + "/SonyBayerFilterParameters.py",
#              projects_directory_location + "/SonyBayerFilterParameters.py")
# shutil.copy2(python_src_directory + "/SonyBayerFilter.py",
#              projects_directory_location + "/SonyBayerFilter.py")
# shutil.copy2(python_src_directory + "/SonyBayerFilterOptimization.py",
#              projects_directory_location + "/SonyBayerFilterOptimization.py")


#! Step 0 Functions

def disable_all_sources():
    '''Disable all sources in the simulation, so that we can selectively turn single sources on at a time'''
    lumapi.evalScript(fdtd_hook.handle, 'switchtolayout;')
    
    # Got to hardcode this part unless we start loading the save_states from optimization runs
    if 'forward_sources' not in globals():
        global forward_sources
        forward_sources = [{'name':'forward_src_x'}, {'name':'forward_src_y'}]
    if 'adjoint_sources' not in globals():
        global adjoint_sources
        adjoint_sources = [[{'name':'adj_src_0x'}, {'name':'adj_src_0y'}],
                        [{'name':'adj_src_1x'}, {'name':'adj_src_1y'}],
                        [{'name':'adj_src_2x'}, {'name':'adj_src_2y'}],
                        [{'name':'adj_src_3x'}, {'name':'adj_src_3y'}]]

    for xy_idx in range(0, 2):
        fdtd_hook.select(forward_sources[xy_idx]['name'])
        fdtd_hook.set('enabled', 0)
        
    for adj_src_idx in range(0, num_adjoint_sources):
        for xy_idx in range(0, 2):
            fdtd_hook.select(adjoint_sources[adj_src_idx][xy_idx]['name'])
            fdtd_hook.set('enabled', 0)
 
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
    # Step 0: Set up desired monitors from which E-fields, Poynting fields, and transmission data will be drawn.
    # Also any other necessary editing of the Lumerical environment and objects.
    #

    print("Beginning Step 0: Monitor Setup")
    sys.stdout.flush()

    #
    # Create dictionary to duplicate FDTD SimObjects as dictionaries for saving out
    #
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
    # Set up the FDTD monitors that weren't present during adjoint optimization
    # We are using dict-like access: see https://support.lumerical.com/hc/en-us/articles/360041873053-Session-management-Python-API
    #

    
    # Grab positions and dimensions from the .fsp itself. Replaces values from LumProcParameters.py
    #! No checks are done to see if there are discrepancies.
    device_size_lateral_um = fdtd_hook.getnamed('design_import','x span') / 1e-6
    device_vertical_maximum_um = fdtd_hook.getnamed('design_import','z max') / 1e-6
    device_vertical_minimum_um = fdtd_hook.getnamed('design_import','z min') / 1e-6
    # fdtd_region_size_lateral_um = fdtd_hook.getnamed('FDTD','x span') / 1e-6
    fdtd_region_maximum_vertical_um = fdtd_hook.getnamed('FDTD','z max') / 1e-6
    fdtd_region_minimum_vertical_um = fdtd_hook.getnamed('FDTD','z min') / 1e-6
    fdtd_region_size_vertical_um = fdtd_region_maximum_vertical_um + abs(fdtd_region_minimum_vertical_um)
    monitorbox_size_lateral_um = device_size_lateral_um + 2 * sidewall_thickness_um + (mesh_spacing_um)*5
    monitorbox_vertical_maximum_um = device_vertical_maximum_um + (mesh_spacing_um*5)
    monitorbox_vertical_minimum_um = device_vertical_minimum_um - (mesh_spacing_um*5)
    adjoint_vertical_um = fdtd_hook.getnamed('transmission_focal_monitor_', 'z') / 1e-6
    
    # These monitors are already created and need no editing, but need to be initialized so that code down below will run smoothly.
    for tm_idx in range(0, num_adjoint_sources):
        create_dict_nx(f'transmission_monitor_{tm_idx}')
        globals()[f'transmission_monitor_{tm_idx}']['frequency points'] = num_design_frequency_points
        #fdtd_hook.setnamed(f'transmission_monitor_{tm_idx}', 'frequency points', num_design_frequency_points)
    create_dict_nx('transmission_focal_monitor_')
    globals()['transmission_focal_monitor_']['frequency points'] = num_design_frequency_points
    #fdtd_hook.setnamed(f'transmission_focal_monitor_', 'frequency points', num_design_frequency_points)
        
    # Monitor right below source to measure overall transmission / reflection
    src_transmission_monitor = {}
    src_transmission_monitor['x'] = 0 * 1e-6
    src_transmission_monitor['x span'] = (monitorbox_size_lateral_um*1.1) * 1e-6
    src_transmission_monitor['y'] = 0 * 1e-6
    src_transmission_monitor['y span'] = (monitorbox_size_lateral_um*1.1) * 1e-6
    src_transmission_monitor['z'] = fdtd_hook.getnamed('forward_src_x','z') - (mesh_spacing_um*5)*1e-6
    src_transmission_monitor['override global monitor settings'] = 1
    src_transmission_monitor['use wavelength spacing'] = 1
    src_transmission_monitor['use source limits'] = 1
    src_transmission_monitor['frequency points'] = num_design_frequency_points
    fdtd_objects['src_transmission_monitor'] = src_transmission_monitor
    
    # Monitor at top of device across FDTD to measure source power that misses the device
    src_spill_monitor = {}
    src_spill_monitor['x'] = 0 * 1e-6
    src_spill_monitor['x span'] = fdtd_region_size_lateral_um * 1e-6
    src_spill_monitor['y'] = 0 * 1e-6
    src_spill_monitor['y span'] = fdtd_region_size_lateral_um * 1e-6
    src_spill_monitor['z'] = monitorbox_vertical_maximum_um * 1e-6
    src_spill_monitor['override global monitor settings'] = 1
    src_spill_monitor['use wavelength spacing'] = 1
    src_spill_monitor['use source limits'] = 1
    src_spill_monitor['frequency points'] = num_design_frequency_points
    fdtd_objects['src_spill_monitor'] = src_spill_monitor
    
    # Monitor right above device to measure input power
    incident_aperture_monitor = {}
    incident_aperture_monitor['x'] = 0 * 1e-6
    incident_aperture_monitor['x span'] = monitorbox_size_lateral_um * 1e-6
    incident_aperture_monitor['y'] = 0 * 1e-6
    incident_aperture_monitor['y span'] = monitorbox_size_lateral_um * 1e-6
    incident_aperture_monitor['z'] = monitorbox_vertical_maximum_um * 1e-6
    incident_aperture_monitor['override global monitor settings'] = 1
    incident_aperture_monitor['use wavelength spacing'] = 1
    incident_aperture_monitor['use source limits'] = 1
    incident_aperture_monitor['frequency points'] = num_design_frequency_points
    fdtd_objects['incident_aperture_monitor'] = incident_aperture_monitor
    
    # Monitor right below device to measure exit aperture power
    exit_aperture_monitor = {}
    exit_aperture_monitor['x'] = 0 * 1e-6
    exit_aperture_monitor['x span'] = monitorbox_size_lateral_um * 1e-6
    exit_aperture_monitor['y'] = 0 * 1e-6
    exit_aperture_monitor['y span'] = monitorbox_size_lateral_um * 1e-6
    exit_aperture_monitor['z'] = monitorbox_vertical_minimum_um * 1e-6
    exit_aperture_monitor['override global monitor settings'] = 1
    exit_aperture_monitor['use wavelength spacing'] = 1
    exit_aperture_monitor['use source limits'] = 1
    exit_aperture_monitor['frequency points'] = num_design_frequency_points
    fdtd_objects['exit_aperture_monitor'] = exit_aperture_monitor
    
    # Sidewall monitors closing off the monitor box around the splitter cube device, to measure side scattering
    side_monitors = []
    sm_x_pos = [monitorbox_size_lateral_um/2, 0, -monitorbox_size_lateral_um/2, 0]
    sm_y_pos = [0, monitorbox_size_lateral_um/2, 0, -monitorbox_size_lateral_um/2]
    sm_xspan_pos = [0, monitorbox_size_lateral_um, 0, monitorbox_size_lateral_um]
    sm_yspan_pos = [monitorbox_size_lateral_um, 0, monitorbox_size_lateral_um, 0]
    
    for sm_idx in range(0, 4):      # We are not using num_sidewalls because these monitors MUST be created
        side_monitor = {}
        #side_monitor['x'] = sm_x_pos[sm_idx] * 1e-6
        side_monitor['x max'] =  (sm_x_pos[sm_idx] + sm_xspan_pos[sm_idx]/2) * 1e-6   # must bypass setting 'x span' because this property is invalid if the monitor is X-normal
        side_monitor['x min'] =  (sm_x_pos[sm_idx] - sm_xspan_pos[sm_idx]/2) * 1e-6
        #side_monitor['y'] = sm_y_pos[sm_idx] * 1e-6
        side_monitor['y max'] =  (sm_y_pos[sm_idx] + sm_yspan_pos[sm_idx]/2) * 1e-6   # must bypass setting 'y span' because this property is invalid if the monitor is Y-normal
        side_monitor['y min'] =  (sm_y_pos[sm_idx] - sm_yspan_pos[sm_idx]/2) * 1e-6
        side_monitor['z max'] = monitorbox_vertical_maximum_um * 1e-6
        side_monitor['z min'] = monitorbox_vertical_minimum_um * 1e-6
        side_monitor['override global monitor settings'] = 1
        side_monitor['use wavelength spacing'] = 1
        side_monitor['use source limits'] = 1
        side_monitor['frequency points'] = num_design_frequency_points
        
        side_monitors.append(side_monitor)
        globals()['side_monitor_'+str(sm_idx)] = side_monitor
        fdtd_objects['side_monitor_'+str(sm_idx)] = side_monitor
    
    scatter_plane_monitor = {}
    scatter_plane_monitor['x'] = 0 * 1e-6\
    # Need to consider: if the scatter plane monitor only includes one quadrant from the adjacent corner focal-plane-quadrants, 
    # the average power captured by the scatter plane monitor is only 83% of the power coming from the exit aperture.
    # It will also miss anything that falls outside of the boundary which is very probable.
    # At the same time the time to run FDTD sim will also go up by a factor squared.
    scatter_plane_monitor['x span'] = ((fdtd_hook.getnamed('transmission_focal_monitor_','x span') /2 * 6) / 1e-6) * 1e-6
    scatter_plane_monitor['y'] = 0 * 1e-6
    scatter_plane_monitor['y span'] = ((fdtd_hook.getnamed('transmission_focal_monitor_','y span') /2 * 6) / 1e-6) * 1e-6
    scatter_plane_monitor['z'] = adjoint_vertical_um * 1e-6
    scatter_plane_monitor['override global monitor settings'] = 1
    scatter_plane_monitor['use wavelength spacing'] = 1
    scatter_plane_monitor['use source limits'] = 1
    scatter_plane_monitor['frequency points'] = num_design_frequency_points
    fdtd_objects['scatter_plane_monitor'] = scatter_plane_monitor
    
    # Side monitors enclosing the space between the device's exit aperture and the focal plane.
    # Purpose is to capture the sum power of oblique scattering.
    vertical_scatter_monitors = []
    vsm_x_pos = [device_size_lateral_um/2, 0, -device_size_lateral_um/2, 0]
    vsm_y_pos = [0, device_size_lateral_um/2, 0, -device_size_lateral_um/2]
    vsm_xspan_pos = [0, device_size_lateral_um, 0, device_size_lateral_um]
    vsm_yspan_pos = [device_size_lateral_um, 0, device_size_lateral_um, 0]
    
    for vsm_idx in range(0, 4):      # We are not using num_sidewalls because these monitors MUST be created
        vertical_scatter_monitor = {}
        #vertical_scatter_monitor['x'] = vsm_x_pos[vsm_idx] * 1e-6
        vertical_scatter_monitor['x max'] =  (vsm_x_pos[vsm_idx] + vsm_xspan_pos[vsm_idx]/2) * 1e-6   # must bypass setting 'x span' because this property is invalid if the monitor is X-normal
        vertical_scatter_monitor['x min'] =  (vsm_x_pos[vsm_idx] - vsm_xspan_pos[vsm_idx]/2) * 1e-6
        #vertical_scatter_monitor['y'] = vsm_y_pos[vsm_idx] * 1e-6
        vertical_scatter_monitor['y max'] =  (vsm_y_pos[vsm_idx] + vsm_yspan_pos[vsm_idx]/2) * 1e-6   # must bypass setting 'y span' because this property is invalid if the monitor is Y-normal
        vertical_scatter_monitor['y min'] =  (vsm_y_pos[vsm_idx] - vsm_yspan_pos[vsm_idx]/2) * 1e-6
        vertical_scatter_monitor['z max'] = device_vertical_minimum_um * 1e-6
        vertical_scatter_monitor['z min'] = adjoint_vertical_um * 1e-6
        vertical_scatter_monitor['override global monitor settings'] = 1
        vertical_scatter_monitor['use wavelength spacing'] = 1
        vertical_scatter_monitor['use source limits'] = 1
        vertical_scatter_monitor['frequency points'] = num_design_frequency_points
        
        vertical_scatter_monitors.append(vertical_scatter_monitor)
        globals()['vertical_scatter_monitor_'+str(vsm_idx)] = vertical_scatter_monitor
        fdtd_objects['vertical_scatter_monitor_'+str(vsm_idx)] = vertical_scatter_monitor
    
    # Device cross-section monitors that take the field inside the device, and extend all the way down to the focal plane. We have 6 slices, 2 sets of 3 for both x and y.
    # Each direction-set has slices at x(y) = -device_size_lateral_um/4, 0, +device_size_lateral_um/4, i.e. intersecting where the focal spot is supposed to be.
    # We also take care to output plots for both E_norm and Re(E), the latter of which captures the wave-like behaviour much more accurately.
    device_xsection_monitors = []
    dxm_pos = [-device_size_lateral_um/4, 0, device_size_lateral_um/4]
    for dev_cross_idx in range(0,6):
        device_cross_monitor = {}
        device_cross_monitor['z max'] = device_vertical_maximum_um * 1e-6
        device_cross_monitor['z min'] = adjoint_vertical_um * 1e-6
        
        if dev_cross_idx < 3:
            device_cross_monitor['y'] = dxm_pos[dev_cross_idx] * 1e-6
            device_cross_monitor['x max'] = (device_size_lateral_um / 2) * 1e-6
            device_cross_monitor['x min'] = -(device_size_lateral_um / 2) * 1e-6
        else:
            device_cross_monitor['x'] = dxm_pos[dev_cross_idx-3] * 1e-6
            device_cross_monitor['y max'] = (device_size_lateral_um / 2) * 1e-6
            device_cross_monitor['y min'] = -(device_size_lateral_um / 2) * 1e-6
        
        device_cross_monitor['override global monitor settings'] = 1
        device_cross_monitor['use wavelength spacing'] = 1
        device_cross_monitor['use source limits'] = 1
        device_cross_monitor['frequency points'] = num_design_frequency_points
        
        device_xsection_monitors.append(device_cross_monitor)
        globals()['device_xsection_monitor_'+str(dev_cross_idx)] = device_cross_monitor
        fdtd_objects['device_xsection_monitor_'+str(dev_cross_idx)] = device_cross_monitor
    
    design_efield_monitor = {}
    
    transmission_monitor_focal = {}
    # transmission_monitor_focal = fdtd_hook.addpower()
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
    fdtd_hook.setnamed(transmission_monitor_focal['name'], 'frequency points', num_design_frequency_points)
    #fdtd_objects['transmission_monitor_focal'] = fdtd_simobject_to_dict(transmission_monitor_focal)
        
    # Create monitors and apply all of the above settings to each monitor
    for monitor in monitors:
        if monitor['enabled']:
            try:
                objExists = fdtd_hook.getnamed(monitor['name'],'name')
            except Exception as ex:
                # template = "An exception of type {0} occurred. Arguments:\n{1!r}\n"
                # message = template.format(type(ex).__name__, ex.args)
                # print(message+'FDTD Object does not exist.')
                fdtd_hook.addpower(name=monitor['name'])
            fdtd_hook.setnamed(monitor['name'], 'monitor type', monitor['monitor type'])
            fdtd_hook.setnamed(monitor['name'], 'enabled', monitor['enabled'])
            for key, val in globals()[monitor['name']].items():
                # print(f'Monitor: {monitor["name"]}, Key: {key}, and Value: {val}')
                fdtd_hook.setnamed(monitor['name'], key, val)


    # Implement finer sidewall mesh
    # Sidewalls may not match index with sidewall monitors...
    sidewall_meshes = []
    sidewall_mesh_override_x_mesh = [1,0,1,0]
    
    for sidewall_idx in range(0, num_sidewalls):
        sidewall_mesh = {}
        sidewall_mesh['name'] = 'mesh_sidewall_' + str(sidewall_idx)
        try:
            sidewall_mesh['override x mesh'] = fdtd_hook.getnamed(sidewall_mesh['name'], 'override x mesh')
        except Exception as err:
            sidewall_mesh['override x mesh'] = sidewall_mesh_override_x_mesh[sidewall_idx]
        if sidewall_mesh['override x mesh']:
            fdtd_hook.setnamed(sidewall_mesh['name'], 'dx', fdtd_hook.getnamed('device_mesh','dx') / 1.5)
        else:
            fdtd_hook.setnamed(sidewall_mesh['name'], 'dy', fdtd_hook.getnamed('device_mesh','dy') / 1.5)
        
        sidewall_meshes.append(sidewall_mesh)
        globals()[sidewall_mesh['name']] = sidewall_mesh
        fdtd_objects[sidewall_mesh['name']] = sidewall_mesh
    
    for sidewall_mesh in sidewall_meshes:
        try:
            objExists = fdtd_hook.getnamed(sidewall_mesh['name'],'name')
            
            for key, val in globals()[sidewall_mesh['name']].items():
                # print(f'sidewall_mesh: {sidewall_mesh["name"]}, Key: {key}, and Value: {val}')
                fdtd_hook.setnamed(sidewall_mesh['name'], key, val)
                
        except Exception as ex:
            # # template = "An exception of type {0} occurred. Arguments:\n{1!r}\n"
            # # message = template.format(type(ex).__name__, ex.args)
            # # print(message+'FDTD Object does not exist.')
            fdtd_hook.addmesh(name=sidewall_mesh['name'])
            
        if num_sidewalls == 0:
            fdtd_hook.setnamed(sidewall_mesh['name'], 'enabled', 0)
        else:
            fdtd_hook.setnamed(sidewall_mesh['name'], 'enabled', 1)
    
    # Change lateral region size of FDTD region and device mesh
    
    create_dict_nx('FDTD')
    FDTD['x span'] = fdtd_region_size_lateral_um * 1e-6
    FDTD['y span'] = fdtd_region_size_lateral_um * 1e-6
    for key, val in globals()['FDTD'].items():
        fdtd_hook.setnamed('FDTD', key, val)
        
    # create_dict_nx('device_mesh')
    # device_mesh['x span'] = fdtd_region_size_lateral_um * 1e-6
    # device_mesh['y span'] = fdtd_region_size_lateral_um * 1e-6
    # for key, val in globals()['device_mesh'].items():
    #     fdtd_hook.setnamed('device_mesh', key, val)

    xy_pol_rotations = [0, 90]
    xy_names = ['x', 'y']
    
    if change_wavelength_range:
        create_dict_nx('forward_src_x')
        forward_src_x['wavelength start'] = lambda_min_um * 1e-6
        forward_src_x['wavelength stop'] = lambda_max_um * 1e-6
        forward_src_x['beam parameters'] = 'Waist size and position'
        forward_src_x['waist radius w0'] = 3e8/np.mean(3e8/np.array([lambda_min_um, lambda_max_um])) / (np.pi*background_index*np.radians(src_div_angle)) * 1e-6
        forward_src_x['distance from waist'] = -(src_maximum_vertical_um - device_size_vertical_um) * 1e-6
        for key, val in globals()['forward_src_x'].items():
            fdtd_hook.setnamed('forward_src_x', key, val)
        
        create_dict_nx('forward_src_y')
        forward_src_y['wavelength start'] = lambda_min_um * 1e-6
        forward_src_y['wavelength stop'] = lambda_max_um * 1e-6
        forward_src_y['beam parameters'] = 'Waist size and position'
        forward_src_y['waist radius w0'] = 3e8/np.mean(3e8/np.array([lambda_min_um, lambda_max_um])) / (np.pi*background_index*np.radians(src_div_angle)) * 1e-6
        forward_src_y['distance from waist'] = -(src_maximum_vertical_um - device_size_vertical_um) * 1e-6
        for key, val in globals()['forward_src_y'].items():
            fdtd_hook.setnamed('forward_src_y', key, val)
            
        #
        # Place Gaussian adjoint sources at the focal plane that can ring in both
        # x-axis and y-axis
        #
        adjoint_sources = []

        for adj_src_idx in range(0, num_adjoint_sources):
            adjoint_sources.append([])
            for xy_idx in range(0, 2):
                
                # Clear out previous adjoint source (dipole)
                fdtd_hook.select('adj_src_' + str(adj_src_idx) + xy_names[xy_idx])
                fdtd_hook.delete()
                
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
                adj_src['x span'] = 1.5*geometry_spacing_um*40*1e-6 # 0.5 * device_size_lateral_um * 1e-6
                adj_src['y span'] = 1.5*geometry_spacing_um*40*1e-6 # 0.5 * device_size_lateral_um * 1e-6

                adj_src["frequency dependent profile"] = 1
                adj_src["number of field profile samples"] = 150
                adj_src['beam parameters'] = 'Waist size and position'
                adj_src['waist radius w0'] = adj_src_beam_rad * 1e-6
                adj_src['distance from waist'] = 0

                adjoint_sources[adj_src_idx].append(adj_src)
        fdtd_objects['adjoint_sources'] = fdtd_simobject_to_dict(adjoint_sources)
    
    # Create classes and arrays to hold device permittivity and gradient information
    
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
    
    # Disable all sources and select monitors
    disable_all_sources
    disable_objects(disable_object_list)
    fdtd_hook.save()

    print("Lumerical environment setup complete.")
    sys.stdout.flush()
    
    #
    # Save out all variables to save state file
    #
    backup_all_vars()


#*---- JOB RUNNING STEP BEGINS ----

#! Step 1 Functions

def snellRefrOffset(src_angle_incidence, object_list):
    ''' Takes a list of objects in between source and device, retrieves heights and indices,
     and calculates injection angle so as to achieve the desired angle of incidence upon entering the device.'''
    # structured such that n0 is the input interface and n_end is the last above the device
    # height_0 should be 0

    heights = [0.]
    indices = [1.]
    total_height = fdtd_hook.getnamed('design_import','z max') / 1e-6
    max_height = fdtd_hook.getnamed('forward_src_x','z') / 1e-6
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

def change_source_theta(theta_value, phi_value):
    '''Changes the angle (theta and phi) of the overhead source.'''
    fdtd_hook.select('forward_src_x')
    
    # Structure the objects list such that the last entry is the last medium above the device.
    input_theta, r_offset = snellRefrOffset(theta_value, objects_above_device)
    
    fdtd_hook.set('angle theta', input_theta) # degrees
    fdtd_hook.set('angle phi', phi_value) # degrees
    fdtd_hook.set('x', r_offset*np.cos(np.radians(phi_value)) * 1e-6)
    fdtd_hook.set('y', r_offset*np.sin(np.radians(phi_value)) * 1e-6)
    
    print(f'Theta set to {input_theta}, Phi set to {phi_value}.')

def change_source_beam_radius(param_value):
    '''For a Gaussian beam source, changes the beam radius to focus or defocus it.'''
    xy_names = ['x', 'y']
    
    for xy_idx in range(0, 2):
        fdtd_hook.select('forward_src_' + xy_names[xy_idx])
        # fdtd_hook.set('beam radius wz', 100 * 1e-6)
        # fdtd_hook.set('divergence angle', 3.42365) # degrees
        # fdtd_hook.set('beam radius wz', 15 * 1e-6)
        # # fdtd_hook.set('beam radius wz', param_value * 1e-6)
        
        fdtd_hook.set('beam parameters', 'Waist size and position')
        fdtd_hook.set('waist radius w0', param_value * 1e-6)
        fdtd_hook.set('distance from waist', 0 * 1e-6)
    
    print(f'Source beam radius set to {param_value}.')
    
# TODO
def change_fdtd_region_size(param_value):
    pass


def change_sidewall_thickness(param_value):
    '''If sidewalls exist, changes their thickness.'''
    
    monitorbox_size_lateral_um = device_size_lateral_um + 2 * param_value
    
    for idx in range(0, num_sidewalls):
        sm_name = 'sidewall_' + str(idx)
        
        try:
            objExists = fdtd_hook.getnamed(sm_name, 'name')
            fdtd_hook.select(sm_name)
        
            if idx == 0:
                fdtd_hook.setnamed(sm_name, 'x', (device_size_lateral_um + param_value)*0.5 * 1e-6)
                fdtd_hook.setnamed(sm_name, 'x span', param_value * 1e-6)
                fdtd_hook.setnamed(sm_name, 'y span', monitorbox_size_lateral_um * 1e-6)
            elif idx == 1:
                fdtd_hook.setnamed(sm_name, 'y', (device_size_lateral_um + param_value)*0.5 * 1e-6)
                fdtd_hook.setnamed(sm_name, 'y span', param_value * 1e-6)
                fdtd_hook.setnamed(sm_name, 'x span', monitorbox_size_lateral_um * 1e-6)
            if idx == 2:
                fdtd_hook.setnamed(sm_name, 'x', (-device_size_lateral_um - param_value)*0.5 * 1e-6)
                fdtd_hook.setnamed(sm_name, 'x span', param_value * 1e-6)
                fdtd_hook.setnamed(sm_name, 'y span', monitorbox_size_lateral_um * 1e-6)
            elif idx == 3:
                fdtd_hook.setnamed(sm_name, 'y', (-device_size_lateral_um - param_value)*0.5 * 1e-6)
                fdtd_hook.setnamed(sm_name, 'y span', param_value * 1e-6)
                fdtd_hook.setnamed(sm_name, 'x span', monitorbox_size_lateral_um * 1e-6)
                    
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}\n"
            message = template.format(type(ex).__name__, ex.args)
            print(message+'FDTD Object does not exist.')
            # fdtd_hook.addmesh(name=sidewall_mesh['name'])
            # pass
    
    
    fdtd_hook.setnamed('incident_aperture_monitor', 'x span', monitorbox_size_lateral_um * 1e-6)
    fdtd_hook.setnamed('incident_aperture_monitor', 'y span', monitorbox_size_lateral_um * 1e-6)
    fdtd_hook.setnamed('exit_aperture_monitor', 'x span', monitorbox_size_lateral_um * 1e-6)
    fdtd_hook.setnamed('exit_aperture_monitor', 'y span', monitorbox_size_lateral_um * 1e-6)
    
    sm_x_pos = [monitorbox_size_lateral_um/2, 0, -monitorbox_size_lateral_um/2, 0]
    sm_y_pos = [0, monitorbox_size_lateral_um/2, 0, -monitorbox_size_lateral_um/2]
    sm_xspan_pos = [0, monitorbox_size_lateral_um, 0, monitorbox_size_lateral_um]
    sm_yspan_pos = [monitorbox_size_lateral_um, 0, monitorbox_size_lateral_um, 0]
    
    for sm_idx in range(0, num_sidewalls):
        sm_name = 'side_monitor_' + str(sm_idx)
        fdtd_hook.select(sm_name)
        fdtd_hook.setnamed(sm_name, 'x max', (sm_x_pos[sm_idx] + sm_xspan_pos[sm_idx]/2) * 1e-6)
        fdtd_hook.setnamed(sm_name, 'x min', (sm_x_pos[sm_idx] - sm_xspan_pos[sm_idx]/2) * 1e-6)
        fdtd_hook.setnamed(sm_name, 'y max', (sm_y_pos[sm_idx] + sm_yspan_pos[sm_idx]/2) * 1e-6)
        fdtd_hook.setnamed(sm_name, 'y min', (sm_y_pos[sm_idx] - sm_yspan_pos[sm_idx]/2) * 1e-6)

def change_divergence_angle(param_value):
    '''For a Gaussian beam source, changes the divergence angle while keeping beam radius constant.'''
    xy_names = ['x', 'y']
    
    for xy_idx in range(0, 2):
        fdtd_hook.select('forward_src_' + xy_names[xy_idx])
        # fdtd_hook.set('beam radius wz', 100 * 1e-6)
        # fdtd_hook.set('divergence angle', 3.42365) # degrees
        # fdtd_hook.set('beam radius wz', 15 * 1e-6)
        # # fdtd_hook.set('beam radius wz', param_value * 1e-6)
        
        
        # Resets distance from waist to 0, fixing waist radius and beam radius properties
        fdtd_hook.set('beam parameters', 'Waist size and position')
        fdtd_hook.set('waist radius w0', src_beam_rad * 1e-6)
        fdtd_hook.set('distance from waist', 0 * 1e-6)
        # Change divergence angle
        fdtd_hook.set('beam parameters', 'Beam size and divergence angle')
        fdtd_hook.set('divergence angle', param_value) # degrees
    
    print(f'Source divergence angle set to {param_value}.')

def change_permittivity_point(param_value, perturbation, axis='z'):
    '''Input param_value is the position in the device, in units of device length, along a certain axis that can be optionally specified.
    This function goes to that position, and perturbs the voxel permittivity by the perturbation amount (direction specified in input).'''
    
    # TODO: Code function!!!
    print('Function not yet coded.')
    
    
     # Set sigmoid strength to most recent epoch and get (filtered) permittivity
    
    cur_design_variable = np.load( 
            projects_directory_location + "/cur_design_variable.npy" )
    bayer_filter.w[0] = cur_design_variable
    bayer_filter.update_permittivity() 
    
    # bayer_filter.update_filters( num_epochs - 1 )
    # bayer_filter.update_permittivity()
    cur_density = bayer_filter.get_permittivity()
    # # Binarize the current permittivity
    cur_density = 1.0 * np.greater_equal( cur_density, 0.5 )        # TODO: Why not greater_equal(cur_density, 0.0)?

    # np.save( projects_directory_location + '/density_eval.npy', cur_density )
    
    # Get point coords
    coord_idx = np.array([len(bayer_filter_region_x)/2, len(bayer_filter_region_y)/2, len(bayer_filter_region_z)/2], dtype=int)
    # coord_idx = np.array([0,0,0], dtype=int)
    
    if axis in ['x']:
        bayer_filter_axis = bayer_filter_region_x
    elif axis in ['y']:
        bayer_filter_axis = bayer_filter_region_y
    else:
        bayer_filter_axis = bayer_filter_region_z
        
    coord_value = bayer_filter_axis[0] + param_value*(bayer_filter_axis[-1] - bayer_filter_axis[0])
    
    if axis in ['x']:
        coord_idx[0] = np.argmin(np.abs(np.array(bayer_filter_axis) - coord_value))
    elif axis in ['y']:
        coord_idx[1] = np.argmin(np.abs(np.array(bayer_filter_axis) - coord_value))
    else:
        coord_idx[2] = np.argmin(np.abs(np.array(bayer_filter_axis) - coord_value))
    
    # Get permittivity at point
        
    # epsilon = cur_density[tuple(x for x in coord_idx)]
    
    dispersive_range_idx = 0

    dispersive_max_permittivity = dispersion_model.average_permittivity( dispersive_ranges_um[ dispersive_range_idx ] )
    disperesive_max_index = index_from_permittivity( dispersive_max_permittivity )
    cur_permittivity = min_device_permittivity + ( dispersive_max_permittivity - min_device_permittivity ) * cur_density
    cur_index = index_from_permittivity( cur_permittivity )
    
    epsilon = cur_permittivity[tuple(x for x in coord_idx)]
    print(f'Epsilon is: {epsilon}')
    print(f'Old index is: {cur_index[tuple(x for x in coord_idx)]}')
    
    # Change permittivity at point - NOT index
    
    # TODO: What amount should it be perturbed by? Should it just be a simple 2.4^2 -> 1.5^2 and vice versa? What happens when it's 2.4 + δa?
    # TODO: Clearly perturbing it BEFORE applying the binarization filter won't work!
    # TODO: Maybe we don't use a two-sided gradient, but merely a one-sided one? Will it be smooth enough if there is +δa on one side of a point
    # TODO: and -δa on the other side of that point?
    
    # cur_density[tuple(x for x in coord_idx)] = epsilon + perturbation
    # print(f'New epsilon is: {epsilon+perturbation}')
    
    # # Translate into index and update cur_index and then design_import
    
    # dispersive_range_idx = 0

    # dispersive_max_permittivity = dispersion_model.average_permittivity( dispersive_ranges_um[ dispersive_range_idx ] )
    # disperesive_max_index = index_from_permittivity( dispersive_max_permittivity )
    # cur_permittivity_2 = min_device_permittivity + ( dispersive_max_permittivity - min_device_permittivity ) * cur_density
    # cur_index_2 = index_from_permittivity( cur_permittivity_2 )
    
    cur_permittivity_2 = cur_permittivity.copy()
    cur_permittivity_2[tuple(x for x in coord_idx)] = epsilon + perturbation
    cur_index_2 = index_from_permittivity( cur_permittivity_2 )
    
    cur_index = cur_index[0:device_voxels_lateral, 0:device_voxels_lateral, 0:num_vertical_layers]
    cur_index_2 = cur_index_2[0:device_voxels_lateral, 0:device_voxels_lateral, 0:num_vertical_layers]
    fdtd_hook.select( 'design_import' )
    fdtd_hook.importnk2( cur_index_2, bayer_filter_region_x, bayer_filter_region_y, bayer_filter_region_z )
    
    print(f'Point at {param_value} set to permittivity {epsilon + perturbation}.')
    print(f'New index at point is: {cur_index_2[tuple(x for x in coord_idx)]}.')

#! Step 2 Functions

def get_sourcepower():
    f = fdtd_hook.getresult('focal_monitor_0','f')
    sp = fdtd_hook.sourcepower(f)
    return sp

def get_afield(monitor_name, field_indicator):
    field_polariations = [field_indicator + 'x',
                          field_indicator + 'y', field_indicator + 'z']
    data_xfer_size_MB = 0

    start = time.time()

    field_pol_0 = fdtd_hook.getdata(monitor_name, field_polariations[0])
    data_xfer_size_MB += field_pol_0.nbytes / (1024. * 1024.)

    total_field = np.zeros([len(field_polariations)] +
                           list(field_pol_0.shape), dtype=np.complex)
    total_field[0] = field_pol_0

    for pol_idx in range(1, len(field_polariations)):
        field_pol = fdtd_hook.getdata(
            monitor_name, field_polariations[pol_idx])
        data_xfer_size_MB += field_pol.nbytes / (1024. * 1024.)

        total_field[pol_idx] = field_pol

    elapsed = time.time() - start

    date_xfer_rate_MB_sec = data_xfer_size_MB / elapsed
    log_file = open(projects_directory_location + "/log.txt", 'a')
    log_file.write("Transferred " + str(data_xfer_size_MB) + " MB\n")
    log_file.write("Data rate = " + str(date_xfer_rate_MB_sec) + " MB/sec\n\n")
    log_file.close()

    return total_field

def get_hfield(monitor_name):
    return get_afield(monitor_name, 'H')

def get_efield(monitor_name):
    return get_afield(monitor_name, 'E')

def get_efield_magnitude(monitor_name):
    e_field = get_afield(monitor_name, 'E')
    Enorm2 = np.square(np.abs(e_field[0,:,:,:,:]))
    for idx in [1,2]:
        Enorm2 += np.square(np.abs(e_field[idx,:,:,:,:]))
    return np.sqrt(Enorm2)

def get_transmission_magnitude(monitor_name):
    '''Returns transmission spectrum, i.e. a [lambda x 1] vector.'''
    # Lumerical defines all transmission measurements as normalized to the source power.
    read_T = fdtd_hook.getresult(monitor_name, 'T')
    return np.abs(read_T['T'])

def get_power(monitor_name):
    '''Returns power as a function of space and wavelength, with three components Px, Py, Pz'''
    read_P = fdtd_hook.getresult(monitor_name, 'P')
    return read_P['P']

def get_overall_power(monitor_name):
    '''Returns power spectrum, power being the summed power over the monitor.'''
    # Lumerical defines all transmission measurements as normalized to the source power.
    source_power = get_sourcepower()
    T = get_transmission_magnitude(monitor_name)
    
    return source_power * np.transpose([T])

def get_beamdata(src_name, field_indicator):
    field_polarizations = [ field_indicator + 'x',
                       field_indicator + 'y', field_indicator + 'z' ]
    data_xfer_size_MB = 0
    
    def extract_lambda_matches(field_pol_old):
        '''Wavelength matching - Getting the beam data in layout mode means that the wavelengths don't line up with what we've specified in params
        e.g. Layout Mode vector has 150 equal frequency values, new vector has 45 equal λ values'''
        new_shape = list(field_pol_old.shape)[:-1] + [num_design_frequency_points]
        field_pol_new = np.zeros(new_shape, dtype=np.complex)
        
        for j, wl in enumerate(lambda_values_um):
            # Matches wl to the closest value in field_dataset['lambda'] (which is c/field_dataset['f']) and finds the corresponding index.
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
    beam_field = get_beamdata(src_name, 'H')
    return truncate_adj_src_field(beam_field, src_name)

def get_efield_mode(src_name):
    beam_field = get_beamdata(src_name, 'E')
    return truncate_adj_src_field(beam_field, src_name)

def truncate_adj_src_field(total_field, src_name):
    '''The source must have span larger than the quadrant it belongs in in order to avoid cut-off errors in simulation. This will crop the field to the right dimensions.'''
    # total_field has shape (3, nx, ny, nz, nλ)
    
    field_dataset = fdtd_hook.getresult( src_name, 'fields' )        # dictionary with keys ['lambda', 'f', 'x', 'y', 'z', 'E', 'H', 'Lumerical_dataset']
                                                                     # the 'E' item has shape (nx, ny, nz, nλ, 3)
    field_x = field_dataset['x']
    field_y = field_dataset['y']
    
    x_min_idx = min(range(len(field_x)), key=lambda i: abs(field_x[i,0] - fdtd_hook.getnamed(transmission_monitor_focal['name'], 'x min')))
    y_min_idx = min(range(len(field_y)), key=lambda i: abs(field_y[i,0] - fdtd_hook.getnamed(transmission_monitor_focal['name'], 'y min')))
    x_max_idx = min(range(len(field_x)), key=lambda i: abs(field_x[i] - fdtd_hook.getnamed(transmission_monitor_focal['name'], 'x max')))
    y_max_idx = min(range(len(field_y)), key=lambda i: abs(field_y[i] - fdtd_hook.getnamed(transmission_monitor_focal['name'], 'y max')))
    #print(f'Src Name: {src_name}, X: {x_min_idx}, {x_max_idx}; Y: {y_min_idx}, {y_max_idx}')
    
    total_field_2 = total_field[:,x_min_idx:x_max_idx, y_min_idx:y_max_idx, :,:]
    return total_field_2


#! Step 3 Functions

def partition_bands(wavelength_vector, band_centers):
    last_idx = len(wavelength_vector) - 1
    
    band_bounds = []
    for b_idx in range(0, len(band_centers)-1):
        band_bounds.append(np.average(np.array([band_centers[b_idx], band_centers[b_idx+1]])))
    
    band_idxs = []
    for bound in band_bounds:
        bound_idx = min(range(len(wavelength_vector)), key=lambda i: abs(wavelength_vector[i]-bound))
        band_idxs.append(bound_idx)
    
    processed_band_idxs = [np.array([0, band_idxs[0]])]
    for b_idx in range(0, len(band_idxs)-1):
        processed_band_idxs.append(np.array([band_idxs[b_idx], band_idxs[b_idx + 1]]))
    processed_band_idxs.append(np.array([band_idxs[-1], last_idx]))
    
    return processed_band_idxs

def partition_scatter_plane_monitor(monitor_data, variable_indicator):
    '''The scatter plane monitor contains the overall focal plane monitor and its individual quadrants at the center. 
    This partitions off the data corresponding to the variable indicated, in the five squares, and returns them as output.'''
    
    # Get spatial data of the monitor
    x_coords = fdtd_hook.getresult(monitor_data['name'],'x')
    y_coords = fdtd_hook.getresult(monitor_data['name'],'y')
    z_coords = fdtd_hook.getresult(monitor_data['name'],'z')
    
    # Find the index of the x-entry with x=0
    zero_idx = min(range(len(x_coords)), key=lambda i: abs(x_coords[i] - 0))
    idxs_from_zero_to_border = min(range(len(x_coords)), key=lambda i: abs(x_coords[i] - device_size_lateral_um*0.5*1e-6)) \
                                - zero_idx
    # Since the focal plane quadrants are squares symmetric about the xy-axes, this only needs to be done once for all four borders
    
    # Slice the overall field array accordingly - this slicing is done for an array of size (x,y,z,lambda) i.e. E-magnitude
    var_scatter_plane = monitor_data[ variable_indicator ]
    var_focal_plane = var_scatter_plane[zero_idx - idxs_from_zero_to_border : zero_idx + idxs_from_zero_to_border,
                                          zero_idx - idxs_from_zero_to_border : zero_idx + idxs_from_zero_to_border,
                                          :,:]
    
    # 0 is in the top right, corresponds to blue
    # Then we go counter-clockwise.
    # 1 is in the top left, corresponds to green, x-pol
    # 2 is in the bottom left, corresponds to red
    var_focal_monitor_arr = [None]*4
    var_focal_monitor_arr[0] = var_scatter_plane[zero_idx: zero_idx + idxs_from_zero_to_border,
                                                zero_idx: zero_idx + idxs_from_zero_to_border,
                                                :,:]
    var_focal_monitor_arr[1] = var_scatter_plane[zero_idx - idxs_from_zero_to_border : zero_idx ,
                                                zero_idx: zero_idx + idxs_from_zero_to_border,
                                                :,:]
    var_focal_monitor_arr[2] = var_scatter_plane[zero_idx - idxs_from_zero_to_border : zero_idx,
                                                zero_idx - idxs_from_zero_to_border : zero_idx,
                                                :,:]
    var_focal_monitor_arr[3] = var_scatter_plane[zero_idx: zero_idx + idxs_from_zero_to_border,
                                                zero_idx - idxs_from_zero_to_border : zero_idx,
                                                :,:]
    
    return var_focal_plane, var_focal_monitor_arr

def append_new_sweep(sweep_values, f_vectors, num_sweep_values, statistics, band_idxs=None):
    '''Logs peak/mean values and corresponding indices of the f_vector values, for each quadrant, IN BAND.
    This means that it takes the values within a defined spectral band and picks a statistical value for each band within that band.
    Produces num_sweep_values * num_bands values, i.e. B->{B,G,R}, G->{B,G,R}, R->{B,G,R}  
    where for e.g. RB is the transmission of R quadrant within the B spectral band
    Statistics determine how the value for the sweep is drawn from the spectrum: peak|mean'''

    if band_idxs is None:
        # Then take the whole spectrum as one band
        band_idxs = [np.array([0, len(f_vectors[0]['var_values'])-1])]
    num_bands = len(band_idxs)
    
    for idx in range(0, num_sweep_values):
        for inband_idx in range(0, num_bands):
            if statistics in ['peak']:
                sweep_index, sweep_val = max(enumerate(f_vectors[idx]['var_values'][slice(*band_idxs[inband_idx])]), key=(lambda x: x[1]))
                sweep_stdev = 0
                
            else:
                sweep_val = np.average(f_vectors[idx]['var_values'][slice(*band_idxs[inband_idx])])
                sweep_index = min(range(len(f_vectors[idx]['var_values'])), key=lambda i: abs(f_vectors[idx]['var_values'][i]-sweep_val))
                sweep_stdev = np.std(f_vectors[idx]['var_values'][slice(*band_idxs[inband_idx])])
        
          
            sweep_values.append({'value': float(sweep_val),
                            'index': sweep_index,
                            'std_dev': sweep_stdev,
                            'statistics': statistics
                            })
    
    return sweep_values

def append_new_fom_sweep(sweep_values, sweep_val, num_sweep_values, statistics, band_idxs=None):

    sweep_values.append({'value': float(sweep_val),
                            'index': 0,
                            'std_dev': 0,
                            'statistics': statistics
                            })
    return sweep_values

def generate_gradient_comparison_spectrum(monitors_data, produce_spectrum, param_idx, statistics='mean'):
    '''Generates FoM spectrum for the overall scatter plane (focal region quadrants and crosstalk/spillover), 
    and logs peak/mean value and corresponding index. Compares with adjoint gradient.'''
    
    print("Beginning Step: Computing Figure of Merit")
    sys.stdout.flush()
    starttime=time.time()
    
    def poynting_flux(E, H, total_surface_area, real_e_h = True, dot_dS = True, pauseDebug=False):
        '''Calculates the expression \int (E x H).dS which, for E: Electric field; H: Magnetic field, is the Poynting flux.
        Assumes a regular and uniform mesh size throughout.'''
        
        # E, H should have shape (3, nx, ny, nz, nλ）
        
        if pauseDebug:
            print('3')
        
        # Get the cross product of E and H along the axis that defines the last axis of the cross product
        print(f'E has shape: {np.shape(E)}')         # TODO: Here the shape is (3,61,61,1)
        print(f'H has shape: {np.shape(H)}')         # TODO: Here the shape is (3,63,63,1)
        # e_max_len = max(np.shape(E))
        # h_max_len = max(np.shape(H))
        # if e_max_len < h_max_len:
        #     H = H[:,0:e_max_len, 0:e_max_len,:]
        # else:
        #     E = E[:,0:h_max_len, 0:h_max_len,:]         # TODO: Get rid of this / fix this! This is only a quick stopgap for back compatibility
        # # TODO: Source of the problem is that device_size_lateral_um = geometry_spacing*41 but transmission monitors are geometry_spacing*40
        
        max_len = np.min([np.shape(E)[1], np.shape(E)[2], np.shape(H)[1], np.shape(H)[2]])
        H = H[:,0:max_len, 0:max_len,:]
        E = E[:,0:max_len, 0:max_len,:]
        print(f'Amended. E has shape: {np.shape(E)} and H has shape: {np.shape(H)}')
        
        
        integral = np.cross(E, H, axis=0)            # shape: (3, nx, ny, nz)
        if real_e_h:
            integral = np.real(integral)
        # Do the dot product, i.e. take the z (normal)-component; 
        if dot_dS:
            integral = integral[2]                       # shape: (nx, ny, nz)
        # then integrate by summing everything and multiplying by S
        integral = np.sum(integral, (0,1,2)) * total_surface_area
        
        return integral
    
    #* Set up some numpy arrays to handle all the data we will pull out of the simulation.
    forward_e_fields = {}                       # Records E-fields throughout device volume
    focal_e_data = {}                           # Records E-fields at focal plane quadrants (locations of adjoint sources) for forward source
    focal_h_data = {}                           # Records H-fields at focal plane quadrants (locations of adjoint sources) for forward source
    quadrant_transmission_data = {}             # Records transmission through each quadrant

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
    
    xy_names = ['x', 'y']
    
    for xy_idx in range(0, 2):
        focal_e_data[xy_names[xy_idx]] = [
                None for idx in range( 0, num_focal_spots ) ]
        focal_h_data[xy_names[xy_idx]] = [
                None for idx in range( 0, num_focal_spots ) ]
        quadrant_transmission_data[xy_names[xy_idx]] = [ 
                None for idx in range( 0, num_focal_spots ) ]
        # mode_e_fields[xy_names[xy_idx]] = [
                # None for idx in range( 0, num_focal_spots ) ]
        # mode_h_fields[xy_names[xy_idx]] = [
                # None for idx in range( 0, num_focal_spots ) ]

    for dispersive_range_idx in range( 0, num_dispersive_ranges ):
        for xy_idx in range(0, 2):
            
            # # TODO: Undo this bit when no longer build-testing
            if running_on_local_machine:
                actual_directory = r'C:\Users\Ian\Dropbox\Caltech\Faraon Group\Simulations\Mode Overlap FoM\fom_dev\fom_gradient_test\ares_mode_overlap_fom_check_gradient'
                actual_filepath = convert_root_folder(job_names[('sweep', param_idx, 'forward', xy_idx, dispersive_range_idx)], actual_directory)
            else:
                actual_filepath = job_names[('sweep', param_idx, 'forward', xy_idx, dispersive_range_idx)]
            print(f'Loading: {isolate_filename(actual_filepath)}')
            sys.stdout.flush()
            fdtd_hook.load(actual_filepath)

            #TODO: Check that this is OK
            fwd_e_fields = get_efield(monitors_data['design_efield_monitor']['name'])
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
                focal_e_data[xy_names[xy_idx]][ adj_src_idx ] = get_efield(monitors_data['transmission_monitor_'+str(adj_src_idx)]['name'])
                focal_e_data[xy_names[xy_idx]][-1] = get_efield(transmission_monitor_focal['name'])
                print(f'Accessed adjoint E-field data over focal quadrant {adj_src_idx}.')
                print(f'Overall data is of shape {np.shape(focal_e_data[xy_names[xy_idx]][-1])}')
                
                # NOTE: If you want focal_h_data['x'] to be an ndarray, it must be initialized as np.empty above.
                focal_h_data[xy_names[xy_idx]][ adj_src_idx ] = get_hfield(monitors_data['transmission_monitor_'+str(adj_src_idx)]['name'])
                focal_h_data[xy_names[xy_idx]][-1] = get_hfield(transmission_monitor_focal['name'])
                print(f'Accessed adjoint H-field data over focal quadrant {adj_src_idx}.')
                print(f'Overall data is of shape {np.shape(focal_h_data[xy_names[xy_idx]][-1])}')

                quadrant_transmission_data[xy_names[xy_idx]][ adj_src_idx ] = get_transmission_magnitude(monitors_data['transmission_monitor_'+str(adj_src_idx)]['name'])
                print(f'Accessed transmission data for quadrant {adj_src_idx}.')
                
                
    endtime=time.time()
    print(f'Loading data took: {(endtime-starttime)/60} min.')


    # These arrays record intensity and transmission FoMs for all focal spots
    figure_of_merit_per_focal_spot = []
    transmission_per_quadrant = []
    # These arrays record the same, but for each wavelength
    transmission_by_wavelength = np.zeros(num_design_frequency_points)
    intensity_fom_by_wavelength = np.zeros(num_design_frequency_points)
    mode_overlap_fom_by_wavelength = np.zeros(num_design_frequency_points)
    transmission_by_quadrant_by_wl = np.zeros((num_focal_spots, num_design_frequency_points))
    intensity_fom_by_quadrant_by_wl = np.zeros((num_focal_spots, num_design_frequency_points))
    mode_overlap_fom_by_quadrant_by_wl = np.zeros((num_focal_spots, num_design_frequency_points))   

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
                
                #* Intensity FoM Calculation: FoM = |E(x_0)|^2                 
                # \text{FoM}_{\text{intensity}} = \sum_i^{} \left{ \sum_\lambda^{} w_{\text{rejection}}(\lambda) w_{f_i}
                # \frac{\left|E_{f_i}(\lambda)\right|^2}{I_{\text{norm, Airy}}}
                # https://i.imgur.com/5jcN01k.png
                compute_fom += weight_spectral_rejection * np.sum(
                    (
                        np.abs(get_focal_e_data[focal_idx][:, 0, 0, 0, spectral_focal_plane_map[focal_idx][0] + spectral_idx])**2 /
                        total_weighting[spectral_idx]
                    )
                )
                
                #* Mode Overlap FoM Calculation:                 
                # \text{FoM}_{\text{m.o.}} = \sum_i^{} \left{ \sum_\lambda^{} w_{\text{rejection}}(\lambda) w_{f_i} 
                # \frac{1/8}{I_{\text{norm, Airy}}} 
                # \frac{\left| \int \textbf{E} \times\overline{\textbf{H}_m} \cdot d\textbf{S} + \int \overline{\textbf{E}_m} \times \textbf{H} \cdot d\textbf{S} \right|^2}
                # {\int Re\left( \textbf{E}_m \times\overline{\textbf{H}_m} \right)\cdot d\textbf{S}}
                # https://i.imgur.com/fCI56r1.png
                
                
                
                fom_integral_1 = poynting_flux(get_focal_e_data[-1][:,:,:,:,spectral_focal_plane_map[focal_idx][0] + spectral_idx],
                                        np.conj(get_mode_h_fields[focal_idx][:,:,:,:,spectral_focal_plane_map[focal_idx][0] + spectral_idx]),
                                        (0.5*device_size_lateral_um)**2)
                fom_integral_2 = poynting_flux(np.conj(get_mode_e_fields[focal_idx][:,:,:,:,spectral_focal_plane_map[focal_idx][0] + spectral_idx]),
                                        get_focal_h_data[-1][:,:,:,:,spectral_focal_plane_map[focal_idx][0] + spectral_idx],
                                        (0.5*device_size_lateral_um)**2)
                fom_integral_3 = poynting_flux(get_mode_e_fields[focal_idx][:,:,:,:,spectral_focal_plane_map[focal_idx][0] + spectral_idx],
                                        np.conj(get_mode_h_fields[focal_idx][:,:,:,:,spectral_focal_plane_map[focal_idx][0] + spectral_idx]),
                                        (0.5*device_size_lateral_um)**2, real_e_h = True)
                
                # compute_fom += weight_spectral_rejection * np.sum(
                #         1/8 * (np.abs(fom_integral_1 + fom_integral_2)**2 / fom_integral_3
                #                 ) /1 # total_weighting[spectral_idx]
                # )

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
                                ) / 1 #total_weighting[spectral_idx]
                )
                
                print(f'FoM integrals have shape: {np.shape(fom_integral_1)}')
                try:
                    print(f'Surface integrals are, 1: {fom_integral_1}; 2: {fom_integral_2}; 3: {fom_integral_3}')
                except Exception as ex: pass
                print(f'Intensity FoM for index {spectral_idx} is {intensity_fom_by_wavelength}')
                print(f'Mode Overlap FoM for index {spectral_idx} is {mode_overlap_fom_by_wavelength}')
                try:
                    print(f'Total weighting for index {spectral_idx} is {total_weighting[spectral_idx]}')
                except Exception as ex: pass
                
            for spectral_idx in range(0, num_bands*num_points_per_band-1):
                
                max_intensity_weighting = max_intensity_by_wavelength[:]
                # Use the max_intensity_weighting to renormalize the intensity distribution used to calculate the FoM
                # and insert the custom weighting schemes here as well.
                total_weighting = max_intensity_weighting / weight_focal_plane_map_performance_weighting[focal_idx]
                one_over_weight_focal_plane_map_performance_weighting = 1. / weight_focal_plane_map_performance_weighting[focal_idx]
                
                print(f'Num. bands is {num_bands}')
                print(f'Num. points per band is {num_points_per_band}')
                print(np.shape(total_weighting))
                
                transmission_by_quadrant_by_wl[focal_idx, spectral_idx] += weight_spectral_rejection * \
                        get_quad_transmission_data[focal_idx][spectral_idx] / \
                            one_over_weight_focal_plane_map_performance_weighting
                
                intensity_fom_by_quadrant_by_wl[focal_idx, spectral_idx] += weight_spectral_rejection * np.sum(
                    (
                        np.abs(get_focal_e_data[focal_idx][:, 31, 31, 0, spectral_idx])**2 /
                        total_weighting[spectral_idx]
                    )
                )
                
                fom_integral_1 = poynting_flux(get_focal_e_data[-1][:,:,:,:,spectral_idx],
                                        np.conj(get_mode_h_fields[focal_idx][:,:,:,:,spectral_idx]),
                                        (0.5*device_size_lateral_um)**2)
                fom_integral_2 = poynting_flux(np.conj(get_mode_e_fields[focal_idx][:,:,:,:,spectral_idx]),
                                        get_focal_h_data[-1][:,:,:,:,spectral_idx],
                                        (0.5*device_size_lateral_um)**2)
                fom_integral_3 = poynting_flux(get_mode_e_fields[focal_idx][:,:,:,:,spectral_idx],
                                        np.conj(get_mode_h_fields[focal_idx][:,:,:,:,spectral_idx]),
                                        (0.5*device_size_lateral_um)**2, real_e_h = True)
                
                mode_overlap_fom_by_quadrant_by_wl[focal_idx, spectral_idx ] += weight_spectral_rejection * np.sum(
                        1/8 * (np.abs(fom_integral_1 + fom_integral_2)**2 / np.abs(fom_integral_3)
                                ) / 1 # total_weighting[spectral_idx]
                )

        figure_of_merit_per_focal_spot.append(compute_fom)
        print(f'Intensity FoM overall is:')
        print(figure_of_merit_per_focal_spot)
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
    # figure_of_merit_evolution[epoch, iteration] = figure_of_merit
    # figure_of_merit_by_wl_evolution[epoch, iteration] = figure_of_merit_per_focal_spot

    # transmission_by_focal_spot_evolution[epoch, iteration] = transmission_per_quadrant
    # transmission_by_wl_evolution[epoch, iteration] = transmission_by_wavelength
    # intensity_fom_by_wavelength_evolution[epoch, iteration] = intensity_fom_by_wavelength
    # mode_overlap_fom_by_wavelength_evolution[epoch, iteration] = mode_overlap_fom_by_wavelength

    # np.save(projects_directory_location + "/figure_of_merit.npy", figure_of_merit_evolution)
    # np.save(projects_directory_location + "/figure_of_merit_by_wl.npy", figure_of_merit_by_wl_evolution)
    # np.save(projects_directory_location + "/transmission_by_focal_spot.npy", transmission_by_focal_spot_evolution)
    # np.save(projects_directory_location + "/transmission_by_wl.npy", transmission_by_wl_evolution)
    # np.save(projects_directory_location + "/intensity_fom_by_wavelength.npy", intensity_fom_by_wavelength_evolution)

    # print("Figure of Merit So Far:")
    # print(figure_of_merit_evolution)
    
    # np.save(projects_directory_location + "/figure_of_merit.npy", figure_of_merit_evolution)
    # np.save(projects_directory_location + "/transmission_by_focal_spot.npy", transmission_by_focal_spot_evolution)
    # np.save(projects_directory_location + "/transmission_by_wl.npy", transmission_by_wl_evolution)
    # np.save(projects_directory_location + "/transmission_by_focal_spot_by_wl.npy", transmission_by_focal_spot_and_wavelength_evolution)
    # np.save(projects_directory_location + "/intensity_fom_by_wavelength.npy", intensity_fom_by_wavelength_evolution)
    # np.save(projects_directory_location + "/intensity_fom_by_focal_spot_by_wl.npy", intensity_fom_by_focal_spot_and_wavelength_evolution)
    # np.save(projects_directory_location + "/mode_overlap_fom_by_wavelength.npy", mode_overlap_fom_by_wavelength_evolution)
    # np.save(projects_directory_location + "/mode_overlap_fom_by_focal_spot_by_wl.npy", mode_overlap_fom_by_focal_spot_and_wavelength_evolution)
    

    print("Completed: Calculated figures of merit.")
    print(figure_of_merit)
    sys.stdout.flush()
    
    # print("Beginning Step 3: Adjoint Optimization Jobs")
    # sys.stdout.flush()

    # ## Get the device / current permittivity's dimensions in terms of (geometry) voxels       NOTE: UNUSED
    # # cur_permittivity_shape = cur_permittivity.shape
    # # Get the device's dimensions in terms of (MESH) voxels
    # field_shape = [device_voxels_simulation_mesh_lateral, device_voxels_simulation_mesh_lateral, device_voxels_simulation_mesh_vertical]

    # # Initialize arrays to store the gradients of each polarization, each the shape of the voxel mesh
    # xy_polarized_gradients = [ np.zeros(field_shape, dtype=np.complex), np.zeros(field_shape, dtype=np.complex) ]

    # for adj_src_idx in range(0, num_adjoint_sources):
    #     polarizations = polarizations_focal_plane_map[adj_src_idx]
    #     spectral_indices = spectral_focal_plane_map[adj_src_idx]

    #     gradient_performance_weight = 0
    #     if not weight_by_individual_wavelength:
    #         gradient_performance_weight = performance_weighting[adj_src_idx]

    #         if do_rejection:
    #             if weight_by_quadrant_transmission:
    #                 gradient_performance_weight *= softplus_prime( transmission_per_quadrant[ adj_src_idx ] )
    #             else:
    #                 gradient_performance_weight *= softplus_prime( figure_of_merit_per_focal_spot[ adj_src_idx ] )

    #     lookup_dispersive_range_idx = adjoint_src_to_dispersive_range_map[ adj_src_idx ]

    #     # Store the design E-field for the adjoint simulations
    #     adjoint_e_fields = []
    #     starttime = time.time()
    #     for xy_idx in range(0, 2):
    #         #
    #         # This is ok because we are only going to be using the spectral idx corresponding to this range in the summation below
    #         #
            
    #         actual_filepath = convert_root_folder(job_names[ ( 'adjoint', adj_src_idx, xy_idx, lookup_dispersive_range_idx ) ], actual_directory)
    #         #fdtd_hook.load( job_names[ ( 'forward', xy_idx, dispersive_range_idx ) ] )
    #         print(f'Loading: {isolate_filename(actual_filepath)}')
    #         fdtd_hook.load(actual_filepath)
    #         #fdtd_hook.load( job_names[ ( 'adjoint', adj_src_idx, xy_idx, lookup_dispersive_range_idx ) ] )
    #         #print('Loading .fsp file: '+job_names[('adjoint', adj_src_idx, xy_idx, lookup_dispersive_range_idx)])
    #         #sys.stdout.flush()

    #         adjoint_e_fields.append(
    #             get_efield(design_efield_monitor['name']))
    #     endtime=time.time()
    #     print(f'Loading data took: {(endtime-starttime)/60} min.')
        
    #     for pol_idx in range(0, len(polarizations)):
    #         pol_name = polarizations[pol_idx]
    #         get_focal_e_data = focal_e_data[pol_name]                           # Get E-field at adjoint source for each polarization
    #         get_focal_h_data = focal_h_data[pol_name]                           # Get H-field at adjoint source for each polarization
    #         get_mode_e_fields = mode_e_fields[pol_name]                         # Get E-field at device for each polarization, given adjoint source
    #         get_mode_h_fields = mode_h_fields[pol_name]                         # Get H-field at device for each polarization, given adjoint source
    #         pol_name_to_idx = polarization_name_to_idx[pol_name]

    #         for xy_idx in range(0, 2):
    #             #* Conjugate of E_{old}(x_0) -field at the adjoint source of interest, with direction along the polarization
    #             # This is going to be the amplitude of the dipole-adjoint source driven at the focal plane
                
    #             # Create a normal z-vector
    #             # normal_vector = [0,0,1]     #NOTE: This is fine, this works in np.cross()
    #             #normal_vector = np.ones(np.shape(get_focal_h_data[adj_src_idx]))
    #             normal_vector = np.zeros(np.shape(get_focal_h_data[adj_src_idx]))
    #             normal_vector[2] = np.ones(np.shape(normal_vector[2]))
    #             mu_0 = 1.25663706 * 1e-6
                
    #             adjoint_integral_1 = poynting_flux(get_focal_e_data[adj_src_idx][:,:,:,:, spectral_indices[0] : spectral_indices[1] : 1],
    #                                                 np.conj(get_mode_h_fields[adj_src_idx][:,:,:,:, spectral_indices[0] : spectral_indices[1] : 1]),
    #                                                 (0.5*device_size_lateral_um)**2)
    #             adjoint_integral_2 = poynting_flux(np.conj(get_mode_e_fields[adj_src_idx][:,:,:,:, spectral_indices[0] : spectral_indices[1] : 1]),
    #                                                 get_focal_h_data[adj_src_idx][:,:,:,:, spectral_indices[0] : spectral_indices[1] : 1],
    #                                                 (0.5*device_size_lateral_um)**2)
    #             adjoint_integral_3 = poynting_flux(get_mode_e_fields[adj_src_idx][:,:,:,:, spectral_indices[0] : spectral_indices[1] : 1],
    #                                                 np.conj(get_mode_h_fields[adj_src_idx][:,:,:,:, spectral_indices[0] : spectral_indices[1] : 1]),
    #                                                 (0.5*device_size_lateral_um)**2, real_e_h=True)
    #             # TODO: 
    #             adjoint_A_factor = 1/4 * np.conj(adjoint_integral_1 + adjoint_integral_2) / (adjoint_integral_3)
    #             print("Calculated adjoint A factor.")
    #             sys.stdout.flush()
                            
                
    #             source_weight =  np.squeeze(adjoint_A_factor) 
    #                 # * (
    #                 # poynting_flux(np.conj(get_focal_h_data[adj_src_idx][:, :,:,:, spectral_indices[0] : spectral_indices[1] : 1]),
    #                 #               normal_vector[:, :,:,:, spectral_indices[0] : spectral_indices[1] : 1], 
    #                 #               (0.5*device_size_lateral_um)**2, dot_dS = False, pauseDebug=False) - 
    #                 # poynting_flux(normal_vector[:, :,:,:, spectral_indices[0] : spectral_indices[1] : 1],
    #                 #               np.conj(get_focal_e_data[adj_src_idx][:, :,:,:, spectral_indices[0] : spectral_indices[1] : 1]),
    #                 #               (0.5*device_size_lateral_um)**2, dot_dS = False, pauseDebug=False) / mu_0
    #                 # ))


    #             #
    #             # todo: do you need to change this if you weight by transmission? the figure of merit still has this weight, the transmission is just
    #             # guiding additional weights on top of this
    #             #
    #             max_intensity_weighting = max_intensity_by_wavelength[spectral_indices[0] : spectral_indices[1] : 1]
    #             total_weighting = max_intensity_weighting / weight_focal_plane_map[adj_src_idx]

    #             #
    #             # We need to properly account here for the current real and imaginary index
    #             # because they both contribute in the end to the real part of the gradient ΔFoM/Δε_r
    #             # in Eq. 5 of Lalau-Keraly paper https://doi.org/10.1364/OE.21.021693
    #             #
    #             dispersive_max_permittivity = dispersion_model.average_permittivity(dispersive_ranges_um[lookup_dispersive_range_idx])
    #             delta_real_permittivity = np.real(dispersive_max_permittivity - min_device_permittivity )
    #             delta_imag_permittivity = np.imag(dispersive_max_permittivity - min_device_permittivity )

    #             for spectral_idx in range(0, len( source_weight ) ):
    #                 # TODO: Performance weighting is likely the thing that tunes the FoM based on the performance thus far in the optimization
    #                 if weight_by_individual_wavelength:
    #                     gradient_performance_weight = performance_weighting[ spectral_indices[0] + spectral_idx ]

    #                     if do_rejection:
    #                         if weight_by_quadrant_transmission:
    #                             gradient_performance_weight *= softplus_prime( transmission_by_wavelength[ spectral_idx ] )
    #                         else:
    #                             gradient_performance_weight *= softplus_prime( intensity_fom_by_wavelength[ spectral_idx ] )
                            
    #                         # gradient_performance_weight *= softplus_prime( intensity_fom_by_wavelength[ spectral_idx ] )

    #                 if do_rejection:
    #                     gradient_performance_weight *= spectral_focal_plane_map_directional_weights[ adj_src_idx ][ spectral_idx ]

    #                 # See Eqs. (5), (6) of Lalau-Keraly et al: https://doi.org/10.1364/OE.21.021693
    #                 print(f'Spectral idx is: {spectral_idx}')
    #                 sys.stdout.flush()
    #                 gradient_component = np.sum(
    #                     (source_weight[spectral_idx] *                                                      # Amplitude of electric dipole at x_0
    #                         gradient_performance_weight / total_weighting[spectral_idx]) *                     # Performance weighting, Manual override weighting, and intensity normalization factors
    #                     adjoint_e_fields[xy_idx][:, :, :, :, spectral_indices[0] + spectral_idx] *          # E-field induced at x from electric dipole at x_0
    #                     forward_e_fields[pol_name][:, :, :, :, spectral_indices[0] + spectral_idx],         # E_old(x)
    #                     axis=0)                                                                             # Dot product

    #                 # Permittivity factor in amplitude of electric dipole at x_0
    #                 # Explanation: For two complex numbers, Re(z1*z2) = Re(z1)*Re(z2) + [-Im(z1)]*Im(z2)
    #                 real_part_gradient = np.real( gradient_component )
    #                 imag_part_gradient = -np.imag( gradient_component )
    #                 get_grad_density = delta_real_permittivity * real_part_gradient + delta_imag_permittivity * imag_part_gradient

    #                 xy_polarized_gradients[pol_name_to_idx] += get_grad_density

    # device_gradient = 2 * ( xy_polarized_gradients[0] + xy_polarized_gradients[1] )
    # print("Completed Step 3: Retrieved Adjoint Optimization results and calculated gradient.")
    # print(np.shape(device_gradient))
    # sys.stdout.flush()
    
    # TODO: Adapt the single FoM value to a sweep_value as in append_new_sweep()
    
    output_plot = {}
    r_vectors = []      # Variables
    f_vectors = []      # Functions
    
    lambda_vector = monitors_data['wavelength']
    r_vectors.append({'var_name': 'Wavelength',
                      'var_values': lambda_vector
                    })
    
    # f_vectors.append({'var_name': 'Intensity FoM, Quadrant 0, Wavelength index 0',
    #                   'var_values': intensity_fom_by_quadrant_by_wl[0, 0] * np.ones(np.shape(lambda_vector))
    #                 })
    
    f_vectors.append({'var_name': 'Intensity FoM',
                      'var_values': np.sum(intensity_fom_by_quadrant_by_wl, axis=None) * np.ones(np.shape(lambda_vector))
                    })
    
    f_vectors.append({'var_name': 'Mode Overlap FoM',
                      'var_values': np.sum(mode_overlap_fom_by_quadrant_by_wl, axis=None) * np.ones(np.shape(lambda_vector))
                    })
    
    f_vectors.append({'var_name': 'Intensity FoM, Quadrant 0',
                      'var_values': intensity_fom_by_quadrant_by_wl[0, :]
                    })
    
    f_vectors.append({'var_name': 'Intensity FoM, Quadrant 2',
                      'var_values': intensity_fom_by_quadrant_by_wl[2, :]
                    })
    
    f_vectors.append({'var_name': 'Mode Overlap FoM, Quadrant 0',
                      'var_values': mode_overlap_fom_by_quadrant_by_wl[0, :]
                    })
    
    f_vectors.append({'var_name': 'Mode Overlap FoM, Quadrant 2',
                      'var_values': mode_overlap_fom_by_quadrant_by_wl[2, :]
                    })
    
    if produce_spectrum:
        output_plot['r'] = r_vectors
        output_plot['f'] = f_vectors
        output_plot['title'] = 'Gradient Comparison'
        output_plot['const_params'] = const_parameters
        
        print('Generated gradient comparison spectrum.')
    
    sweep_values_fom = []
    num_sweep_values = len(f_vectors)
    sweep_values_fom = append_new_sweep(sweep_values_fom, f_vectors, num_sweep_values, statistics)
    
    print('Picked gradient comparison values to pass to variable sweep.')
    
    return output_plot, sweep_values_fom

#! Step 4 Functions

if not running_on_local_machine:	
    from matplotlib import font_manager
    font_manager._rebuild()
    fp = font_manager.FontProperties(fname=r"/central/home/ifoo/.fonts/Helvetica-Neue-Medium-Extended.ttf")
    print('Font name is ' + fp.get_name())
    plt.rcParams.update({'font.sans-serif':fp.get_name()}) 

plt.rcParams.update({'font.sans-serif':'Helvetica Neue',            # Change this based on whatever custom font you have installed
                     'font.weight': 'normal', 'font.size':20})              
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Helvetica Neue'
plt.rcParams['mathtext.it'] = 'Helvetica Neue:italic'

# mpl.rcParams['font.sans-serif'] = 'Helvetica Neue'
# mpl.rcParams['font.family'] = 'sans-serif'
marker_style = dict(linestyle='-', linewidth=2.2, marker='o', markersize=4.5)

def apply_common_plot_style(ax=None, plt_kwargs={}, showLegend=True):
    '''Sets universal axis properties for all plots. Function is called in each plot generating function.'''
    
    if ax is None:
        ax = plt.gca()
    
    # Creates legend    
    if showLegend:
        #ax.legend(prop={'size': 10})
        ax.legend(prop={'size': 10}, loc='center left', bbox_to_anchor=(1,.5))
    
    # Figure size
    fig_width = 8.0
    fig_height = fig_width*4/5
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(fig_width)/(r-l)
    figh = float(fig_height)/(t-b)
    ax.figure.set_size_inches(figw, figh, forward=True)
    
    # Minor Ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    # Y-Axis Exponent Repositioning
    # ax.get_yaxis().get_offset_text().set_position((0, 0.5))
    
    return ax

def enter_plot_data_1d(plot_config, fig=None, ax=None):
    '''Every plot function calls this function to actually put the data into the plot. The main input is plot_config, a dictionary that contains
    parameters controlling every single property of the plot that might be relevant to data.'''
    
    for plot_idx in range(0, len(plot_config['lines'])):
        data_line = plot_config['lines'][plot_idx]
        
        x_plot_data = data_line['x_axis']['factor'] * np.array(data_line['x_axis']['values']) + data_line['x_axis']['offset']
        y_plot_data = data_line['y_axis']['factor'] * np.array(data_line['y_axis']['values']) + data_line['y_axis']['offset']
        
        plt.plot(x_plot_data[data_line['cutoff']], y_plot_data[data_line['cutoff']],
                 color=data_line['color'], label=data_line['legend'], **data_line['marker_style'])
    
    
    plt.title(plot_config['title'])
    plt.xlabel(plot_config['x_axis']['label'])
    plt.ylabel(plot_config['y_axis']['label'])
    
    if plot_config['x_axis']['limits']:
        plt.xlim(plot_config['x_axis']['limits'])
    if plot_config['y_axis']['limits']:
        plt.ylim(plot_config['y_axis']['limits'])
    
    ax = apply_common_plot_style(ax, {})
    plt.tight_layout()
    
    # # plt.show(block=False)
    # # plt.show()
    
    return fig, ax

class BasicPlot():
    def __init__(self, plot_data):
        '''Initializes the plot_config variable of this class object and also the Plot object.'''
        self.r_vectors = plot_data['r']
        self.f_vectors = plot_data['f']
        
        self.fig, self.ax = plt.subplots()
        self.plot_config = {'title': None,
                    'x_axis': {'label':'', 'limits':[]},
                    'y_axis': {'label':'', 'limits':[]},
                    'lines': []
                    }
    
    def append_line_data(self, plot_colors=None, plot_labels=None):
        '''Appends all line data stored in the f_vectors to the plot_config. Here we can assign colors and legend labels all at once.'''
        
        for plot_idx in range(0, len(self.f_vectors)):
            line_data = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
                        'y_axis': {'values': None, 'factor': 1, 'offset': 0},
                        'cutoff': None,
                        'color': None,
                        'alpha': 1.0,
                        'legend': None,
                        'marker_style': marker_style
                    }
            
            line_data['x_axis']['values'] = self.r_vectors[0]['var_values']
            line_data['y_axis']['values'] = self.f_vectors[plot_idx]['var_values']
            line_data['cutoff'] = slice(0, len(line_data['x_axis']['values']))
            
            if plot_colors is not None:
                line_data['color'] = plot_colors[plot_idx]
            if plot_labels is not None:
                line_data['legend'] = plot_labels[plot_idx]
            
            self.plot_config['lines'].append(line_data)
    
    def alter_line_property(self, key, new_value, axis_str=None):
        '''Alters a property throughout all of the line data.
        If new_value is an array it will change all of the values index by index; otherwise it will blanket change everything.
        The axis_str argument is to cover multiple-level dictionaries.'''
        
        for line_idx, line_data in enumerate(self.plot_config['lines']):
            if axis_str is None:
                if not isinstance(new_value, list):
                    line_data[key] = new_value
                else:
                    line_data[key] = new_value[line_idx]
            else:
                if not isinstance(new_value, list):
                    line_data[axis_str][key] = new_value
                else:
                    line_data[axis_str][key] = new_value[line_idx]
    
    def assign_title(self, title_string, job_idx):
        '''Replaces title of plot.'''
        
        for sweep_param_value in list(sweep_parameters.values()):
            current_value = sweep_param_value['var_values'][job_idx[2][0]]
            optimized_value = sweep_param_value['var_values'][sweep_param_value['peakInd']]
            title_string += '\n' + sweep_param_value['var_name'] + f': Current Value {current_value}, Optimized for {optimized_value}'
            
        self.plot_config['title'] = title_string
    
    def assign_axis_labels(self, x_label_string, y_label_string):
        '''Replaces axis labels of plot.'''
        
        self.plot_config['x_axis']['label'] = x_label_string
        self.plot_config['y_axis']['label'] = y_label_string
        
    def export_plot_config(self, plot_subfolder_name, job_idx, close_plot=True):
        '''Creates plot using the plot config, and then exports.'''
        self.fig, self.ax = enter_plot_data_1d(self.plot_config, self.fig, self.ax)
        
        if not os.path.isdir(plot_directory_location + '/' + plot_subfolder_name):
            os.makedirs(plot_directory_location + '/' + plot_subfolder_name)
        plt.savefig(plot_directory_location + '/' + plot_subfolder_name\
            + '/' + isolate_filename(job_names[job_idx]).replace('.fsp', '') + '.png',
            bbox_inches='tight')
        
        export_msg_string = (plot_subfolder_name.replace("_", " ")).title()
        print('Exported: ' + export_msg_string)
        sys.stdout.flush()
        
        if close_plot:
            plt.close()


class SpectrumPlot(BasicPlot):
    def __init__(self, plot_data):
        '''Initializes the plot_config variable of this class object and also the Plot object.'''
        super(SpectrumPlot, self).__init__(plot_data)
    
    def append_line_data(self, plot_colors=None, plot_labels=None):
        super(SpectrumPlot, self).append_line_data(plot_colors, plot_labels)
        self.alter_line_property('factor', 1e6, axis_str='x_axis')
    
    
class SweepPlot(BasicPlot):
    def __init__(self, plot_data, slice_coords):
        '''Initializes the plot_config variable of this class object and also the Plot object.'''
        super(SweepPlot, self).__init__(plot_data)
        
        # Whichever entry is a slice means that variable in r_vectors is the x-axis of this plot
        r_vector_value_idx = [index for (index , item) in enumerate(slice_coords) if isinstance(item, slice)][0]
        self.r_vectors = list(plot_data['r'].values())[r_vector_value_idx]
        #f_vectors = plot_data['f']              # This is a list of N-D arrays, containing the function values for each parameter combination
    
    
    def append_line_data(self, slice_coords, plot_stdDev, plot_colors=None, plot_labels=None, normalize_against_max=False):
        '''Appends all line data stored in the f_vectors to the plot_config. Here we can assign colors and legend labels all at once.
        Overwriting the method in BasePlot.'''
        
        for plot_idx in range(0, len(self.f_vectors)-0):
            line_data = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
                        'y_axis': {'values': None, 'factor': 1, 'offset': 0},
                        'cutoff': None,
                        'color': None,
                        'alpha': 1.0,
                        'legend': None,
                        'marker_style': marker_style
                    }
        
            y_plot_data = self.f_vectors[plot_idx]['var_values']
            y_plot_data = np.reshape(y_plot_data, (len(y_plot_data), 1))
            y_plot_data = y_plot_data[tuple(slice_coords)]
            
            normalization_factor = np.max(y_plot_data) if normalize_against_max else 1
            line_data['y_axis']['values'] = y_plot_data / normalization_factor
            line_data['x_axis']['values'] = self.r_vectors['var_values']
            # line_data['cutoff'] = slice(0, len(line_data['x_axis']['values']))
            line_data['cutoff'] = slice(0 + cutoff_1d_sweep_offset[0], len(line_data['x_axis']['values']) + cutoff_1d_sweep_offset[1])
            
            line_data['color'] = plot_colors[plot_idx]
            line_data['legend'] = plot_labels[plot_idx]
            
            self.plot_config['lines'].append(line_data)
            
            
            if plot_stdDev and self.f_vectors[plot_idx]['statistics'] in ['mean']:
                line_data_2 = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
                        'y_axis': {'values': None, 'factor': 1, 'offset': 0},
                        'cutoff': None,
                        'color': None,
                        'alpha': 0.3,
                        'legend': '_nolegend_',
                        'marker_style': dict(linestyle='-', linewidth=1.0)
                    }
            
                line_data_2['x_axis']['values'] = self.r_vectors['var_values']
                line_data_2['y_axis']['values'] = (y_plot_data + self.f_vectors[plot_idx]['var_stdevs']) / normalization_factor
                # line_data_2['cutoff'] = slice(0, len(line_data_2['x_axis']['values']))
                line_data_2['cutoff'] = slice(0 + cutoff_1d_sweep_offset[0], len(line_data_2['x_axis']['values']) + cutoff_1d_sweep_offset[1])
                
                colors_so_far = plt.rcParams["axes.prop_cycle"].by_key()["color"]
                line_data_2['color'] = colors_so_far[-1]
                
                self.plot_config['lines'].append(line_data_2)
                
                line_data_3 = copy.deepcopy(line_data_2)
                line_data_3['y_axis']['values'] = (y_plot_data - self.f_vectors[plot_idx]['var_stdevs']) / normalization_factor
                self.plot_config['lines'].append(line_data_3)
        
    def assign_title(self, title_string):
        '''Replaces title of plot. 
        Overwriting the method in BasePlot.'''
        
        for sweep_param_value in list(sweep_parameters.values()):
            optimized_value = sweep_param_value['var_values'][sweep_param_value['peakInd']]
            title_string += '\n' + sweep_param_value['var_name'] + f': Optimized at {optimized_value}'
        self.plot_config['title'] = title_string
    
    def assign_axis_labels(self, slice_coords, y_label_string):
        '''Replaces axis labels of plot.
        Overwriting the method in BasePlot.'''
        
        sweep_variable_idx = [index for (index, item) in enumerate(slice_coords) if type(item) is slice][0]
        self.plot_config['x_axis']['label'] = list(sweep_parameters.values())[sweep_variable_idx]['var_name']
        self.plot_config['y_axis']['label'] = y_label_string
    
    def export_plot_config(self, file_type_name, slice_coords, plot_subfolder_name='', close_plot=True):
        '''Creates plot using the plot config, and then exports.
        Overwriting the method in BasePlot.'''
        
        self.fig, self.ax = enter_plot_data_1d(self.plot_config, self.fig, self.ax)
        
        if not os.path.isdir(plot_directory_location + '/' + plot_subfolder_name):
            os.makedirs(plot_directory_location + '/' + plot_subfolder_name)
        plt.savefig(plot_directory_location + '/' + plot_subfolder_name\
            + '/' + file_type_name + create_parameter_filename_string(slice_coords) + '.png',
            bbox_inches='tight')
        
        export_msg_string = (file_type_name.replace("_", " ")).title()
        print('Exported: ' + export_msg_string + create_parameter_filename_string(slice_coords))
        sys.stdout.flush()
        
        if close_plot:
            plt.close()
            
def plot_gradient_comparison_sweep_1d(plot_data, slice_coords, plot_stdDev = True):
    '''Produces a 1D focal region power plot using given coordinates to slice the N-D array of data contained in plot_data.
    X-axis i.e. sweep variable is given by the index of the entry of type 'slice' in the slice_coords.'''
    
    def NormalizeData(data):
        def normalize(values):
            # Normalizes a 1D vector to between 0 and 1
            return (values - np.min(values)) / (np.max(values) - np.min(values))
        
        return normalize(data)

    projects_directory_subfolder = "/intensity_fom"

    # Load device gradient
    device_gradient = np.load(
        projects_directory_location + projects_directory_subfolder + "/device_gradient.npy")
    device_gradient = device_gradient[0:device_voxels_lateral, 0:device_voxels_lateral, 0:num_vertical_layers]
    device_gradient_vals = []
    
    for idx, x in np.ndenumerate(np.zeros(sweep_parameters_shape)):
        for t_idx, p_idx in enumerate(idx):
            variable = list(sweep_parameters.values())[t_idx]
            variable_name = variable['var_name']
            variable_value = variable['var_values'][p_idx]
            axis = 'z'      #! Hard-coded
            
            # Get point coords
            coord_idx = np.array([len(bayer_filter_region_x)/2, len(bayer_filter_region_y)/2, len(bayer_filter_region_z)/2], dtype=int)
            # coord_idx = np.array([0,0,0], dtype=int)
            
            if axis in ['x']:
                bayer_filter_axis = bayer_filter_region_x
            elif axis in ['y']:
                bayer_filter_axis = bayer_filter_region_y
            else:
                bayer_filter_axis = bayer_filter_region_z
                
            coord_value = bayer_filter_axis[0] + variable_value*(bayer_filter_axis[-1] - bayer_filter_axis[0])
            
            if axis in ['x']:
                coord_idx[0] = np.argmin(np.abs(np.array(bayer_filter_axis) - coord_value))
            elif axis in ['y']:
                coord_idx[1] = np.argmin(np.abs(np.array(bayer_filter_axis) - coord_value))
            else:
                coord_idx[2] = np.argmin(np.abs(np.array(bayer_filter_axis) - coord_value))
            
            device_gradient_vals.append(device_gradient[tuple(x for x in coord_idx)])
            print(f'Coord_idx is {coord_idx}')
            print(device_gradient[tuple(x for x in coord_idx)])
    #device_gradient_vals = np.flip(device_gradient_vals)
    
    # Load f(ε)
    # unperturbed_figure_of_merit_evolution = np.load(
    #          projects_directory_location + projects_directory_subfolder + "/figure_of_merit.npy")
    
    unperturbed_intensity_fom_by_wavelength_evolution = np.load(
             projects_directory_location + "/intensity_fom" + "/intensity_fom_by_wavelength.npy")
    unperturbed_intensity_fom_evolution = np.sum(unperturbed_intensity_fom_by_wavelength_evolution)
    print(unperturbed_intensity_fom_evolution)
    
    unperturbed_mode_overlap_fom_by_wavelength_evolution = np.load(
             projects_directory_location + "/mode_overlap_fom" + "/mode_overlap_fom_by_wavelength.npy")
    unperturbed_mode_overlap_fom_evolution = np.sum(unperturbed_mode_overlap_fom_by_wavelength_evolution)
    print(unperturbed_mode_overlap_fom_evolution)
    
    sp = SweepPlot(plot_data, slice_coords)
    # Before: was f(ε+δa)
    sp.f_vectors[0]['var_values'] = (sp.f_vectors[0]['var_values'] - unperturbed_intensity_fom_evolution)/coord_permittivity_perturbation
    sp.f_vectors[2]['var_values'] = (sp.f_vectors[2]['var_values'] - unperturbed_intensity_fom_evolution/4)/coord_permittivity_perturbation
    sp.f_vectors[3]['var_values'] = (sp.f_vectors[3]['var_values'] - unperturbed_intensity_fom_evolution/4)/coord_permittivity_perturbation
    # sp.f_vectors[0]['var_values'] = (sp.f_vectors[0]['var_values'])/coord_permittivity_perturbation
    
    sp.f_vectors[1]['var_values'] = (sp.f_vectors[1]['var_values'] - unperturbed_mode_overlap_fom_evolution)/coord_permittivity_perturbation
    sp.f_vectors[4]['var_values'] = (sp.f_vectors[4]['var_values'] - unperturbed_mode_overlap_fom_evolution/4)/coord_permittivity_perturbation
    sp.f_vectors[5]['var_values'] = (sp.f_vectors[5]['var_values'] - unperturbed_mode_overlap_fom_evolution/4)/coord_permittivity_perturbation
    # Now: is (f(ε+δa) - f(ε))/δa
    
    # Duplicate and add another sweep - the device gradient
    import copy
    sp.f_vectors.append(copy.deepcopy(sp.f_vectors[0]))
    sp.f_vectors[-1]['var_values'] = device_gradient_vals
    
    # Rescale both data sets
    for fvec_idx in range(0, len(sp.f_vectors)):
        sp.f_vectors[fvec_idx]['var_values'] = NormalizeData(sp.f_vectors[fvec_idx]['var_values'])
    # sp.f_vectors[0]['var_values'] = NormalizeData(sp.f_vectors[0]['var_values'])
    # sp.f_vectors[-1]['var_values'] = NormalizeData(sp.f_vectors[-1]['var_values'])
    # sp.f_vectors[0]['var_values'] = np.reshape(sp.f_vectors[0]['var_values'], np.shape(sp.f_vectors[-1]['var_values']))
    
    replace_f_vectors = copy.deepcopy(sp.f_vectors)
    sp.f_vectors = [replace_f_vectors[2], replace_f_vectors[-1]]
    #sp.f_vectors = [replace_f_vectors[1], replace_f_vectors[-1]]
    
    plot_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    
    # plot_labels = ['Intensity FoM Gradient', 'Mode Overlap FoM Gradient',
    #                'Int. Quad. 0','Int. Quad. 2', 
    #                'MO Quad. 0', 'MO Quad. 2', 
    #                'Algorithmic Gradient']
    plot_labels = ['Mode Overlap FoM Gradient', 'Algorithmic Gradient']
    sp.append_line_data(slice_coords, plot_stdDev, plot_colors, plot_labels)
    
    sp.assign_title('Gradient Comparison - Mode Overlap FoM:')
    sp.assign_axis_labels(slice_coords, 'Derivative (Rescaled)')
    
    sp.export_plot_config('gradient_comp_intensity_fom_quad0_', slice_coords)
    
    return sp.fig, sp.ax

          
# #
# #! Set up some numpy arrays to handle all the data we will pull out of the simulation.
# # These are just to keep track of the arrays we create. We re-initialize these right before their respective loops
# #
mode_e_fields = {}                          # Records E-field profiles at focal plane quadrants 
mode_h_fields = {}                          # Records H-field profiles at focal plane quadrants

# monitors_data = {}

#
# Set up queue for parallel jobs
#

jobs_queue = queue.Queue()

def add_job(job_name, queue_in):
    full_name = projects_directory_location + "/" + job_name
    fdtd_hook.save(full_name)
    queue_in.put(full_name)

    return full_name

def run_jobs(queue_in):
    small_queue = queue.Queue()

    while not queue_in.empty():
        
        for node_idx in range(0, num_nodes_available):
            # this scheduler schedules jobs in blocks of num_nodes_available, waits for them ALL to finish, and then schedules the next block
            # TODO: write a scheduler that enqueues a new job as soon as one finishes
            # not urgent because for the most part, all jobs take roughly the same time
            if queue_in.qsize() > 0:
                small_queue.put(queue_in.get())

        run_jobs_inner(small_queue)

def run_jobs_inner(queue_in):
    processes = []
    job_idx = 0
    while not queue_in.empty():
        get_job_path = queue_in.get()
        
        print(f'Node {cluster_hostnames[job_idx]} is processing file at {get_job_path}.')
        sys.stdout.flush()

        process = subprocess.Popen(
            [
                '/home/gdrobert/Develompent/adjoint_lumerical/inverse_design/run_proc.sh',
                cluster_hostnames[job_idx],
                get_job_path
            ]
        )
        processes.append(process)

        job_idx += 1
        
    sys.stdout.flush()

    completed_jobs = [0 for i in range(0, len(processes))]
    while np.sum(completed_jobs) < len(processes):
        for job_idx in range(0, len(processes)):
            sys.stdout.flush()
            
            if completed_jobs[job_idx] == 0:

                poll_result = processes[job_idx].poll()
                if not(poll_result is None):
                    completed_jobs[job_idx] = 1

        time.sleep(1)

# Choose the dispersion model for this optimization
dispersion_model = DevicePermittivityModel()

restart = False
if restart:
    current_job_idx = (0,0,0)
    
#
# Run the simulation
#

start_epoch = 0
for epoch in range(start_epoch, num_epochs):
    fdtd_hook.switchtolayout()
    
    start_iter = 0
    for iteration in range(start_iter, num_iterations_per_epoch):
        print("Working on epoch " + str(epoch) +
              " and iteration " + str(iteration))
        
        job_names = {}

        fdtd_hook.switchtolayout()
        
        
        #
        # Step 1: Import sweep parameters, assemble jobs corresponding to each value of the sweep. Edit the environment of each job accordingly.
        # Queue and run parallel, save out.
        #
        
        if start_from_step == 0 or start_from_step == 1:
            if start_from_step == 1:
                load_backup_vars()
            
            print("Beginning Step 1: Running Sweep Jobs")
            sys.stdout.flush()


            for idx, x in np.ndenumerate(np.zeros(sweep_parameters_shape)):
                parameter_filename_string = create_parameter_filename_string(idx)
                current_job_idx = idx
                print(f'Creating Job: Current parameter index is {current_job_idx}')
                
                for t_idx, p_idx in enumerate(idx):
                    
                    variable = list(sweep_parameters.values())[t_idx]
                    variable_name = variable['var_name']
                    variable_value = variable['var_values'][p_idx]
                    
                    globals()[variable_name] = variable_value
                    
                    fdtd_hook.switchtolayout()
                    # depending on what variable is, edit Lumerical
                    
                    if variable_name in ['angle_theta']:
                        change_source_theta(variable_value, angle_phi)
                    elif variable_name in ['coord_perturbed']:
                        coord_permittivity_perturbation = 2.0
                        change_permittivity_point(variable_value, coord_permittivity_perturbation)
                        
                
                
                for dispersive_range_idx in range( 0, num_dispersive_ranges ):
                    # dispersive_max_permittivity = dispersion_model.average_permittivity( dispersive_ranges_um[ dispersive_range_idx ] )
                    # disperesive_max_index = index_from_permittivity( dispersive_max_permittivity )

                    # fdtd_hook.switchtolayout()

                    # cur_permittivity = min_device_permittivity + ( dispersive_max_permittivity - min_device_permittivity ) * cur_density
                    # cur_index = index_from_permittivity( cur_permittivity )

                    # fdtd_hook.select( 'design_import' )
                    # fdtd_hook.importnk2( cur_index, bayer_filter_region_x, 
                    #                     bayer_filter_region_y, bayer_filter_region_z )

                    for xy_idx in range(0, 2):
                        disable_all_sources()

                        fdtd_hook.select( forward_sources[xy_idx]['name'] )
                        fdtd_hook.set( 'enabled', 1 )
                        
                        for adj_src_idx in range(0, num_adjoint_sources):
                            fdtd_hook.select( adjoint_sources[adj_src_idx][xy_idx]['name'] )
                            fdtd_hook.set('z', (-focal_length_um + mesh_spacing_um*5)*1e-6)
                            fdtd_hook.set('direction', 'Backward')
                        
                        job_name = 'sweep_' + parameter_filename_string + \
                            '_fwd_' + str( xy_idx ) + "_" + str( dispersive_range_idx ) + '.fsp'
                        fdtd_hook.save(projects_directory_location + "/optimization.fsp")
                        print('Saved out ' + job_name)
                        job_names[('sweep', idx, 'forward', xy_idx, dispersive_range_idx)
                                    ] = add_job(job_name, jobs_queue)
                        
                    
                    for xy_idx in range(0,2):
                        # Easier to initialize in a separate loop
                        mode_e_fields[xy_names[xy_idx]] = [
                            None for idx in range( 0, num_focal_spots ) ]
                        mode_h_fields[xy_names[xy_idx]] = [
                            None for idx in range( 0, num_focal_spots ) ]
                    
                    for adj_src_idx in range(0, num_adjoint_sources):
                        for xy_idx in range(0, 2):
                            disable_all_sources()
                            
                            fdtd_hook.select( adjoint_sources[adj_src_idx][xy_idx]['name'] )
                            fdtd_hook.set( 'enabled', 1 )
                            fdtd_hook.set('z', (-focal_length_um + mesh_spacing_um*0)*1e-6)
                            fdtd_hook.set('direction', 'Backward')       # Correct orientation
                            # fdtd_hook.set('direction', 'Forward')        # Wrong, but all this does is add a minus sign and phase that should be 0 at the injection plane anyway

                            # for not_adj_src_idx in range(0, num_adjoint_sources):
                            # # if not_adj_src_idx != adj_src_idx:
                            #     fdtd_hook.select( adjoint_sources[not_adj_src_idx][xy_idx]['name'] )
                            #     fdtd_hook.set( 'enabled', 1 )
                                
                            #     # Aim down and push above focal plane a bit
                            #     fdtd_hook.set('z', (-focal_length_um + mesh_spacing_um*5)*1e-6)
                            #     fdtd_hook.set('direction', 'Backward')
                            
                            # Pull out adjoint source, user-defined mode data: Em, Hm
                            mode_e_fields[xy_names[xy_idx]][adj_src_idx] = get_efield_mode(adjoint_sources[adj_src_idx][xy_idx]['name'])
                            print(f'Accessed mode E-field data over focal quadrant {adj_src_idx} for polarization {xy_idx}.')
                            
                            
                            mode_h_fields[xy_names[xy_idx]][adj_src_idx] = get_hfield_mode(adjoint_sources[adj_src_idx][xy_idx]['name'])
                            print(f'Accessed mode H-field data over focal quadrant {adj_src_idx} for polarization {xy_idx}.')
                            
                            # Set direction back to upwards, restore z position to focal plane
                            # for not_adj_src_idx in range(0, num_adjoint_sources):
                            #     if not_adj_src_idx != adj_src_idx:
                            #         fdtd_hook.select( adjoint_sources[not_adj_src_idx][xy_idx]['name'] )
                            #         fdtd_hook.set('z', adjoint_vertical_um * 1e-6)
                            #         fdtd_hook.set('direction', 'Forward')
                            
                            fdtd_hook.select( adjoint_sources[adj_src_idx][xy_idx]['name'] )
                            fdtd_hook.set('z', adjoint_vertical_um * 1e-6)
                            fdtd_hook.set('direction', 'Forward')
                            
                            # Test the denominator of the FoM
                            # a = mode_e_fields['x'][0]
                            # b = mode_h_fields['x'][0]
                            # c = poynting_flux(a,np.conj(b),1)

                            ## job_name = 'adjoint_job_' + str( adj_src_idx ) + '_' + str( xy_idx ) + '_' + str( dispersive_range_idx ) + '.fsp'
                            ## fdtd_hook.save( projects_directory_location + "/optimization.fsp" )
                            ## job_names[ ( 'adjoint', adj_src_idx, xy_idx, dispersive_range_idx ) ] = add_job( job_name, jobs_queue )
                            
                            # job_name = 'sweep_' + parameter_filename_string + \
                            #             '_adj_' + str( adj_src_idx ) + '_' + \
                            #             str( xy_idx ) + "_" + str( dispersive_range_idx ) + '.fsp'
                            # fdtd_hook.save(projects_directory_location + "/optimization.fsp")
                            # print('Saved out ' + job_name)
                            # job_names[('sweep', idx, 'adjoint', adj_src_idx, xy_idx, dispersive_range_idx)
                            #             ] = add_job(job_name, jobs_queue)
                    

            backup_all_vars()        
            run_jobs(jobs_queue)
            
            print("Completed Step 1: Forward Optimization Jobs Completed.")
            sys.stdout.flush()
            
            backup_all_vars()
            
            
        #
        # Step 2: Extract fields from each monitor, pass as inputs to premade functions in order to extract plot data. Data is typically a
        # spectrum for each function (e.g. sorting efficiency) and also the peak value across that spectrum.
        #
        
        if start_from_step == 0 or start_from_step == 2:
            if start_from_step == 2:
                load_backup_vars()
                
            print("Beginning Step 2: Extracting Fields from Monitors.")
            sys.stdout.flush()
            
            jobs_data = {}
            for idx, x in np.ndenumerate(np.zeros(sweep_parameters_shape)):
                parameter_filename_string = create_parameter_filename_string(idx)
                current_job_idx = idx
                print(f'----- Accessing Job: Current parameter index is {current_job_idx}')
                
                # Load each individual job
                for dispersive_range_idx in range( 0, num_dispersive_ranges ):
                    for xy_idx in range(0, 2): # range(0,2) if considering source with both polarizations
                        if start_from_step == 0:
                            fdtd_hook.load(
                                job_names[('sweep', idx, 'forward', xy_idx, dispersive_range_idx)])
                            
                        else:
                            # Extract filename, directly declare current directory address
                            
                            # TODO: Undo this bit when no longer build-testing
                            if running_on_local_machine:
                                actual_directory = r'C:\Users\Ian\Dropbox\Caltech\Faraon Group\Simulations\Mode Overlap FoM\fom_dev\fom_gradient_test\ares_mode_overlap_fom_check_gradient'
                                actual_filepath = convert_root_folder(job_names[('sweep', idx, 'forward', xy_idx, dispersive_range_idx)],
                                                    actual_directory)
                            else:
                                actual_filepath = job_names[('sweep', idx, 'forward', xy_idx, dispersive_range_idx)]
                            print(f'Loading: {isolate_filename(actual_filepath)}')
                            sys.stdout.flush()
                            fdtd_hook.load(actual_filepath)
                            
                        monitors_data = {}
                        # Add wavelength and sourcepower information, common to every monitor in a job
                        monitors_data['wavelength'] = np.divide(3e8, fdtd_hook.getresult('focal_monitor_0','f'))
                        monitors_data['sourcepower'] = get_sourcepower()
                        
                        # Go through each individual monitor and extract respective data
                        for monitor in monitors:
                            if monitor['name'] not in ['src_spill_monitor']:
                                monitor_data = {}
                                monitor_data['name'] = monitor['name']
                                monitor_data['save'] = monitor['save']
                                
                                if monitor['enabled'] and monitor['getFromFDTD']:
                                    mn_nm = monitor['name']
                                    print(f'Getting data from {mn_nm}')
                                    # monitor_data['E'] = get_efield(monitor['name'])
                                    # print('Getting E-field')
                                    # #monitor_data['E_alt'] = get_E
                                    # monitor_data['P'] = get_overall_power(monitor['name'])
                                    # print('Getting H-field')
                                    # monitor_data['T'] = get_transmission_magnitude(monitor['name'])
                                    # print('Getting transmission')
                                    
                                    if monitor['name'] in ['spill_plane_monitor', 'scatter_plane_monitor',
                                                        'transmission_monitor_0','transmission_monitor_1','transmission_monitor_2','transmission_monitor_3',
                                                        'transmission_focal_monitor_',
                                                        'device_xsection_monitor_0', 'device_xsection_monitor_1', 'device_xsection_monitor_2', 
                                                        'device_xsection_monitor_3', 'device_xsection_monitor_4', 'device_xsection_monitor_5',
                                                        'design_efield_monitor']:
                                        monitor_data['coords'] = {'x': fdtd_hook.getresult(monitor['name'],'x'),
                                                                'y': fdtd_hook.getresult(monitor['name'],'y'),
                                                                'z': fdtd_hook.getresult(monitor['name'],'z')}
                                    
                                    # if monitor['name'] in ['device_xsection_monitor_0', 'device_xsection_monitor_1', 'device_xsection_monitor_2', 
                                    #                         'device_xsection_monitor_3', 'device_xsection_monitor_4', 'device_xsection_monitor_5',
                                    #                         'design_efield_monitor']:
                                    #     monitor_data['E_real'] = get_efield(monitor['name'])
                                    
                                monitors_data[monitor['name']] = monitor_data
                        
                        # # # Load also the corresponding job where the device is disabled, to measure spill power
                        # if start_from_step == 0:
                        #     fdtd_hook.load(
                        #         job_names[('nodev', xy_idx, idx)])
                        # else:
                        #     # Extract filename, directly declare current directory address
                        #     fdtd_hook.load(
                        #         convert_root_folder(job_names[('nodev',xy_idx,idx)],
                        #                             projects_directory_location)
                        #     )
                        
                        # # Go through each individual monitor and extract respective data
                        # for monitor in monitors:
                        #     if monitor['name'] in ['incident_aperture_monitor',
                        #                            'src_spill_monitor']:
                        #         monitor_data = {}
                        #         monitor_data['name'] = monitor['name']
                        #         monitor_data['save'] = monitor['save']
                                
                        #         if monitor['enabled'] and monitor['getFromFDTD']:
                        #             monitor_data['E'] = get_efield_magnitude(monitor['name'])
                        #             #monitor_data['E_alt'] = get_E
                        #             monitor_data['P'] = get_overall_power(monitor['name'])
                        #             monitor_data['T'] = get_transmission_magnitude(monitor['name'])
                                    
                        #             if monitor['name'] in ['spill_plane_monitor', 'scatter_plane_monitor',
                        #                                 'transmission_monitor_0','transmission_monitor_1','transmission_monitor_2','transmission_monitor_3',
                        #                                 'transmission_focal_monitor_']:
                        #                 monitor_data['coords'] = {'x': fdtd_hook.getresult(monitor['name'],'x'),
                        #                                         'y': fdtd_hook.getresult(monitor['name'],'y'),
                        #                                         'z': fdtd_hook.getresult(monitor['name'],'z')}
                                    
                        #         monitors_data[monitor['name']] = monitor_data
                        
                        
                        # Save out to a dictionary, using job_names as keys
                        # jobs_data[job_names[('sweep', xy_idx, idx)]] = monitors_data
                        jobs_data[job_names[('sweep', idx, 'forward', xy_idx, dispersive_range_idx)]] = monitors_data
            
            
            print("Completed Step 2: Extracted all monitor fields.")
            sys.stdout.flush()
            
            backup_all_vars()
            
        #
        # Step 3: For each job / parameter combination, create the relevant plot data as a dictionary.
        # The relevant plot types are all stored in sweep_settings.json
        #
    
        if start_from_step == 0 or start_from_step == 3:
            if start_from_step == 3:
                load_backup_vars()
                
            print("Beginning Step 3: Gathering Function Data for Plots")
            sys.stdout.flush()
    
            plots_data = {}
            
            for idx, x in np.ndenumerate(np.zeros(sweep_parameters_shape)):
                print(f'----- Gathering Plot Data: Current parameter index is {idx}')
                
                # Load each individual job
                for xy_idx in range(0, 1): # range(0,2) if considering source with both polarizations
                    
                    monitors_data = jobs_data[job_names[('sweep', idx, 'forward', xy_idx, dispersive_range_idx)]]
            
                    plot_data = {}
                    plot_data['wavelength'] = monitors_data['wavelength']
                    plot_data['sourcepower'] = monitors_data['sourcepower']
                    
                    for plot in plots:
                        if plot['enabled']:
                            
                            if plot['name'] in ['sorting_efficiency']:
                                # plot_data[plot['name']+'_spectrum'], plot_data[plot['name']+'_sweep'] = \
                                #     generate_sorting_eff_spectrum(monitors_data, plot['generate_plot_per_job'], add_opp_polarizations=True)
                                continue
                            
                            elif plot['name'] in ['gradient_comparison']:
                                plot_data[plot['name']+'_spectrum'], plot_data[plot['name']+'_sweep'] = \
                                    generate_gradient_comparison_spectrum(monitors_data, plot['generate_plot_per_job'], idx)
                                
                    plots_data[job_names[('sweep', idx, 'forward', xy_idx, dispersive_range_idx)]] = plot_data
                        
                        
            print("Completed Step 3: Processed and saved relevant plot data.")
            sys.stdout.flush()
        
            # TODO: Undo the link between Step 3 and 4
            backup_all_vars()
        
        #
        # Step 4: Plot Sweeps across Jobs
        #
        
        if start_from_step == 0 or start_from_step == 4:
            if start_from_step == 4:
                load_backup_vars()
            
            startTime = time.time()
            
            print("Beginning Step 4: Extracting Sweep Data for Each Plot, Creating Plots and Saving Out")
            sys.stdout.flush()
            
            if 'plots_data' not in globals():
                print('*** Warning! Plots Data from previous step not present. Check your save state.')
                sys.stdout.flush()
                
            if not os.path.isdir(plot_directory_location):
                os.mkdir(plot_directory_location)
            
            #* Here a sweep plot means data points have been extracted from each job (each representing a value of a sweep parameter)
            #* and we are going to combine them all into a single plot.
            
            # Create dictionary to hold all the sweep plots
            sweep_plots = {}
            for plot in plots:
                # Each entry in the plots section of sweep_settings.json should have a sweep plot created for it
                
                # Initialize the dictionary to hold output plot data:
                output_plot = {}
                if plot['name'] not in ['Enorm_focal_plane_image'] and plot['enabled']:
                    
                    r_vectors = sweep_parameters
                    
                    f_vectors = []
                    job_name = list(job_names.values())[0]
                    # How many different things are being tracked over the sweep?
                    for f_idx in range(0, len(plots_data[job_name][plot['name']+'_sweep'])):
                        f_vectors.append({'var_name': plot['name'] + '_' + str(f_idx),
                                          'var_values': np.zeros(sweep_parameters_shape),
                                          'var_stdevs': np.zeros(sweep_parameters_shape),
                                          'statistics': ''
                                          })
                    
                    for xy_idx in range(0, 1): # range(0,2) if considering source with both polarizations    
                        for idx, x in np.ndenumerate(np.zeros(sweep_parameters_shape)):
                            
                            # if running_on_local_machine:
                            #     job_name = job_names[('sweep', xy_idx, idx)]
                            # else:
                            #     job_name = job_names[('sweep', idx, 'forward', xy_idx, dispersive_range_idx)]
                                
                            job_name = job_names[('sweep', idx, 'forward', xy_idx, dispersive_range_idx)]
                            f_values = plots_data[job_name][plot['name']+'_sweep']
                            
                            for f_idx, f_value in enumerate(f_values):
                                f_vectors[f_idx]['var_values'][idx] = f_value['value']
                                f_vectors[f_idx]['var_stdevs'][idx] = f_value['std_dev']
                                f_vectors[f_idx]['statistics'] = f_value['statistics']

                    output_plot['r'] = r_vectors
                    output_plot['f'] = f_vectors
                    output_plot['title'] = plot['name']+'_sweep'
            
                    sweep_plots[plot['name']] = output_plot
            plots_data['sweep_plots'] = sweep_plots
            
            
            #* We now have data for each and every plot that we are going to export.
            
            #
            # Step 4.1: Go through each job and export the spectra for that job
            #
            
            #
            # Step 4.2: Go through all the sweep plots and export each one
            #
            
            for plot in plots:
                if plot['enabled']:
                        
                    if plot['name'] in ['sorting_efficiency']:
                        # Here there is the most variability in what the user might wish to plot, so it will have to be hardcoded for the most part.
                        # plot_function_sweep(plots_data['sweep_plots'][plot['name']], 'th', plot_sorting_eff_sweep_1d.__name__)
                        # plot_sorting_eff_sweep(plots_data['sweep_plots'][plot['name']], 'phi')
                        # plot_sorting_eff_sweep_1d(plots_data['sweep_plots'][plot['name']], [slice(None), 0])
                        continue
                            
                    elif plot['name'] in ['gradient_comparison']:
                        plot_gradient_comparison_sweep_1d(plots_data['sweep_plots'][plot['name']], [slice(None), 0], plot_stdDev = False)


            backup_var('plots_data', os.path.basename(python_src_directory))
            backup_all_vars()
            
                        
            print("Completed Step 4: Created and exported all plots and plot data.")
            sys.stdout.flush()
            
            endtime = time.time()
            print(f'Step 4 took {(endtime-startTime)/60} minutes.')
               
print('Reached end of file. Code completed.')