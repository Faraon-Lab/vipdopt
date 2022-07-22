import os
import shutil
import sys
import shelve
from threading import activeCount

# Add filepath to default path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from LumProcParameters import *
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
    num_nodes_available = int( sys.argv[ 1 ] )  # should equal --nodes in batch script. Could also get this from len(cluster_hostnames)
    num_cpus_per_node = 8
    cluster_hostnames = get_slurm_node_list()
    print(f'There are {len(cluster_hostnames)} nodes available.')
    sys.stdout.flush()

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
        
        
    # Create monitors and apply all of the above settings to each monitor
    for monitor in monitors:
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

    if change_wavelength_range:
        create_dict_nx('forward_src_x')
        forward_src_x['wavelength start'] = lambda_min_um * 1e-6
        forward_src_x['wavelength stop'] = lambda_max_um * 1e-6
        forward_src_x['beam parameters'] = 'Waist size and position'
        forward_src_x['waist radius w0'] = 3e8/np.mean(3e8/np.array([lambda_min_um, lambda_max_um])) / (np.pi*background_index*np.radians(src_div_angle)) * 1e-6
        forward_src_x['distance from waist'] = -(src_maximum_vertical_um - device_size_vertical_um) * 1e-6
        for key, val in globals()['forward_src_x'].items():
            fdtd_hook.setnamed('forward_src_x', key, val)
    
    # Duplicate devices and set in an array
    if device_array_num > 1:
        create_dict_nx('device_mesh')
        device_mesh['x span'] = fdtd_region_size_lateral_um * 1e-6
        device_mesh['y span'] = fdtd_region_size_lateral_um * 1e-6
        for key, val in globals()['device_mesh'].items():
            fdtd_hook.setnamed('device_mesh', key, val)  

        # sidewall_thickness_um = 0.4
        device_arr_x_positions_um = (device_size_lateral_um + sidewall_thickness_um)*1e-6*np.array([1,1,0,-1,-1,-1,0,1])
        device_arr_y_positions_um = (device_size_lateral_um + sidewall_thickness_um)*1e-6*np.array([0,1,1,1,0,-1,-1,-1])

        for dev_arr_idx in range(0, device_array_num**2-1):
            fdtd_hook.select('design_import')
            fdtd_hook.copy(device_arr_x_positions_um[dev_arr_idx], device_arr_y_positions_um[dev_arr_idx])
            fdtd_hook.set('name', 'design_import_' + str(dev_arr_idx))
            
            for idx in range(0, num_sidewalls):
                sm_name = 'sidewall_' + str(idx)
                sm_mesh_name = 'mesh_sidewall_' + str(idx)
                
                # Duplicate sidewalls
                try:
                    objExists = fdtd_hook.getnamed(sm_name, 'name')
                    fdtd_hook.select(sm_name)
                    fdtd_hook.copy(device_arr_x_positions_um[dev_arr_idx], device_arr_y_positions_um[dev_arr_idx])
                    fdtd_hook.set('name', 'sidewall_misc')
                
                except Exception as ex:
                    template = "An exception of type {0} occurred. Arguments:\n{1!r}\n"
                    message = template.format(type(ex).__name__, ex.args)
                    print(message+'FDTD Object Sidewall does not exist.')
                    # fdtd_hook.addmesh(name=sidewall_mesh['name'])
                    # pass
                
                # Duplicate sidewall meshes
                try:
                    objExists = fdtd_hook.getnamed(sm_mesh_name, 'name')
                    fdtd_hook.select(sm_mesh_name)
                    fdtd_hook.copy(device_arr_x_positions_um[dev_arr_idx], device_arr_y_positions_um[dev_arr_idx])
                    fdtd_hook.set('name', 'mesh_sidewall_misc')
                
                except Exception as ex:
                    template = "An exception of type {0} occurred. Arguments:\n{1!r}\n"
                    message = template.format(type(ex).__name__, ex.args)
                    print(message+'FDTD Object Sidewall Mesh does not exist.')
                    # fdtd_hook.addmesh(name=sidewall_mesh['name'])
                    # pass
            print('2')
        print('3')

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

def generate_sorting_eff_spectrum(monitors_data, produce_spectrum, statistics='peak', add_opp_polarizations = False):
    '''Generates sorting efficiency spectrum and logs peak/mean values and corresponding indices of each quadrant. 
    Sorting efficiency of a quadrant is defined as overall power in that quadrant normalized against overall power in all four quadrants.
    Statistics determine how the value for the sweep is drawn from the spectrum: peak|mean'''
    output_plot = {}
    r_vectors = []      # Variables
    f_vectors = []      # Functions
    
    lambda_vector = monitors_data['wavelength']
    r_vectors.append({'var_name': 'Wavelength',
                      'var_values': lambda_vector
                    })
    
    ### E-field way of calculating sorting spectrum
    # Enorm_scatter_plane = monitors_data['scatter_plane_monitor']['E']
    # Enorm_focal_plane, Enorm_focal_monitor_arr = partition_scatter_plane_monitor(monitors_data['scatter_plane_monitor'], 'E')
    
    # # Integrates over area
    # Enorm_fp_overall = np.sum(Enorm_focal_plane, (0,1,2))
    # Enorm_fm_overall = []
    # quadrant_names = ['Blue', 'Green (x-pol.)', 'Red', 'Green (y-pol.)']
    # for idx in range(0, num_adjoint_sources):
    #     Enorm_fm_overall.append(np.sum(Enorm_focal_monitor_arr[idx], (0,1,2)))
    #     f_vectors.append({'var_name': quadrant_names[idx],
    #                   'var_values': np.divide(Enorm_fm_overall[idx], Enorm_fp_overall)
    #                 })
    
    quadrant_names = ['Blue', 'Green (x-pol.)', 'Red', 'Green (y-pol.)']
    for idx in range(0, num_adjoint_sources):
        f_vectors.append({'var_name': quadrant_names[idx],
                        'var_values': np.divide(monitors_data['transmission_monitor_'+str(idx)]['P'],
                                                monitors_data['transmission_focal_monitor_']['P'])
                        })
    
    if add_opp_polarizations:
        f_vectors[1]['var_name'] = 'Green'
        f_vectors[1]['var_values'] = f_vectors[1]['var_values'] + f_vectors[3]['var_values']
        f_vectors.pop(3)
    
    if produce_spectrum:
        output_plot['r'] = r_vectors
        output_plot['f'] = f_vectors
        output_plot['title'] = 'Sorting Efficiency Spectrum'
        output_plot['const_params'] = const_parameters
        
        # ## Plotting code to test
        # fig,ax = plt.subplots()
        # plt.plot(r_vectors[0]['var_values'], f_vectors[0]['var_values'], color='blue')
        # plt.plot(r_vectors[0]['var_values'], f_vectors[1]['var_values'], color='green')
        # plt.plot(r_vectors[0]['var_values'], f_vectors[2]['var_values'], color='red')
        # plt.plot(r_vectors[0]['var_values'], f_vectors[3]['var_values'], color='gray')
        # plt.show(block=False)
        # plt.show()
        print('Generated sorting efficiency spectrum.')

    
    sweep_values = []
    num_sweep_values = num_adjoint_sources if not add_opp_polarizations else num_bands
    sweep_values = append_new_sweep(sweep_values, f_vectors, num_sweep_values, statistics)
    
    # for idx in range(0, num_sweep_values):
    #     if statistics in ['peak']:
    #         sweep_index, sweep_val = max(enumerate(f_vectors[idx]['var_values']), key=(lambda x: x[1]))
    #         sweep_stdev = 0
    #     else:
    #         sweep_val = np.average(f_vectors[idx]['var_values'])
    #         sweep_index = min(range(len(f_vectors[idx]['var_values'])), key=lambda i: abs(f_vectors[idx]['var_values'][i]-sweep_val))
    #         sweep_stdev = np.std(f_vectors[idx]['var_values'])
            
    #     sweep_values.append({'value': float(sweep_val),
    #                     'index': sweep_index,
    #                     'std_dev': sweep_stdev,
    #                     'statistics': statistics
    #                     })
    
    print('Picked peak efficiency points for each quadrant to pass to sorting efficiency variable sweep.')
    
    return output_plot, sweep_values

def generate_sorting_transmission_spectrum(monitors_data, produce_spectrum, statistics='peak', add_opp_polarizations = False):
    '''Generates sorting efficiency spectrum and logs peak/mean values and corresponding indices of each quadrant. 
    Sorting efficiency of a quadrant is defined as overall power in that quadrant normalized against **sourcepower** (accounting for angle).
    Statistics determine how the value for the sweep is drawn from the spectrum: peak|mean'''
    output_plot = {}
    r_vectors = []      # Variables
    f_vectors = []      # Functions
    
    lambda_vector = monitors_data['wavelength']
    r_vectors.append({'var_name': 'Wavelength',
                      'var_values': lambda_vector
                    })
    
    quadrant_names = ['Blue', 'Green (x-pol.)', 'Red', 'Green (y-pol.)']
    for idx in range(0, num_adjoint_sources):
        f_vectors.append({'var_name': quadrant_names[idx],
                        'var_values': np.divide(monitors_data['transmission_monitor_'+str(idx)]['P'],
                                                monitors_data['incident_aperture_monitor']['P'])
                        })
    
    if add_opp_polarizations:
        f_vectors[1]['var_name'] = 'Green'
        f_vectors[1]['var_values'] = f_vectors[1]['var_values'] + f_vectors[3]['var_values']
        f_vectors.pop(3)
    
    if produce_spectrum:
        output_plot['r'] = r_vectors
        output_plot['f'] = f_vectors
        output_plot['title'] = 'Sorting Transmission Spectrum'
        output_plot['const_params'] = const_parameters
        
        # ## Plotting code to test
        # fig,ax = plt.subplots()
        # plt.plot(r_vectors[0]['var_values'], f_vectors[0]['var_values'], color='blue')
        # plt.plot(r_vectors[0]['var_values'], f_vectors[1]['var_values'], color='green')
        # plt.plot(r_vectors[0]['var_values'], f_vectors[2]['var_values'], color='red')
        # plt.plot(r_vectors[0]['var_values'], f_vectors[3]['var_values'], color='gray')
        # plt.show(block=False)
        # plt.show()
        print('Generated sorting transmission spectrum.')
  
    sweep_values = []
    num_sweep_values = num_adjoint_sources if not add_opp_polarizations else num_bands
    sweep_values = append_new_sweep(sweep_values, f_vectors, num_sweep_values, statistics)
    
    print('Picked peak transmission points for each quadrant to pass to sorting transmission variable sweep.')
    
    return output_plot, sweep_values

def generate_inband_val_sweep(monitors_data, produce_spectrum, statistics='peak', add_opp_polarizations = False, band_idxs = None):
    '''Generates sorting efficiency spectrum and logs peak/mean values and corresponding indices of each quadrant, IN BAND.
    This means that it takes the values within a defined spectral band and picks a statistical value for each band within that band.
    Produces num_bands**2 sweep_values, i.e. B->{B,G,R}, G->{B,G,R}, R->{B,G,R}  where for e.g. RB is the transmission of R quadrant within the B spectral band
    Sorting efficiency of a quadrant is defined as overall power in that quadrant normalized against **sourcepower** (accounting for angle).
    Statistics determine how the value for the sweep is drawn from the spectrum: peak|mean'''
    output_plot = {}
    r_vectors = []      # Variables
    f_vectors = []      # Functions
    
    lambda_vector = monitors_data['wavelength']
    r_vectors.append({'var_name': 'Wavelength',
                      'var_values': lambda_vector
                    })
    
    ### E-field way of calculating sorting spectrum
    # Enorm_scatter_plane = monitors_data['scatter_plane_monitor']['E']
    # Enorm_focal_plane, Enorm_focal_monitor_arr = partition_scatter_plane_monitor(monitors_data['scatter_plane_monitor'], 'E')
    
    # # Integrates over area
    # Enorm_fp_overall = np.sum(Enorm_focal_plane, (0,1,2))
    # Enorm_fm_overall = []
    # quadrant_names = ['Blue', 'Green (x-pol.)', 'Red', 'Green (y-pol.)']
    # for idx in range(0, num_adjoint_sources):
    #     Enorm_fm_overall.append(np.sum(Enorm_focal_monitor_arr[idx], (0,1,2)))
    #     f_vectors.append({'var_name': quadrant_names[idx],
    #                   'var_values': np.divide(Enorm_fm_overall[idx], Enorm_fp_overall)
    #                 })
    
    quadrant_names = ['Blue', 'Green (x-pol.)', 'Red', 'Green (y-pol.)']
    for idx in range(0, num_adjoint_sources):
        f_vectors.append({'var_name': quadrant_names[idx],
                        'var_values': np.divide(monitors_data['transmission_monitor_'+str(idx)]['P'],
                                                monitors_data['incident_aperture_monitor']['P'])
                        })
    
    if add_opp_polarizations:
        f_vectors[1]['var_name'] = 'Green'
        f_vectors[1]['var_values'] = f_vectors[1]['var_values'] + f_vectors[3]['var_values']
        f_vectors.pop(3)
    
    if produce_spectrum:
        output_plot['r'] = r_vectors
        output_plot['f'] = f_vectors
        output_plot['title'] = 'Sorting Transmission Spectrum'
        output_plot['const_params'] = const_parameters
        
        # ## Plotting code to test
        # fig,ax = plt.subplots()
        # plt.plot(r_vectors[0]['var_values'], f_vectors[0]['var_values'], color='blue')
        # plt.plot(r_vectors[0]['var_values'], f_vectors[1]['var_values'], color='green')
        # plt.plot(r_vectors[0]['var_values'], f_vectors[2]['var_values'], color='red')
        # plt.plot(r_vectors[0]['var_values'], f_vectors[3]['var_values'], color='gray')
        # plt.show(block=False)
        # plt.show()
        print('Generated sorting transmission spectrum.')
  
  
    sweep_values = []
    num_sweep_values = num_adjoint_sources if not add_opp_polarizations else num_bands
    
    # Splits entire spectrum into equal bands, gets indices of starts and ends
    if band_idxs is None:
        band_idxs = []
        for inband_idx in range(0, num_sweep_values):
            band_idxs.append((np.floor(len(r_vectors[0]['var_values'])/num_sweep_values) * 
                                np.asarray([inband_idx, inband_idx+1]) - np.asarray([0, 1])).astype(int))
    
    for idx in range(0, num_sweep_values):
        
        # e.g. For B quadrant, get the peak in the B Spectral band
        if statistics in ['peak']:
                sweep_index, sweep_val = max(enumerate(f_vectors[idx]['var_values']), key=(lambda x: x[1]))
        
        for inband_idx in range(0, num_sweep_values):
            # e.g. For G quadrant, get the value at the peak in the B spectral band
            if statistics in ['peak']:
                # sweep_index, sweep_val = max(enumerate(f_vectors[idx]['var_values'][slice(*band_idxs[idx])]), key=(lambda x: x[1]))
                sweep_val = f_vectors[inband_idx]['var_values'][sweep_index]
                sweep_stdev = 0
                
            else:
                # TODO: Check that the mean is indeed taken in-band. Check band starts and band ends - a bit sus
                sweep_val = np.average(f_vectors[idx]['var_values'][slice(*band_idxs[inband_idx])])
                sweep_index = min(range(len(f_vectors[idx]['var_values'])), key=lambda i: abs(f_vectors[idx]['var_values'][i]-sweep_val))
                sweep_stdev = np.std(f_vectors[idx]['var_values'][slice(*band_idxs[inband_idx])])
          
            sweep_values.append({'value': float(sweep_val),
                            'index': sweep_index,
                            'std_dev': sweep_stdev,
                            'statistics': statistics
                            })
    
    print('Picked mean transmission points, within band, for each quadrant to pass to sorting transmission variable sweep.')
    
    return sweep_values
        
def generate_device_transmission_spectrum(monitors_data, produce_spectrum, statistics='mean'):
    '''Generates device transmission spectrum and logs peak/mean value and corresponding index. 
    Transmission is defined as power coming out of exit aperture, normalized against power entering input aperture.'''
    
    output_plot = {}
    r_vectors = []      # Variables
    f_vectors = []      # Functions
    
    lambda_vector = monitors_data['wavelength']
    r_vectors.append({'var_name': 'Wavelength',
                      'var_values': lambda_vector
                    })
    
    P_in = monitors_data['incident_aperture_monitor']['P']
    # P_in = monitors_data['sourcepower'] - monitors_data['src_spill_monitor']['P']
    P_out = monitors_data['exit_aperture_monitor']['P']
    T_device = np.divide(P_out, P_in)
    f_vectors.append({'var_name': 'Device Transmission',
                      'var_values': T_device
                    })
    
    if produce_spectrum:
        output_plot['r'] = r_vectors
        output_plot['f'] = f_vectors
        output_plot['title'] = 'Transmission Spectrum'
        output_plot['const_params'] = const_parameters
        
        print('Generated transmission spectrum.')
    
    sweep_values = []
    num_sweep_values = 1
    sweep_values = append_new_sweep(sweep_values, f_vectors, num_sweep_values, statistics)
    
    print('Picked transmission point to pass to variable sweep.')
    
    return output_plot, sweep_values

def generate_Enorm_focal_plane(monitors_data, produce_spectrum):
    '''Generates image slice at focal plane (including focal scatter). The output is over all wavelengths so a wavelength still needs to be selected.'''
    
    try:
        monitor = monitors_data['scatter_plane_monitor']
    except Exception as err:
        monitor = monitors_data['spill_plane_monitor']
    
    output_plot = {}
    r_vectors = []      # Variables
    f_vectors = []      # Functions
    lambda_vectors = [] # Wavelengths
    
    r_vectors.append({'var_name': 'x-axis',
                      'var_values': monitor['coords']['x']
                      })
    r_vectors.append({'var_name': 'y-axis',
                      'var_values': monitor['coords']['x']
                      })
    
    f_vectors.append({'var_name': 'E_norm scatter',
                      'var_values': np.squeeze(monitor['E'])
                        })
    
    lambda_vectors.append({'var_name': 'Wavelength',
                           'var_values': monitors_data['wavelength']
                            })
    
    output_plot['r'] = r_vectors
    output_plot['f'] = f_vectors
    output_plot['lambda'] = lambda_vectors 
    output_plot['title'] = 'E-norm Image (Spectrum)'
    output_plot['const_params'] = const_parameters
    
    # # Also generate crosstalk relative numbers
    # E_vals = output_plot['f'][0]['var_values']
    # l = np.array_split(E_vals,3,axis=0)
    # split_E_vals = []
    # for a in l:
    #     l = np.array_split(a,3,axis=1)
    #     split_E_vals += l
    # split_E_vals = np.array(split_E_vals)
    
    # # split_E_vals has shape (9, nx/3, ny/3, nλ)
    # split_power_vals = np.sum(np.square(split_E_vals),(1,2))
    # sum_split_power_vals = np.sum(split_power_vals, 0)
    # crosstalk_vals = np.divide(split_power_vals, sum_split_power_vals)
    
    
    print('Generated E-norm focal plane image plot.')
    
    # ## Plotting code to test
    # fig, ax = plt.subplots()
    # #plt.contour([np.squeeze(r_vectors[0]['var_values']), np.squeeze(r_vectors[1]['var_values'])], f_vectors[0]['var_values'][:,:,5])
    # plt.imshow(f_vectors[0]['var_values'][:,:,5])
    # plt.show(block=False)
    # plt.show()
    
    return output_plot, {}

def generate_device_cross_section_spectrum(monitors_data, produce_spectrum):
    '''Generates image slice at device cross-section. The output is over all wavelengths so a wavelength still needs to be selected.'''
    
    
    output_plot = {}
    r_vectors = []      # Variables
    f_vectors = []      # Functions
    lambda_vectors = [] # Wavelengths
    
    for device_cross_idx in range(0,6):
        
        monitor = monitors_data['device_xsection_monitor_' + str(device_cross_idx)]
    
        r_vectors.append({'var_name': 'x-axis',
                        'var_values': monitor['coords']['x']
                        })
        r_vectors.append({'var_name': 'y-axis',
                        'var_values': monitor['coords']['y']
                        })
        r_vectors.append({'var_name': 'z-axis',
                        'var_values': monitor['coords']['z']
                        })
        
        f_vectors.append({'var_name': 'E_norm',
                        'var_values': np.squeeze(monitor['E'])
                         })
        f_vectors.append({'var_name': 'E_real',
                        'var_values': np.squeeze(monitor['E_real'])
                        })
        
        
        lambda_vectors.append({'var_name': 'Wavelength',
                            'var_values': monitors_data['wavelength']
                            })
    
    output_plot['r'] = r_vectors
    output_plot['f'] = f_vectors
    output_plot['lambda'] = lambda_vectors 
    output_plot['title'] = 'Device Cross-Section Image (Spectrum)'
    output_plot['const_params'] = const_parameters
    
    print('Generated device cross section image plot.')
    
    # ## Plotting code to test
    # fig, ax = plt.subplots()
    # #plt.contour([np.squeeze(r_vectors[0]['var_values']), np.squeeze(r_vectors[2]['var_values'])], f_vectors[0]['var_values'][:,:,5])
    # plt.imshow(f_vectors[0]['var_values'][:,:,5])
    # plt.show(block=False)
    # plt.show()
    
    return output_plot, {}

def generate_source_spill_power_spectrum(monitors_data, produce_spectrum, statistics='mean'):
    '''Generates power spectrum for the power from the source that does not go into the input aperture
    and logs peak/mean value and corresponding index.'''
    
    output_plot = {}
    r_vectors = []      # Variables
    f_vectors = []      # Functions
    
    lambda_vector = monitors_data['wavelength']
    r_vectors.append({'var_name': 'Wavelength',
                      'var_values': lambda_vector
                    })
    
    device_input_power = monitors_data['incident_aperture_monitor']['P']
    spill_power = monitors_data['sourcepower'] - monitors_data['incident_aperture_monitor']['P']
    source_power = monitors_data['sourcepower']
    
    f_vectors.append({'var_name': 'Source Spill Power',
                      'var_values': spill_power / source_power
                    })
    
    if produce_spectrum:
        output_plot['r'] = r_vectors
        output_plot['f'] = f_vectors
        output_plot['title'] = 'Source Spill Power Spectrum'
        output_plot['const_params'] = const_parameters
        
        print('Generated source spill power spectrum.')
    
    sweep_values = []
    num_sweep_values = 1
    sweep_values = append_new_sweep(sweep_values, f_vectors, num_sweep_values, statistics)
    
    print('Picked source spill power point to pass to variable sweep.')
    
    return output_plot, sweep_values

def generate_overall_reflection_spectrum(monitors_data, produce_spectrum, statistics='mean'):
    '''Generates overall FDTD reflection spectrum and logs peak/mean value and corresponding index. 
    Reflection is calculated directly from a transmission monitor directly between the source and device input interface, and is normalized against sourcepower.
    This monitor spans the FDTD region.'''
    
    output_plot = {}
    r_vectors = []      # Variables
    f_vectors = []      # Functions
    
    lambda_vector = monitors_data['wavelength']
    r_vectors.append({'var_name': 'Wavelength',
                      'var_values': lambda_vector
                    })
    
    R_overall = np.subtract(1, monitors_data['src_transmission_monitor']['T'])
    f_vectors.append({'var_name': 'Device Reflection',
                      'var_values': R_overall
                    })
    
    if produce_spectrum:
        output_plot['r'] = r_vectors
        output_plot['f'] = f_vectors
        output_plot['title'] = 'Reflection Spectrum'
        output_plot['const_params'] = const_parameters
        
        print('Generated reflection spectrum.')
    
    sweep_values = []
    num_sweep_values = 1
    sweep_values = append_new_sweep(sweep_values, f_vectors, num_sweep_values, statistics)
    
    print('Picked reflection point to pass to variable sweep.')
    
    return output_plot, sweep_values

def generate_crosstalk_power_spectrum(monitors_data, produce_spectrum, statistics='mean'):
    '''Generates power spectrum for adjacent pixels at focal plane.'''
    
    output_plot = {}
    r_vectors = []      # Variables
    f_vectors = []      # Functions
    
    lambda_vector = monitors_data['wavelength']
    r_vectors.append({'var_name': 'Wavelength',
                      'var_values': lambda_vector
                    })
    
    try:
        monitor = monitors_data['scatter_plane_monitor']
    except Exception as err:
        monitor = monitors_data['spill_plane_monitor']
    E_vals = np.squeeze(monitor['E'])
    E_vals = E_vals
    
    # Generate crosstalk relative numbers
    n_x = np.shape(E_vals)[0]
    n_x = int(4*np.floor(n_x/4))
    n_y = np.shape(E_vals)[0]
    n_y = int(4*np.floor(n_x/4))
    E_vals = E_vals[0:n_x, 0:n_y, :]        # First, make sure the array is split nicely into equal sections
    
    l = np.array_split(E_vals,4,axis=0)
    split_E_vals = []
    for a in l:
        l = np.array_split(a,4,axis=1)
        split_E_vals += l
    split_E_vals = np.array(split_E_vals)
    
    # split_E_vals has shape (16, nx/4, ny/4, nλ)
    split_power_vals = np.sum(np.square(split_E_vals),(1,2))
    sum_split_power_vals = np.sum(split_power_vals, 0)
    crosstalk_vals_mod = np.divide(split_power_vals, sum_split_power_vals)
    # crosstalk_vals = np.divide(split_power_vals, sum_split_power_vals)
    
    crosstalk_vals = np.zeros((9,np.shape(crosstalk_vals_mod)[1]))
    crosstalk_vals[0,:] = crosstalk_vals_mod[0,:]
    crosstalk_vals[1,:] = crosstalk_vals_mod[1,:] + crosstalk_vals_mod[2,:]
    crosstalk_vals[2,:] = crosstalk_vals_mod[3,:]
    crosstalk_vals[3,:] = crosstalk_vals_mod[4,:] + crosstalk_vals_mod[8,:]
    crosstalk_vals[4,:] = crosstalk_vals_mod[5,:] + crosstalk_vals_mod[6,:] + crosstalk_vals_mod[9,:] + crosstalk_vals_mod[10,:]
    crosstalk_vals[5,:] = crosstalk_vals_mod[7,:] + crosstalk_vals_mod[11,:]
    crosstalk_vals[6,:] = crosstalk_vals_mod[12,:]
    crosstalk_vals[7,:] = crosstalk_vals_mod[13,:] + crosstalk_vals_mod[14,:]
    crosstalk_vals[8,:] = crosstalk_vals_mod[15,:]
    
    for idx in range(0, 9):
        f_vectors.append({'var_name': f'Crosstalk Power, Pixel {idx}',
                        'var_values': crosstalk_vals[idx,:]
                        })
    
    if produce_spectrum:
        output_plot['r'] = r_vectors
        output_plot['f'] = f_vectors
        output_plot['title'] = 'Crosstalk Power Spectrum'
        output_plot['const_params'] = const_parameters
        
        # ## Plotting code to test
        # fig,ax = plt.subplots()
        # plt.plot(r_vectors[0]['var_values'], f_vectors[0]['var_values'], color='blue')
        # plt.plot(r_vectors[0]['var_values'], f_vectors[1]['var_values'], color='green')
        # plt.plot(r_vectors[0]['var_values'], f_vectors[2]['var_values'], color='red')
        # plt.plot(r_vectors[0]['var_values'], f_vectors[3]['var_values'], color='gray')
        # plt.show(block=False)
        # plt.show()
        print('Generated crosstalk power spectrum.')
    
    sweep_values = []
    num_sweep_values = 9
    sweep_values = append_new_sweep(sweep_values, f_vectors, num_sweep_values, statistics)
    
    print('Picked crosstalk power points for each Bayer pixel, to pass to variable sweep.')
    
    sweep_mean_vals = []
    for swp_val in sweep_values:
        sweep_mean_vals.append(swp_val['value'])
    print(sweep_mean_vals)
    
    print('Generated adjacent pixel crosstalk power plot.')
    
    # ## Plotting code to test
    # fig, ax = plt.subplots()
    # #plt.contour([np.squeeze(r_vectors[0]['var_values']), np.squeeze(r_vectors[1]['var_values'])], f_vectors[0]['var_values'][:,:,5])
    # plt.imshow(f_vectors[0]['var_values'][:,:,5])
    # plt.show(block=False)
    # plt.show()
    
    return output_plot, sweep_values

def generate_side_power_spectrum(monitors_data, produce_spectrum, statistics='mean'):
    '''Generates side scattering power spectrum and logs peak/mean values and corresponding indices of each side monitor.'''
    
    output_plot = {}
    r_vectors = []      # Variables
    f_vectors = []      # Functions
    
    lambda_vector = monitors_data['wavelength']
    r_vectors.append({'var_name': 'Wavelength',
                      'var_values': lambda_vector
                    })
    
    for idx in range(0, 4):
        f_vectors.append({'var_name': f'Side Scattering Power {idx}',
                        'var_values': monitors_data['side_monitor_'+str(idx)]['P']
                        })
    
    if produce_spectrum:
        output_plot['r'] = r_vectors
        output_plot['f'] = f_vectors
        output_plot['title'] = 'Side Scattering Power Spectrum'
        output_plot['const_params'] = const_parameters
        
        # ## Plotting code to test
        # fig,ax = plt.subplots()
        # plt.plot(r_vectors[0]['var_values'], f_vectors[0]['var_values'], color='blue')
        # plt.plot(r_vectors[0]['var_values'], f_vectors[1]['var_values'], color='green')
        # plt.plot(r_vectors[0]['var_values'], f_vectors[2]['var_values'], color='red')
        # plt.plot(r_vectors[0]['var_values'], f_vectors[3]['var_values'], color='gray')
        # plt.show(block=False)
        # plt.show()
        print('Generated side scattering power spectrum.')
    
    sweep_values = []
    num_sweep_values = 4
    sweep_values = append_new_sweep(sweep_values, f_vectors, num_sweep_values, statistics)
    
    print('Picked side scattering power points for each quadrant to pass to variable sweep.')
    
    return output_plot, sweep_values

def generate_focal_power_spectrum(monitors_data, produce_spectrum, statistics='mean'):
    '''Generates power spectrum for the combined 4 focal plane quadrants, and logs peak/mean value and corresponding index.'''
    
    output_plot = {}
    r_vectors = []      # Variables
    f_vectors = []      # Functions
    
    lambda_vector = monitors_data['wavelength']
    r_vectors.append({'var_name': 'Wavelength',
                      'var_values': lambda_vector
                    })
    
    f_vectors.append({'var_name': 'Focal Region Power',
                      'var_values': monitors_data['transmission_focal_monitor_']['P']
                    })
    
    if produce_spectrum:
        output_plot['r'] = r_vectors
        output_plot['f'] = f_vectors
        output_plot['title'] = 'Focal Region Power Spectrum'
        output_plot['const_params'] = const_parameters
        
        print('Generated focal region power spectrum.')
    
    sweep_values = []
    num_sweep_values = 1
    sweep_values = append_new_sweep(sweep_values, f_vectors, num_sweep_values, statistics)
    
    print('Picked focal region power point to pass to variable sweep.')
    
    return output_plot, sweep_values

def generate_oblique_scattering_power_spectrum(monitors_data, produce_spectrum, statistics='mean'):
    '''Generates oblique scattering power spectrum and logs peak/mean values and corresponding indices of each side monitor.'''
    
    output_plot = {}
    r_vectors = []      # Variables
    f_vectors = []      # Functions
    
    lambda_vector = monitors_data['wavelength']
    r_vectors.append({'var_name': 'Wavelength',
                      'var_values': lambda_vector
                    })
    
    total_var_values = 0
    for idx in range(0, 4):
        total_var_values = total_var_values + monitors_data['vertical_scatter_monitor_'+str(idx)]['P']
        f_vectors.append({'var_name': f'Oblique Scattering Power {idx}',
                        'var_values': monitors_data['vertical_scatter_monitor_'+str(idx)]['P']
                        })
    f_vectors.append({'var_name': f'Total Oblique Scattering Power',
                        'var_values': total_var_values
                        })
    
    if produce_spectrum:
        output_plot['r'] = r_vectors
        output_plot['f'] = f_vectors
        output_plot['title'] = 'Oblique Scattering Power Spectrum'
        output_plot['const_params'] = const_parameters
        
        # ## Plotting code to test
        # fig,ax = plt.subplots()
        # plt.plot(r_vectors[0]['var_values'], f_vectors[0]['var_values'], color='blue')
        # plt.plot(r_vectors[0]['var_values'], f_vectors[1]['var_values'], color='green')
        # plt.plot(r_vectors[0]['var_values'], f_vectors[2]['var_values'], color='red')
        # plt.plot(r_vectors[0]['var_values'], f_vectors[3]['var_values'], color='gray')
        # plt.plot(r_vectors[0]['var_values'], f_vectors[4]['var_values'], color='brown')
        # plt.show(block=False)
        # plt.show()
        print('Generated oblique scattering power spectrum.')
    
    sweep_values = []
    num_sweep_values = 4+1
    sweep_values = append_new_sweep(sweep_values, f_vectors, num_sweep_values, statistics)
    
    print('Picked oblique scattering power points for each side, and the total, to pass to variable sweep.')
    
    return output_plot, sweep_values

def generate_scatter_plane_power_spectrum(monitors_data, produce_spectrum, statistics='mean'):
    '''Generates power spectrum for the overall scatter plane (focal region quadrants and crosstalk/spillover), 
    and logs peak/mean value and corresponding index.'''
    
    output_plot = {}
    r_vectors = []      # Variables
    f_vectors = []      # Functions
    
    lambda_vector = monitors_data['wavelength']
    r_vectors.append({'var_name': 'Wavelength',
                      'var_values': lambda_vector
                    })
    
    try:
        scatter_power = monitors_data['scatter_plane_monitor']['P']
    except Exception as err:
        scatter_power = monitors_data['spill_plane_monitor']['P']
    
    f_vectors.append({'var_name': 'Scatter Plane Power',
                      'var_values': scatter_power
                    })
    
    if produce_spectrum:
        output_plot['r'] = r_vectors
        output_plot['f'] = f_vectors
        output_plot['title'] = 'Scatter Plane Power Spectrum'
        output_plot['const_params'] = const_parameters
        
        print('Generated scatter plane power spectrum.')
    
    sweep_values = []
    num_sweep_values = 1
    sweep_values = append_new_sweep(sweep_values, f_vectors, num_sweep_values, statistics)
    
    print('Picked scatter plane power point to pass to variable sweep.')
    
    return output_plot, sweep_values

def generate_exit_power_distribution_spectrum(monitors_data, produce_spectrum, statistics='mean'):
    '''Generates two power spectra on the same plot for both focal region power and crosstalk/spillover, 
    and logs peak/mean value and corresponding index.'''
    # Focal scatter and focal region power, normalized against exit aperture power

    output_plot = {}
    r_vectors = []      # Variables
    f_vectors = []      # Functions
    
    lambda_vector = monitors_data['wavelength']
    r_vectors.append({'var_name': 'Wavelength',
                      'var_values': lambda_vector
                    })
    
    # try:
    #     scatter_power_2 = monitors_data['scatter_plane_monitor']['P']
    # except Exception as err:
    #     scatter_power_2 = monitors_data['spill_plane_monitor']['P']
    scatter_power = 0
    for vsm_idx in range(0,4):
        scatter_power = scatter_power + monitors_data['vertical_scatter_monitor_' + str(vsm_idx)]['P']
    focal_power = monitors_data['transmission_focal_monitor_']['P']
    exit_aperture_power = monitors_data['exit_aperture_monitor']['P']
    
    # Q: Does scatter + focal = exit_aperture?
    # A: No - still under by about (3.5 ± 0.882) %
    
    f_vectors.append({'var_name': 'Focal Region Power',
                      'var_values': focal_power/(focal_power + scatter_power)
                    })
    
    f_vectors.append({'var_name': 'Oblique Scattering Power',
                      'var_values': scatter_power/(focal_power + scatter_power)
                    })
    
    if produce_spectrum:
        output_plot['r'] = r_vectors
        output_plot['f'] = f_vectors
        output_plot['title'] = 'Focal Plane Power Distribution Spectrum'
        output_plot['const_params'] = const_parameters
        
        print('Generated focal plane power distribution spectrum.')
    
    sweep_values = []
    num_sweep_values = len(f_vectors)
    sweep_values = append_new_sweep(sweep_values, f_vectors, num_sweep_values, statistics)
    
    print('Picked exit power distribution points to pass to variable sweep.')
    
    return output_plot, sweep_values

def generate_device_rta_spectrum(monitors_data, produce_spectrum, statistics='mean'):
    '''Generates power spectra on the same plot for all contributions to RTA, 
    and logs peak/mean value and corresponding index.
    Everything is normalized to the power input to the device.'''
    # Overall RTA of the device, normalized against sum.
            # - Overall Reflection
            # - Side Powers
            # - Focal Region Power
            # - Oblique Scattering
            # - Device Absorption
    
    
    output_plot = {}
    r_vectors = []      # Variables
    f_vectors = []      # Functions
    
    lambda_vector = monitors_data['wavelength']
    r_vectors.append({'var_name': 'Wavelength',
                      'var_values': lambda_vector
                    })
    
    sourcepower = monitors_data['sourcepower']
    input_power = monitors_data['incident_aperture_monitor']['P']
    
    R_power = np.reshape(1 - monitors_data['src_transmission_monitor']['T'],    (len(lambda_vector),1))
    R_power = R_power * monitors_data['sourcepower']
    
    side_powers = []
    side_directions = ['E','N','W','S']
    sides_power = 0
    for idx in range(0, 4):
        side_powers.append(monitors_data['side_monitor_'+str(idx)]['P'])
        sides_power = sides_power + monitors_data['side_monitor_'+str(idx)]['P']
    
    try:
        scatter_power_2 = monitors_data['scatter_plane_monitor']['P']
    except Exception as err:
        scatter_power_2 = monitors_data['spill_plane_monitor']['P']
        
    focal_power = monitors_data['transmission_focal_monitor_']['P']
    
    scatter_power = 0
    for idx in range(0,4):
        scatter_power = scatter_power + monitors_data['vertical_scatter_monitor_'+str(idx)]['P']
        
    exit_aperture_power = monitors_data['exit_aperture_monitor']['P']
    
    power_sum = R_power + focal_power + scatter_power
    for idx in range(0, 4):
        power_sum += side_powers[idx]
    
    f_vectors.append({'var_name': 'Device Reflection',
                      'var_values': R_power/input_power
                    })
    
    for idx in range(0, 4):
        f_vectors.append({'var_name': 'Side Scattering ' + side_directions[idx],
                      'var_values': side_powers[idx]/input_power
                    })    
        
    
    f_vectors.append({'var_name': 'Focal Region Power',
                      'var_values': focal_power/input_power
                    })
    
    f_vectors.append({'var_name': 'Oblique Scattering Power',
                      'var_values': scatter_power/input_power   # (exit_aperture_power - focal_power)/input_power
                    })
    
    f_vectors.append({'var_name': 'Absorbed Power',
                      'var_values': 1 - power_sum/input_power
                    })
    
    if produce_spectrum:
        output_plot['r'] = r_vectors
        output_plot['f'] = f_vectors
        output_plot['title'] = 'Device RTA Spectrum'
        output_plot['const_params'] = const_parameters
        
        print('Generated device RTA power spectrum.')
    
    sweep_values = []
    num_sweep_values = len(f_vectors)
    sweep_values = append_new_sweep(sweep_values, f_vectors, num_sweep_values, statistics)
    
    print('Picked device RTA points to pass to variable sweep.')
    
    return output_plot, sweep_values

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
    
# -- Spectrum Plot Functions (Per Job)

def plot_sorting_eff_spectrum(plot_data, job_idx):
    '''For the given job index, plots the sorting efficiency spectrum corresponding to that job.'''
    
    sp = SpectrumPlot(plot_data)
    
    plot_colors = ['blue', 'green', 'red', 'gray']
    plot_labels = ['Blue', 'Green (x-pol.)', 'Red', 'Green (y-pol.)']
    sp.append_line_data(plot_colors, plot_labels)
    
    sp.assign_title('Spectrum in Each Quadrant', job_idx)
    sp.assign_axis_labels('Wavelength (um)', 'Sorting Efficiency')
    
    sp.export_plot_config('sorting_efficiency_spectra', job_idx)

    # r_vectors = plot_data['r']
    # f_vectors = plot_data['f']
    
    # fig, ax = plt.subplots()
    # plot_config = {'title': None,
    #                'x_axis': {'label':'', 'limits':[]},
    #                'y_axis': {'label':'', 'limits':[]},
    #                'lines': []
    #             }
    
    # plot_colors = ['blue', 'green', 'red', 'gray']
    # plot_labels = ['Blue', 'Green (x-pol.)', 'Red', 'Green (y-pol.)']
    
    # for plot_idx in range(0, len(f_vectors)):
    #     line_data = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
    #                  'y_axis': {'values': None, 'factor': 1, 'offset': 0},
    #                  'cutoff': None,
    #                  'color': None,
    #                  'alpha': 1.0,
    #                  'legend': None,
    #                  'marker_style': marker_style
    #             }
        
    #     line_data['x_axis']['values'] = r_vectors[0]['var_values']
    #     line_data['x_axis']['factor'] = 1e6
    #     line_data['y_axis']['values'] = f_vectors[plot_idx]['var_values']
    #     line_data['cutoff'] = slice(0, len(line_data['x_axis']['values']))
        
    #     line_data['color'] = plot_colors[plot_idx]
    #     line_data['legend'] = plot_labels[plot_idx]
        
    #     plot_config['lines'].append(line_data)
        
    
    # title_string = 'Spectrum in Each Quadrant'
    # for sweep_param_value in list(sweep_parameters.values()):
    #     current_value = sweep_param_value['var_values'][job_idx[2][0]]
    #     optimized_value = sweep_param_value['var_values'][sweep_param_value['peakInd']]
    #     title_string += '\n' + sweep_param_value['var_name'] + f': Current Value {current_value}, Optimized for {optimized_value}'
    # plot_config['title'] = title_string
    
    # plot_config['x_axis']['label'] = 'Wavelength (um)'
    # plot_config['y_axis']['label'] = 'Sorting Efficiency'
    
    
    
    # fig, ax = enter_plot_data_1d(plot_config, fig, ax)
    
    
    # plot_subfolder_name = 'sorting_efficiency_spectra'
    # if not os.path.isdir(plot_directory_location + '/' + plot_subfolder_name):
    #     os.makedirs(plot_directory_location + '/' + plot_subfolder_name)
    # plt.savefig(plot_directory_location + '/' + plot_subfolder_name\
    #     + '/' + isolate_filename(job_names[job_idx]).replace('.fsp', '') + '.png',
    #     bbox_inches='tight')
    # print('Exported: Sorting Efficiency Spectrum')
    
    # plt.close()
    
    # return fig, ax
    return sp.fig, sp.ax

def plot_sorting_transmission_spectrum(plot_data, job_idx):
    '''For the given job index, plots the sorting efficiency (transmission) spectrum corresponding to that job.'''
    
    sp = SpectrumPlot(plot_data)
    
    plot_colors = ['blue', 'green', 'red', 'gray']
    plot_labels = ['Blue', 'Green (x-pol.)', 'Red', 'Green (y-pol.)']
    sp.append_line_data(plot_colors, plot_labels)
    
    sp.assign_title('Spectrum in Each Quadrant', job_idx)
    sp.assign_axis_labels('Wavelength (um)', 'Sorting Transmission Efficiency')
    
    sp.export_plot_config('sorting_trans_efficiency_spectra', job_idx)
    return sp.fig, sp.ax

def plot_Enorm_focal_plane_image_spectrum(plot_data, job_idx, plot_wavelengths = None, ignore_opp_polarization = True):
    '''For the given job index, plots the E-norm f.p. image at specific input spectra, corresponding to that job.
    Note: Includes focal scatter region. '''
    
    r_vectors = plot_data['r']
    f_vectors = plot_data['f']
    lambda_vectors = plot_data['lambda']
    wl_vector = np.squeeze(lambda_vectors[0]['var_values']).tolist()
    
    if plot_wavelengths is None:
        # plot_wavelengths = [lambda_min_um, lambda_values_um[int(len(lambda_values_um/2))]lambda_max_um]
        spectral_band_midpoints = np.multiply(num_points_per_band, np.arange(0.5, num_bands + 0.5)).astype(int)         # This fails if num_points_per_band and num_bands are different than what was optimized
        plot_wavelengths = np.array(wl_vector)[spectral_band_midpoints].tolist()
        
    plot_colors = ['blue', 'green', 'red', 'yellow']
    
    if ignore_opp_polarization:
        plot_wavelengths = plot_wavelengths[:-1]
    
    for plot_wl in plot_wavelengths:
        plot_wl = float(plot_wl)
         
        fig, ax = plt.subplots()
        
        wl_index = min(range(len(wl_vector)), key=lambda i: abs(wl_vector[i]-plot_wl))
        # plt.contour([np.squeeze(r_vectors[0]['var_values']), np.squeeze(r_vectors[1]['var_values'])], 
        #             f_vectors[0]['var_values'][:,:,wl_index])
        # plt.imshow(f_vectors[0]['var_values'][:,:,wl_index])
        Y_grid, X_grid = np.meshgrid(np.squeeze(r_vectors[0]['var_values'])*1e6, np.squeeze(r_vectors[1]['var_values'])*1e6)
        c = ax.pcolormesh(X_grid, Y_grid, f_vectors[0]['var_values'][:,:,wl_index],
                          cmap='jet', shading='auto')      # cmap='RdYlBu_r' is also good
        plt.gca().set_aspect('equal')
        
        wl_str = f'{plot_wl*1e9:.0f} nm' if plot_wl < 1e-6 else f'{plot_wl*1e6:.3f} um'
        title_string = r'$E_{norm}$' + ' at Focal Plane: $\lambda = $ ' + f'{wl_str}'
        plt.title(title_string)
        plt.xlabel('x (um)')
        plt.ylabel('y (um)')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.25)
        fig.colorbar(c, cax=cax)
        
        fig_width = 8.0
        fig_height = fig_width*1.0
        l = ax.figure.subplotpars.left
        r = ax.figure.subplotpars.right
        t = ax.figure.subplotpars.top
        b = ax.figure.subplotpars.bottom
        figw = float(fig_width)/(r-l)
        figh = float(fig_height)/(t-b)
        ax.figure.set_size_inches(figw, figh, forward=True)
        plt.tight_layout()
        
        #plt.show(block=False)
        #plt.show()
        
        plot_subfolder_name = 'Enorm_fp_image_spectra'
        if not os.path.isdir(plot_directory_location + '/' + plot_subfolder_name):
            os.makedirs(plot_directory_location + '/' + plot_subfolder_name)
            
        plt.savefig(plot_directory_location + '/' + plot_subfolder_name\
            + '/' + isolate_filename(job_names[job_idx]).replace('.fsp', '')\
            + '_' + f'{wl_str}' + '.png',
            bbox_inches='tight')
        print('Exported: Enorm Focal Plane Image at wavelength ' + wl_str)
    
        plt.close()
    
    return fig, ax

def plot_device_cross_section_spectrum(plot_data, job_idx, plot_wavelengths = None, ignore_opp_polarization = True):
    '''For the given job index, plots the device cross section image plots at specific input spectra, corresponding to that job.'''
    
    r_vectors = plot_data['r']
    f_vectors = plot_data['f']
    lambda_vectors = plot_data['lambda']
    wl_vector = np.squeeze(lambda_vectors[0]['var_values']).tolist()
    
    if plot_wavelengths is None:
        # plot_wavelengths = [lambda_min_um, lambda_values_um[int(len(lambda_values_um/2))]lambda_max_um]
        spectral_band_midpoints = np.multiply(num_points_per_band, np.arange(0.5, num_bands + 0.5)).astype(int)         # This fails if num_points_per_band and num_bands are different than what was optimized
        plot_wavelengths = np.array(wl_vector)[spectral_band_midpoints].tolist()
        
    plot_colors = ['blue', 'green', 'red', 'yellow']
    
    if ignore_opp_polarization:
        plot_wavelengths = plot_wavelengths[:-1]
    
    for device_cross_idx in range(0,6):
        
        #* Create E-norm images
        for plot_wl in plot_wavelengths:
            plot_wl = float(plot_wl)
            
            fig, ax = plt.subplots()
            
            is_x_slice = False
            y_grid = r_vectors[device_cross_idx * 3 + 2]['var_values']                   # Plot the z-axis as the vertical
            x_grid = r_vectors[device_cross_idx * 3]['var_values']                       # Check if it's the x-axis or y-axis that is the slice; plot the non-slice as the horizontal of the image plot
            slice_val = r_vectors[device_cross_idx * 3 + 1]['var_values']
            slice_str = f'$y = '
            if isinstance(x_grid, (int, float)):
                is_x_slice = True
                x_grid = r_vectors[device_cross_idx * 3 + 1]['var_values']
                slice_val = r_vectors[device_cross_idx * 3]['var_values']
                slice_str = f'$x = '
            slice_str +=  f'{slice_val:.2e}$; '
            
                
            
            wl_index = min(range(len(wl_vector)), key=lambda i: abs(wl_vector[i]-plot_wl))
            # plt.contour([np.squeeze(r_vectors[0]['var_values']), np.squeeze(r_vectors[1]['var_values'])], 
            #             f_vectors[0]['var_values'][:,:,wl_index])
            # plt.imshow(f_vectors[0]['var_values'][:,:,wl_index])
            Y_grid, X_grid = np.meshgrid(np.squeeze(y_grid)*1e6, np.squeeze(x_grid)*1e6)
            c = ax.pcolormesh(X_grid, Y_grid, 
                              f_vectors[device_cross_idx * 2]['var_values'][:,:,wl_index],
                            cmap='jet', shading='auto')      # cmap='RdYlBu_r' is also good
            plt.axhline(0,color='black', linestyle='-',linewidth=2.2)
            plt.gca().set_aspect('auto')
            
            wl_str = f'{plot_wl*1e9:.0f} nm' if plot_wl < 1e-6 else f'{plot_wl*1e6:.3f} um'
            title_string = r'$E_{norm}$' + ', Device Cross-Section:\n' + slice_str + '$\lambda = $ ' + f'{wl_str}'
            plt.title(title_string)
            plt.xlabel('y (um)' if is_x_slice else 'x (um)')
            plt.ylabel('z (um)')
            
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.25)
            fig.colorbar(c, cax=cax)
            
            fig_width = 8.0
            fig_height = fig_width / np.shape(X_grid)[0] * np.shape(X_grid)[1]
            l = ax.figure.subplotpars.left
            r = ax.figure.subplotpars.right
            t = ax.figure.subplotpars.top
            b = ax.figure.subplotpars.bottom
            figw = float(fig_width)/(r-l)
            figh = float(fig_height)/(t-b)
            ax.figure.set_size_inches(figw, figh, forward=True)
            plt.tight_layout()
            
            #plt.show(block=False)
            #plt.show()
            
            plot_subfolder_name = 'device_cross_section_spectra'
            if not os.path.isdir(plot_directory_location + '/' + plot_subfolder_name):
                os.makedirs(plot_directory_location + '/' + plot_subfolder_name)
                
            plt.savefig(plot_directory_location + '/' + plot_subfolder_name\
                + '/' + isolate_filename(job_names[job_idx]).replace('.fsp', '')\
                + '_Enorm_' + f'{device_cross_idx}' + '_' + f'{wl_str}' + '.png',
                bbox_inches='tight')
            print(f'Exported: Device Cross Section Enorm Image {device_cross_idx} at wavelength ' + wl_str)

        #* Create real_E_x images
        display_vector_labels = ['x','y','z']
        display_vector_component = 0            # 0 for x, 1 for y, 2 for z
        
        for display_vector_component in range(0,3):     # range(0,3):
            for plot_wl in plot_wavelengths:
                plot_wl = float(plot_wl)
                
                fig, ax = plt.subplots()
                
                is_x_slice = False
                y_grid = r_vectors[device_cross_idx * 3 + 2]['var_values']                   # Plot the z-axis as the vertical
                x_grid = r_vectors[device_cross_idx * 3]['var_values']                       # Check if it's the x-axis or y-axis that is the slice; plot the non-slice as the horizontal of the image plot
                slice_val = r_vectors[device_cross_idx * 3 + 1]['var_values']
                slice_str = f'$y = '
                if isinstance(x_grid, (int, float)):
                    is_x_slice = True
                    x_grid = r_vectors[device_cross_idx * 3 + 1]['var_values']
                    slice_val = r_vectors[device_cross_idx * 3]['var_values']
                    slice_str = f'$x = '
                slice_str +=  f'{slice_val:.2e}$; '
                    
                
                wl_index = min(range(len(wl_vector)), key=lambda i: abs(wl_vector[i]-plot_wl))
                # plt.contour([np.squeeze(r_vectors[0]['var_values']), np.squeeze(r_vectors[1]['var_values'])], 
                #             f_vectors[0]['var_values'][:,:,wl_index])
                # plt.imshow(f_vectors[0]['var_values'][:,:,wl_index])
                Y_grid, X_grid = np.meshgrid(np.squeeze(y_grid)*1e6, np.squeeze(x_grid)*1e6)
                c = ax.pcolormesh(X_grid, Y_grid, 
                                  np.real(f_vectors[device_cross_idx * 2 + 1]['var_values'][display_vector_component][:,:,wl_index]),
                                cmap='jet', shading='auto')      # cmap='RdYlBu_r' is also good
                plt.axhline(0,color='black', linestyle='-',linewidth=2.2)
                plt.gca().set_aspect('auto')
                
                wl_str = f'{plot_wl*1e9:.0f} nm' if plot_wl < 1e-6 else f'{plot_wl*1e6:.3f} um'
                title_string = r'$Re(E_x)$' + ', Device Cross-Section:\n' + slice_str + '$\lambda = $ ' + f'{wl_str}'
                plt.title(title_string)
                plt.xlabel('y (um)' if is_x_slice else 'x (um)')
                plt.ylabel('z (um)')
                
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.25)
                fig.colorbar(c, cax=cax)
                
                fig_width = 8.0
                fig_height = fig_width / np.shape(X_grid)[0] * np.shape(X_grid)[1]
                l = ax.figure.subplotpars.left
                r = ax.figure.subplotpars.right
                t = ax.figure.subplotpars.top
                b = ax.figure.subplotpars.bottom
                figw = float(fig_width)/(r-l)
                figh = float(fig_height)/(t-b)
                ax.figure.set_size_inches(figw, figh, forward=True)
                plt.tight_layout()
                
                #plt.show(block=False)
                #plt.show()
                
                plot_subfolder_name = 'device_cross_section_spectra'
                if not os.path.isdir(plot_directory_location + '/' + plot_subfolder_name):
                    os.makedirs(plot_directory_location + '/' + plot_subfolder_name)
                    
                plt.savefig(plot_directory_location + '/' + plot_subfolder_name\
                    + '/' + isolate_filename(job_names[job_idx]).replace('.fsp', '')\
                    + '_realE' + f'{display_vector_labels[display_vector_component]}' + '_' + f'{device_cross_idx}' + '_' + f'{wl_str}' + '.png',
                    bbox_inches='tight')
                print(f'Exported: Device Cross Section E{display_vector_labels[display_vector_component]}_real Image {device_cross_idx} at wavelength ' + wl_str)
        
        plt.close()
    
    gc.collect()
    
    return fig, ax

def plot_src_spill_power_spectrum(plot_data, job_idx):
    '''For the given job index, plots the sourcepower that misses the device corresponding to that job.'''
    
    sp = SpectrumPlot(plot_data)

    plot_labels = ['Source Spill Power']
    sp.append_line_data(plot_labels=plot_labels)
    
    sp.assign_title('Source Spill Power', job_idx)
    sp.assign_axis_labels('Wavelength (um)', 'Power')
    
    sp.export_plot_config('src_spill_power_spectra', job_idx)
    return sp.fig, sp.ax

def plot_device_transmission_spectrum(plot_data, job_idx):
    '''For the given job index, plots the device transmission spectrum corresponding to that job.'''
    
    sp = SpectrumPlot(plot_data)

    sp.append_line_data()
    
    sp.assign_title('Device Transmission', job_idx)
    sp.assign_axis_labels('Wavelength (um)', 'Device Transmission')
    
    sp.export_plot_config('device_transmission_spectra', job_idx)
    return sp.fig, sp.ax

def plot_overall_reflection_spectrum(plot_data, job_idx):
    '''For the given job index, plots the overall reflection spectrum corresponding to that job.'''
    
    sp = SpectrumPlot(plot_data)

    sp.append_line_data()
    
    sp.assign_title('Overall Device Reflection', job_idx)
    sp.assign_axis_labels('Wavelength (um)', 'Overall Reflection')
    
    sp.export_plot_config('overall_reflection_spectra', job_idx)
    return sp.fig, sp.ax

def plot_crosstalk_power_spectrum(plot_data, job_idx):
    '''For the given job index, plots the crosstalk power corresponding to that job.'''
    
    sp = SpectrumPlot(plot_data)

    plot_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plot_labels = ['NW', 'N', 'NE', 'W', 'Center', 'E', 'SW', 'S', 'SE']
    sp.append_line_data(plot_colors, plot_labels)
    
    sp.assign_title('Device Crosstalk Power', job_idx)
    sp.assign_axis_labels('Wavelength (um)', 'Normalized Power')
    
    sp.export_plot_config('crosstalk_power_spectra', job_idx)
    return sp.fig, sp.ax

def plot_side_power_spectrum(plot_data, job_idx):
    '''For the given job index, plots the device side scattering corresponding to that job.'''
    
    sp = SpectrumPlot(plot_data)

    sp.append_line_data()
    
    sp.assign_title('Device Side Scattering', job_idx)
    sp.assign_axis_labels('Wavelength (um)', 'Side Scattering Power')
    
    sp.export_plot_config('side_power_spectra', job_idx)
    return sp.fig, sp.ax

def plot_focal_power_spectrum(plot_data, job_idx):
    '''For the given job index, plots the power falling on the desired focal region corresponding to that job.'''
    
    sp = SpectrumPlot(plot_data)

    sp.append_line_data()
    
    sp.assign_title('Focal Region Power', job_idx)
    sp.assign_axis_labels('Wavelength (um)', 'Focal Region Power')
    
    sp.export_plot_config('focal_region_power_spectra', job_idx)
    return sp.fig, sp.ax

def plot_oblique_scattering_spectrum(plot_data, job_idx, only_total = False):
    '''For the given job index, plots the oblique scattering corresponding to that job.'''
    
    sp = SpectrumPlot(plot_data)

    plot_labels = ['E', 'N', 'W', 'S', 'Total']
    #sp.append_line_data(plot_labels=plot_labels)
    
    if only_total:
        start_index = len(sp.f_vectors)-1
    else:
        start_index = 0
    for plot_idx in range(start_index, len(sp.f_vectors)):
        line_data = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'y_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'cutoff': None,
                     'color': None,
                     'alpha': 1.0,
                     'legend': None,
                     'marker_style': marker_style
                }
        
        line_data['x_axis']['values'] = sp.r_vectors[0]['var_values']
        line_data['x_axis']['factor'] = 1e6
        line_data['y_axis']['values'] = sp.f_vectors[plot_idx]['var_values']
        line_data['cutoff'] = slice(0, len(line_data['x_axis']['values']))
        
        line_data['legend'] = 'Oblique Scattering ' + plot_labels[plot_idx]
        
        sp.plot_config['lines'].append(line_data)
    
    sp.assign_title('Oblique Scattering', job_idx)
    sp.assign_axis_labels('Wavelength (um)', 'Oblique Scattering Power')
    
    sp.export_plot_config('oblique_scatt_spectra', job_idx)
    return sp.fig, sp.ax

def plot_scatter_plane_power_spectrum(plot_data, job_idx):
    '''For the given job index, plots the device transmission spectrum corresponding to that job.'''
    
    sp = SpectrumPlot(plot_data)

    sp.append_line_data()
    
    sp.assign_title('Focal Scatter Plane Power', job_idx)
    sp.assign_axis_labels('Wavelength (um)', 'Focal Scatter Plane Power')
    
    sp.export_plot_config('scatter_plane_power', job_idx)
    return sp.fig, sp.ax

def plot_exit_power_distribution_spectrum(plot_data, job_idx):
    '''For the given job index, plots the power spectrum for focal region power and oblique scattering power, corresponding to that job.
    The two are normalized against their sum.'''
    
    sp = SpectrumPlot(plot_data)

    plot_colors = []
    plot_labels = []
    for f_vector in plot_data['f']:
        plot_labels.append(f_vector['var_name'])
    
    sp.append_line_data(None, plot_labels)
    
    sp.assign_title('Power Distribution at Focal Plane', job_idx)
    sp.assign_axis_labels('Wavelength (um)', 'Normalized Power')
    
    sp.export_plot_config('exit_power_distribution_spectra', job_idx)
    return sp.fig, sp.ax
        
def plot_device_rta_spectrum(plot_data, job_idx):
    '''For the given job index, plots the device RTA and all power components in and out, corresponding to that job.
    The normalization factor is the power through the device input aperture.'''
    
    sp = SpectrumPlot(plot_data)

    plot_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plot_labels = ['Device Reflection', 'Side Scattering E', 'Side Scattering N', 'Side Scattering W', 'Side Scattering S',
                   'Focal Region Power', 'Oblique Scattering Power', 'Absorbed Power']
    sp.append_line_data(plot_colors, plot_labels)
    
    sp.assign_title('Device RTA', job_idx)
    sp.assign_axis_labels('Wavelength (um)', 'Normalized Power')
    
    sp.export_plot_config('device_rta_spectra', job_idx)
    return sp.fig, sp.ax


# -- Sweep Plot Functions (Overall)

def plot_function_sweep(plot_data, var_code, function_name):
    '''Given N swept variables, creates a series of plots all with the same x-axis.
    var_code selects the variable of length l_i that is to be the x-axis for every plot.
    This function then creates l_1*l_2*...*l_{j \neq i}*...*l_N plots.'''
    
    # Finds occurrence of var_code in keys of sweep_parameters dictionary, gets index
    r_vector_value_idx = [index for (index, item) in enumerate(sweep_parameters.keys()) if var_code in item][0]
    
    temp_sweep_parameters_shape = sweep_parameters_shape.copy()         # Avoid altering original
    del temp_sweep_parameters_shape[r_vector_value_idx]                 # Deletes entry of tuple corresponding to sweep variable
    
    # We are creating an (N-1)-D array and iterating through, i.e. creating a plot with the sweep variable as the x-axis, for each combination
    # of the other variables that are not being plotted as the x-axis.
    for idx, x in np.ndenumerate(np.zeros(temp_sweep_parameters_shape)):
        slice_coords = list(idx).copy()
        slice_coords.insert(r_vector_value_idx, slice(None))            # Here we create the slice object that is used to slice the ndarray accordingly
        fig, ax = globals()[function_name](plot_data, slice_coords, False)
    
    #* Example:
    # if sweep_parameters_shape = (6,9,2), and we are iterating over the variable at index 1,
    # we pop it to get (6,2), and then np.ndenumerate gives
    # (0,slice(None),0), (0,slice(None),1), (0,slice(None),2),
    # (1,slice(None),0), (1,slice(None),1), (1,slice(None),2),
    #                            ...
    # (6,slice(None),0), (6,slice(None),1), (6,slice(None),2)
    # and we call the 1D plotting function using each of these slice objects.
    
    return fig, ax

def plot_sorting_eff_sweep_1d(plot_data, slice_coords, plot_stdDev = True):
    '''Produces a 1D sorting efficiency plot using given coordinates to slice the N-D array of data contained in plot_data.
    X-axis i.e. sweep variable is given by the index of the entry of type 'slice' in the slice_coords.'''
    
    sp = SweepPlot(plot_data, slice_coords)
    
    plot_colors = ['blue', 'green', 'red', 'gray']
    plot_labels = ['Blue', 'Green (x-pol.)', 'Red', 'Green (y-pol.)']
    sp.append_line_data(slice_coords, plot_stdDev, plot_colors, plot_labels)
    
    # # Whichever entry is a slice means that variable in r_vectors is the x-axis of this plot
    # r_vector_value_idx = [index for (index , item) in enumerate(slice_coords) if isinstance(item, slice)][0]
    # r_vectors = list(plot_data['r'].values())[r_vector_value_idx]
    # f_vectors = plot_data['f']              # This is a list of N-D arrays, containing the function values for each parameter combination
    
    # fig, ax = plt.subplots()
    # plot_config = {'title': None,
    #                'x_axis': {'label':'', 'limits':[]},
    #                'y_axis': {'label':'', 'limits':[]},
    #                'lines': []
    #             }
    
    # plot_colors = ['blue', 'green', 'red', 'gray']
    # plot_labels = ['Blue', 'Green (x-pol.)', 'Red', 'Green (y-pol.)']
    # normalize_against_max = False
    
    # for plot_idx in range(0, len(f_vectors)-0):
    #     line_data = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
    #                  'y_axis': {'values': None, 'factor': 1, 'offset': 0},
    #                  'cutoff': None,
    #                  'color': None,
    #                  'alpha': 1.0,
    #                  'legend': None,
    #                  'marker_style': marker_style
    #             }
        
        
    #     y_plot_data = f_vectors[plot_idx]['var_values']
    #     y_plot_data = np.reshape(y_plot_data, (len(y_plot_data), 1))
    #     y_plot_data = y_plot_data[tuple(slice_coords)]
        
    #     normalization_factor = np.max(y_plot_data) if normalize_against_max else 1
    #     line_data['y_axis']['values'] = y_plot_data / normalization_factor
    #     line_data['x_axis']['values'] = r_vectors['var_values']
    #     # line_data['cutoff'] = slice(0, len(line_data['x_axis']['values']))
    #     line_data['cutoff'] = slice(0 + cutoff_1d_sweep_offset[0], len(line_data['x_axis']['values']) + cutoff_1d_sweep_offset[1])
        
    #     line_data['color'] = plot_colors[plot_idx]
    #     line_data['legend'] = plot_labels[plot_idx]
        
    #     plot_config['lines'].append(line_data)
        
        
    #     if plot_stdDev and f_vectors[plot_idx]['statistics'] in ['mean']:
    #         line_data_2 = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
    #                  'y_axis': {'values': None, 'factor': 1, 'offset': 0},
    #                  'cutoff': None,
    #                  'color': None,
    #                  'alpha': 0.3,
    #                  'legend': '_nolegend_',
    #                  'marker_style': dict(linestyle='-', linewidth=1.0)
    #             }
        
    #         line_data_2['x_axis']['values'] = r_vectors['var_values']
    #         line_data_2['y_axis']['values'] = (y_plot_data + f_vectors[plot_idx]['var_stdevs']) / normalization_factor
    #         # line_data_2['cutoff'] = slice(0, len(line_data_2['x_axis']['values']))
    #         line_data_2['cutoff'] = slice(0 + cutoff_1d_sweep_offset[0], len(line_data_2['x_axis']['values']) + cutoff_1d_sweep_offset[1])
            
    #         colors_so_far = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    #         line_data_2['color'] = colors_so_far[-1]
            
    #         plot_config['lines'].append(line_data_2)
            
    #         line_data_3 = copy.deepcopy(line_data_2)
    #         line_data_3['y_axis']['values'] = (y_plot_data - f_vectors[plot_idx]['var_stdevs']) / normalization_factor
    #         plot_config['lines'].append(line_data_3)
    
    sp.assign_title('Sorting Efficiency - Parameter Range:')
    sp.assign_axis_labels(slice_coords, 'Sorting Efficiency')
    
    # title_string = 'Sorting Efficiency - Parameter Range:'
    # for sweep_param_value in list(sweep_parameters.values()):
    #     optimized_value = sweep_param_value['var_values'][sweep_param_value['peakInd']]
    #     title_string += '\n' + sweep_param_value['var_name'] + f': Optimized at {optimized_value}'
    # plot_config['title'] = title_string
    
    # sweep_variable_idx = [index for (index, item) in enumerate(slice_coords) if type(item) is slice][0]
    # plot_config['x_axis']['label'] = list(sweep_parameters.values())[sweep_variable_idx]['var_name']
    # plot_config['y_axis']['label'] = 'Sorting Efficiency'
    
    sp.export_plot_config('sorting_eff_sweep_', slice_coords)
    
    # fig, ax = enter_plot_data_1d(plot_config, fig, ax)
    
    # plot_subfolder_name = ''
    # if not os.path.isdir(plot_directory_location + '/' + plot_subfolder_name):
    #     os.makedirs(plot_directory_location + '/' + plot_subfolder_name)
    # plt.savefig(plot_directory_location + '/' + plot_subfolder_name\
    #     + '/' + 'sorting_eff_sweep_' + create_parameter_filename_string(slice_coords) + '.png',
    #     bbox_inches='tight')
    # print('Exported: Sorting Efficiency Sweep: ' + create_parameter_filename_string(slice_coords))
    
    return sp.fig, sp.ax

def plot_sorting_transmission_sweep_1d(plot_data, slice_coords, plot_stdDev = True):
    '''Produces a 1D sorting (transmission) efficiency plot using given coordinates to slice the N-D array of data contained in plot_data.
    X-axis i.e. sweep variable is given by the index of the entry of type 'slice' in the slice_coords.'''
    
    sp = SweepPlot(plot_data, slice_coords)
    
    plot_colors = ['blue', 'green', 'red', 'gray']
    plot_labels = ['Blue', 'Green (x-pol.)', 'Red', 'Green (y-pol.)']
    sp.append_line_data(slice_coords, plot_stdDev, plot_colors, plot_labels)
    
    sp.assign_title('Sorting Transmission Efficiency - Parameter Range:')
    sp.assign_axis_labels(slice_coords, 'Sorting Transmission Efficiency')
    
    sp.export_plot_config('sorting_trans_eff_sweep_', slice_coords)
    
    return sp.fig, sp.ax

def plot_sorting_meanband_transmission_sweep_1d(plot_data, slice_coords, plot_stdDev = True):
    '''Produces a 1D sorting (transmission) efficiency plot using given coordinates to slice the N-D array of data contained in plot_data.
    Function is transmission averaged within the spectral band.
    X-axis i.e. sweep variable is given by the index of the entry of type 'slice' in the slice_coords.'''
    
    sp = SweepPlot(plot_data, slice_coords)
    
    plot_colors = ['blue', 'green', 'red', 'gray']
    #plot_labels = ['Blue', 'Green (x-pol.)', 'Red', 'Green (y-pol.)']
    plot_labels = ['Blue', 'Green', 'Red']
    normalize_against_max = False
    
    # Here we have 9 f_vectors: B->{B,G,R}, G->{B,G,R}, R->{B,G,R}
    # where for e.g. RB is the transmission of R quadrant within the B spectral band
    for f_idx, plot_idx in enumerate([0, 4, 8]):  # Hard coded: corresponds to BB, GG, RR
        line_data = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'y_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'cutoff': None,
                     'color': None,
                     'alpha': 1.0,
                     'legend': None,
                     'marker_style': marker_style
                }
        
        y_plot_data = sp.f_vectors[plot_idx]['var_values']
        y_plot_data = np.reshape(y_plot_data, (len(y_plot_data), 1))
        y_plot_data = y_plot_data[tuple(slice_coords)]
        
        normalization_factor = np.max(y_plot_data) if normalize_against_max else 1
        line_data['y_axis']['values'] = y_plot_data / normalization_factor
        line_data['x_axis']['values'] = sp.r_vectors['var_values']
        # line_data['cutoff'] = slice(0, len(line_data['x_axis']['values']))
        line_data['cutoff'] = slice(0 + cutoff_1d_sweep_offset[0], len(line_data['x_axis']['values']) + cutoff_1d_sweep_offset[1])
                
        line_data['color'] = plot_colors[f_idx]
        line_data['legend'] = plot_labels[f_idx]
        
        sp.plot_config['lines'].append(line_data)
        
        
        if plot_stdDev and f_vectors[plot_idx]['statistics'] in ['mean']:
            line_data_2 = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'y_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'cutoff': None,
                     'color': None,
                     'alpha': 0.3,
                     'legend': '_nolegend_',
                     'marker_style': dict(linestyle='-', linewidth=1.0)
                }
        
            line_data_2['x_axis']['values'] = sp.r_vectors['var_values']
            line_data_2['y_axis']['values'] = (y_plot_data + sp.f_vectors[plot_idx]['var_stdevs']) / normalization_factor
            # line_data_2['cutoff'] = slice(0, len(line_data_2['x_axis']['values']))
            line_data_2['cutoff'] = slice(0 + cutoff_1d_sweep_offset[0], len(line_data_2['x_axis']['values']) + cutoff_1d_sweep_offset[1])
            
            colors_so_far = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            line_data_2['color'] = colors_so_far[-1]
            
            sp.plot_config['lines'].append(line_data_2)
            
            line_data_3 = copy.deepcopy(line_data_2)
            line_data_3['y_axis']['values'] = (y_plot_data - sp.f_vectors[plot_idx]['var_stdevs']) / normalization_factor
            sp.plot_config['lines'].append(line_data_3)
    
    
    sp.assign_title('Sorting Transmission Efficiency (Mean In-Band) - Parameter Range:')
    sp.assign_axis_labels(slice_coords, 'Sorting Transmission Efficiency')
    
    sp.export_plot_config('sorting_trans_eff_meanband_sweep_', slice_coords)
    
    return sp.fig, sp.ax

def plot_sorting_contrastband_transmission_sweep_1d(plot_data, slice_coords, plot_stdDev = True, add_opp_polarizations=True):
    '''Produces a 1D sorting (transmission) efficiency plot using given coordinates to slice the N-D array of data contained in plot_data.
    Function is contrast i.e. average transmission within the corresponding spectral band, normalized against sum of noise i.e. other quadrants transmission within that same spectral band.
    Mathematically, contrast_R = (T_good-T_bad)/(T_good+T_bad) where T_good = T_RR, T_bad = (T_RB+T_RG) ... where for e.g. RB is the transmission of R quadrant within the B spectral band
    X-axis i.e. sweep variable is given by the index of the entry of type 'slice' in the slice_coords.'''
    
    sp = SweepPlot(plot_data, slice_coords)
    
    plot_colors = ['blue', 'green', 'red', 'gray']
    #plot_labels = ['Blue', 'Green (x-pol.)', 'Red', 'Green (y-pol.)']
    plot_labels = ['Blue', 'Green', 'Red']
    normalize_against_max = False
    
    num_sweep_values = num_adjoint_sources if not add_opp_polarizations else num_bands
    
    # Here we have 9 f_vectors: B->{B,G,R}, G->{B,G,R}, R->{B,G,R}
    # where for e.g. RB is the transmission of R quadrant within the B spectral band
    for plot_idx in range(0, num_sweep_values):  
        line_data = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'y_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'cutoff': None,
                     'color': None,
                     'alpha': 1.0,
                     'legend': None,
                     'marker_style': marker_style
                }
        
        # Get X_good, then X_bad is the sum of the other transmissions, then calculate contrast accordingly
        f_idx = plot_idx*num_sweep_values + (plot_idx)
        X_good = sp.f_vectors[f_idx]['var_values']
        X_bad = -X_good
        for bad_idx in range(plot_idx*num_sweep_values, (plot_idx+1)*num_sweep_values):
            X_bad += sp.f_vectors[bad_idx]['var_values']
        
        # Testing of contrast bug  
        # t_idx = 34
        # print(sweep_parameters['th']['var_values'][t_idx])
        # print(f_vectors[6]['var_values'][t_idx])
        # print(f_vectors[7]['var_values'][t_idx])
        # print(f_vectors[8]['var_values'][t_idx])
        # print(X_good[t_idx])
        # print(X_bad[t_idx])
        # print((X_good[t_idx]-X_bad[t_idx]) / (X_good[t_idx]+X_bad[t_idx]))
        
        y_plot_data = (X_good - X_bad)/(X_good + X_bad)
        y_plot_data = np.reshape(y_plot_data, (len(y_plot_data), 1))
        y_plot_data = y_plot_data[tuple(slice_coords)]
        
        normalization_factor = np.max(y_plot_data) if normalize_against_max else 1
        line_data['y_axis']['values'] = y_plot_data / normalization_factor
        line_data['x_axis']['values'] = sp.r_vectors['var_values']
        # line_data['cutoff'] = slice(0, len(line_data['x_axis']['values']))
        line_data['cutoff'] = slice(0 + cutoff_1d_sweep_offset[0], len(line_data['x_axis']['values']) + cutoff_1d_sweep_offset[1])
                
        line_data['color'] = plot_colors[plot_idx]
        line_data['legend'] = plot_labels[plot_idx]
        
        sp.plot_config['lines'].append(line_data)
        
        
        if plot_stdDev and sp.f_vectors[plot_idx]['statistics'] in ['mean']:
            line_data_2 = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'y_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'cutoff': None,
                     'color': None,
                     'alpha': 0.3,
                     'legend': '_nolegend_',
                     'marker_style': dict(linestyle='-', linewidth=1.0)
                }
        
            line_data_2['x_axis']['values'] = sp.r_vectors['var_values']
            line_data_2['y_axis']['values'] = (y_plot_data + sp.f_vectors[plot_idx]['var_stdevs']) / normalization_factor
            # line_data_2['cutoff'] = slice(0, len(line_data_2['x_axis']['values']))
            line_data_2['cutoff'] = slice(0 + cutoff_1d_sweep_offset[0], len(line_data_2['x_axis']['values']) + cutoff_1d_sweep_offset[1])
            
            colors_so_far = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            line_data_2['color'] = colors_so_far[-1]
            
            sp.plot_config['lines'].append(line_data_2)
            
            line_data_3 = copy.deepcopy(line_data_2)
            line_data_3['y_axis']['values'] = (y_plot_data - sp.f_vectors[plot_idx]['var_stdevs']) / normalization_factor
            sp.plot_config['lines'].append(line_data_3)
            
    
    sp.assign_title('Sorting Transmission Efficiency (Contrast) - Parameter Range:')
    sp.assign_axis_labels(slice_coords, 'Sorting Trans.Eff. Contrast')
    
    sp.export_plot_config('sorting_trans_eff_contrastband_sweep_', slice_coords)
    
    return sp.fig, sp.ax

def plot_src_spill_power_sweep_1d(plot_data, slice_coords, plot_stdDev = True):
    '''Produces a 1D source spill power plot using given coordinates to slice the N-D array of data contained in plot_data.
    X-axis i.e. sweep variable is given by the index of the entry of type 'slice' in the slice_coords.'''
    
    sp = SweepPlot(plot_data, slice_coords)
    
    plot_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plot_labels = ['Source Spill Power']
    sp.append_line_data(slice_coords, plot_stdDev, plot_colors, plot_labels)
    
    sp.assign_title('Source Spill Power - Parameter Range:')
    sp.assign_axis_labels(slice_coords, 'Source Spill Power')
    
    sp.export_plot_config('src_spill_power_sweep_', slice_coords)
    
    return sp.fig, sp.ax

def plot_device_transmission_sweep_1d(plot_data, slice_coords, plot_stdDev = True):
    '''Produces a 1D device transmission plot using given coordinates to slice the N-D array of data contained in plot_data.
    X-axis i.e. sweep variable is given by the index of the entry of type 'slice' in the slice_coords.'''
    
    sp = SweepPlot(plot_data, slice_coords)
    
    plot_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plot_labels = ['Device Transmission']
    sp.append_line_data(slice_coords, plot_stdDev, plot_colors, plot_labels)
    
    sp.assign_title('Device Transmission - Parameter Range:')
    sp.assign_axis_labels(slice_coords, 'Device Transmission')
    
    sp.export_plot_config('dev_transmission_sweep_', slice_coords)
    
    return sp.fig, sp.ax

def plot_overall_reflection_sweep_1d(plot_data, slice_coords, plot_stdDev = True):
    '''Produces a 1D FDTD-region reflection plot using given coordinates to slice the N-D array of data contained in plot_data.
    X-axis i.e. sweep variable is given by the index of the entry of type 'slice' in the slice_coords.'''
    
    sp = SweepPlot(plot_data, slice_coords)
    
    plot_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plot_labels = ['Overall Reflection']
    sp.append_line_data(slice_coords, plot_stdDev, plot_colors, plot_labels)
    
    sp.assign_title('Overall Device Reflection - Parameter Range:')
    sp.assign_axis_labels(slice_coords, 'Overall Reflection')
    
    sp.export_plot_config('overall_ref_sweep_', slice_coords)
    
    return sp.fig, sp.ax

def plot_side_power_sweep_1d(plot_data, slice_coords, plot_stdDev = True):
    '''Produces a 1D side scattering power plot using given coordinates to slice the N-D array of data contained in plot_data.
    X-axis i.e. sweep variable is given by the index of the entry of type 'slice' in the slice_coords.'''
    
    sp = SweepPlot(plot_data, slice_coords)
    
    plot_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plot_labels = ['Side Scattering E', 'Side Scattering N', 'Side Scattering W', 'Side Scattering S']
    sp.append_line_data(slice_coords, plot_stdDev, plot_colors, plot_labels)
    
    sp.assign_title('Side Scattering - Parameter Range:')
    sp.assign_axis_labels(slice_coords, 'Side Scattering Power')
    
    sp.export_plot_config('side_power_sweep_', slice_coords)
    
    return sp.fig, sp.ax

def plot_focal_power_sweep_1d(plot_data, slice_coords, plot_stdDev = True):
    '''Produces a 1D focal region power plot using given coordinates to slice the N-D array of data contained in plot_data.
    X-axis i.e. sweep variable is given by the index of the entry of type 'slice' in the slice_coords.'''
    
    sp = SweepPlot(plot_data, slice_coords)
    
    plot_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plot_labels = ['Focal Region Power']
    sp.append_line_data(slice_coords, plot_stdDev, plot_colors, plot_labels)
    
    sp.assign_title('Focal Region Power - Parameter Range:')
    sp.assign_axis_labels(slice_coords, 'Focal Region Power')
    
    sp.export_plot_config('focal_power_sweep_', slice_coords)
    
    return sp.fig, sp.ax

def plot_scatter_plane_power_sweep_1d(plot_data, slice_coords, plot_stdDev = True):
    '''Produces a 1D scatter plane region power plot using given coordinates to slice the N-D array of data contained in plot_data.
    X-axis i.e. sweep variable is given by the index of the entry of type 'slice' in the slice_coords.'''
    
    sp = SweepPlot(plot_data, slice_coords)
    
    plot_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plot_labels = ['Scatter Plane Power']
    sp.append_line_data(slice_coords, plot_stdDev, plot_colors, plot_labels)
    
    sp.assign_title('Focal Plane Scattering Power - Parameter Range:')
    sp.assign_axis_labels(slice_coords, 'Focal Plane Scattering Power')
    
    sp.export_plot_config('focal_power_sweep_', slice_coords)
    
    return sp.fig, sp.ax

def plot_oblique_scattering_sweep_1d(plot_data, slice_coords, plot_stdDev = True, only_total = False):
    '''Produces a 1D oblique scattering power plot using given coordinates to slice the N-D array of data contained in plot_data.
    X-axis i.e. sweep variable is given by the index of the entry of type 'slice' in the slice_coords.'''
    
    sp = SweepPlot(plot_data, slice_coords)
    
    plot_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plot_labels = ['E', 'N', 'W', 'S', 'Total']
    sp.append_line_data(slice_coords, plot_stdDev, plot_colors, plot_labels)
    
    sp.assign_title('Oblique Scattering - Parameter Range:')
    sp.assign_axis_labels(slice_coords, 'Oblique Scattering Power')
    
    sp.export_plot_config('oblique_scatt_sweep_', slice_coords)
    
    return sp.fig, sp.ax

def plot_exit_power_distribution_sweep_1d(plot_data, slice_coords, plot_stdDev = True):
    '''Produces a 1D power plot of focal plane power distribution using given coordinates to slice the N-D array 
    of data contained in plot_data.
    X-axis i.e. sweep variable is given by the index of the entry of type 'slice' in the slice_coords.'''
    
    sp = SweepPlot(plot_data, slice_coords)
    
    plot_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plot_labels = ['Focal Region Power', 'Oblique Scattering']
    sp.append_line_data(slice_coords, plot_stdDev, plot_colors, plot_labels)
    
    sp.assign_title('Power at Focal Plane - Parameter Range:')
    sp.assign_axis_labels(slice_coords, 'Normalized Focal Plane Power')
    
    sp.export_plot_config('exit_power_dist_sweep_', slice_coords)
    
    return sp.fig, sp.ax

def plot_device_rta_sweep_1d(plot_data, slice_coords, plot_stdDev = True):
    '''Produces a 1D power plot of device RTA using given coordinates to slice the N-D array 
    of data contained in plot_data.
    X-axis i.e. sweep variable is given by the index of the entry of type 'slice' in the slice_coords.'''
    
    sp = SweepPlot(plot_data, slice_coords)
    
    mean_vals = []
    for f_vec in plot_data['f']:
        var_value = f_vec['var_values'][0]
        mean_vals.append(var_value)
    print('Device RTA values are:')
    print(mean_vals)
    print('with total side scattering at:')
    print(sum(mean_vals[1:5]))
    
    plot_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plot_labels = ['Device Reflection', 'Side Scattering E', 'Side Scattering N', 'Side Scattering W', 'Side Scattering S',
                   'Focal Region Power', 'Oblique Scattering Power', 'Absorbed Power']
    sp.append_line_data(slice_coords, plot_stdDev, plot_colors, plot_labels)
    
    sp.assign_title('Device RTA - Parameter Range:')
    sp.assign_axis_labels(slice_coords, 'Normalized Power')
    
    sp.export_plot_config('device_rta_sweep_', slice_coords)
    
    return sp.fig, sp.ax

# #
# # Set up some numpy arrays to handle all the data we will pull out of the simulation.
# # These are just to keep track of the arrays we create. We re-initialize these right before their respective loops
# #

forward_e_fields = {}
focal_data = {}
quadrant_efield_data = {}
quadrant_transmission_data = {}

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
                    elif variable_name in ['angle_phi']:
                        change_source_theta(angle_theta, variable_value)
                    elif variable_name in ['beam_radius_um']:
                        change_source_beam_radius(variable_value)
                    elif variable_name in ['sidewall_thickness_um']:
                        change_sidewall_thickness(variable_value)
                    elif variable_name in ['divergence_angle']:
                        change_divergence_angle(variable_value)
                        
                    
                for xy_idx in range(0, 1): # range(0,2) if considering source with both polarizations
                    disable_all_sources()
                    disable_objects(disable_object_list)
                    fdtd_hook.select('design_import')
                    fdtd_hook.set('enabled', 1)

                    fdtd_hook.select(forward_sources[xy_idx]['name'])
                    fdtd_hook.set('enabled', 1)
                    
                    job_name = 'sweep_job_' + \
                        parameter_filename_string + '.fsp'
                    fdtd_hook.save(projects_directory_location +
                                    "/optimization.fsp")
                    print('Saved out ' + job_name)
                    job_names[('sweep', xy_idx, idx)
                                ] = add_job(job_name, jobs_queue)
                    
                    # Disable design import and measure power that misses device
                    fdtd_hook.select('src_spill_monitor')
                    fdtd_hook.set('enabled', 1)
                    fdtd_hook.select('design_import')
                    fdtd_hook.set('enabled', 0)
                    
                    if device_array_num > 1:
                        for dev_arr_idx in range(0, device_array_num**2-1):
                            fdtd_hook.select('design_import_'+str(dev_arr_idx))
                            fdtd_hook.set('enabled', 0)
                    
                    for sw_idx in range(0, num_sidewalls):
                        sm_name = 'sidewall_' + str(sw_idx)
                        sm_mesh_name = 'mesh_sidewall_' + str(sw_idx)
                        
                        try:
                            objExists = fdtd_hook.getnamed(sm_name, 'name')
                            fdtd_hook.select(sm_name)
                            fdtd_hook.set('enabled', 0)
                        except Exception as ex:
                            pass
                        try:
                            objExists = fdtd_hook.getnamed(sm_mesh_name, 'name')
                            fdtd_hook.select(sm_mesh_name)
                            fdtd_hook.set('enabled', 0)
                        except Exception as ex:
                            pass
                    
                    job_name = 'sweep_job_nodev_' + \
                        parameter_filename_string + '.fsp'
                    fdtd_hook.save(projects_directory_location +
                                    "/optimization.fsp")
                    print('Saved out ' + job_name)
                    job_names[('nodev', xy_idx, idx)
                                ] = add_job(job_name, jobs_queue)
                    

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
                current_job_idx = idx
                print(f'----- Accessing Job: Current parameter index is {current_job_idx}')
                
                # Load each individual job
                for xy_idx in range(0, 1): # range(0,2) if considering source with both polarizations
                    if start_from_step == 0:
                        fdtd_hook.load(
                            job_names[('sweep', xy_idx, idx)])
                    else:
                        # Extract filename, directly declare current directory address
                        
                        # TODO: Undo this bit when no longer build-testing
                        actual_directory = r'C:\Users\Ian\Dropbox\Caltech\Faraon Group\Simulations\Sony Bayer\[v3] crosstalk_initial\Data Processing\th0_PEC_60nm_ext\ares_sony_th0_PEC_60nm_ext_dev'
                        actual_filepath = convert_root_folder(job_names[('sweep',xy_idx,idx)],
                                                actual_directory)
                        #fdtd_hook.load( job_names[ ( 'forward', xy_idx, dispersive_range_idx ) ] )
                        print(f'Loading: {isolate_filename(actual_filepath)}')
                        sys.stdout.flush()
                        fdtd_hook.load(actual_filepath)
                        
                        # fdtd_hook.load(
                        #     convert_root_folder(job_names[('sweep',xy_idx,idx)],
                        #                         projects_directory_location)
                        # )   
                        
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
                                monitor_data['E'] = get_efield_magnitude(monitor['name'])
                                #monitor_data['E_alt'] = get_E
                                monitor_data['P'] = get_overall_power(monitor['name'])
                                monitor_data['T'] = get_transmission_magnitude(monitor['name'])
                                
                                if monitor['name'] in ['spill_plane_monitor', 'scatter_plane_monitor',
                                                    'transmission_monitor_0','transmission_monitor_1','transmission_monitor_2','transmission_monitor_3',
                                                    'transmission_focal_monitor_',
                                                    'device_xsection_monitor_0', 'device_xsection_monitor_1', 'device_xsection_monitor_2', 
                                                    'device_xsection_monitor_3', 'device_xsection_monitor_4', 'device_xsection_monitor_5']:
                                    monitor_data['coords'] = {'x': fdtd_hook.getresult(monitor['name'],'x'),
                                                            'y': fdtd_hook.getresult(monitor['name'],'y'),
                                                            'z': fdtd_hook.getresult(monitor['name'],'z')}
                                
                                if monitor['name'] in ['device_xsection_monitor_0', 'device_xsection_monitor_1', 'device_xsection_monitor_2', 
                                                        'device_xsection_monitor_3', 'device_xsection_monitor_4', 'device_xsection_monitor_5']:
                                    monitor_data['E_real'] = get_efield(monitor['name'])
                                
                            monitors_data[monitor['name']] = monitor_data
                    
                    # # Load also the corresponding job where the device is disabled, to measure spill power
                    if start_from_step == 0:
                        fdtd_hook.load(
                            job_names[('nodev', xy_idx, idx)])
                    else:
                        # Extract filename, directly declare current directory address
                        
                        # TODO: Undo this bit when no longer build-testing
                        actual_directory = r'C:\Users\Ian\Dropbox\Caltech\Faraon Group\Simulations\Sony Bayer\[v3] crosstalk_initial\Data Processing\th0_PEC_60nm_ext\ares_sony_th0_PEC_60nm_ext_dev'
                        actual_filepath = convert_root_folder(job_names[('nodev',xy_idx,idx)],
                                                actual_directory)
                        #fdtd_hook.load( job_names[ ( 'forward', xy_idx, dispersive_range_idx ) ] )
                        print(f'Loading: {isolate_filename(actual_filepath)}')
                        sys.stdout.flush()
                        fdtd_hook.load(actual_filepath)
                        
                        # fdtd_hook.load(
                        #     convert_root_folder(job_names[('nodev',xy_idx,idx)],
                        #                         projects_directory_location)
                        # )
                    
                    # Go through each individual monitor and extract respective data
                    for monitor in monitors:
                        if monitor['name'] in ['incident_aperture_monitor',
                                               'src_spill_monitor']:
                            monitor_data = {}
                            monitor_data['name'] = monitor['name']
                            monitor_data['save'] = monitor['save']
                            
                            if monitor['enabled'] and monitor['getFromFDTD']:
                                monitor_data['E'] = get_efield_magnitude(monitor['name'])
                                #monitor_data['E_alt'] = get_E
                                monitor_data['P'] = get_overall_power(monitor['name'])
                                monitor_data['T'] = get_transmission_magnitude(monitor['name'])
                                
                                if monitor['name'] in ['spill_plane_monitor', 'scatter_plane_monitor',
                                                    'transmission_monitor_0','transmission_monitor_1','transmission_monitor_2','transmission_monitor_3',
                                                    'transmission_focal_monitor_']:
                                    monitor_data['coords'] = {'x': fdtd_hook.getresult(monitor['name'],'x'),
                                                            'y': fdtd_hook.getresult(monitor['name'],'y'),
                                                            'z': fdtd_hook.getresult(monitor['name'],'z')}
                                
                            monitors_data[monitor['name']] = monitor_data
                    
                    
                    # Save out to a dictionary, using job_names as keys
                    jobs_data[job_names[('sweep', xy_idx, idx)]] = monitors_data
            
            
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
                    
                    monitors_data = jobs_data[job_names[('sweep', xy_idx, idx)]]
            
                    plot_data = {}
                    plot_data['wavelength'] = monitors_data['wavelength']
                    plot_data['sourcepower'] = monitors_data['sourcepower']
                    
                    for plot in plots:
                        if plot['enabled']:
                            
                            if plot['name'] in ['sorting_efficiency']:
                                plot_data[plot['name']+'_spectrum'], plot_data[plot['name']+'_sweep'] = \
                                    generate_sorting_eff_spectrum(monitors_data, plot['generate_plot_per_job'], add_opp_polarizations=True)
                                    
                            elif plot['name'] in ['sorting_transmission']:
                                plot_data[plot['name']+'_spectrum'], plot_data[plot['name']+'_sweep'] = \
                                    generate_sorting_transmission_spectrum(monitors_data, plot['generate_plot_per_job'], add_opp_polarizations=True)
                                    
                            elif plot['name'] in ['sorting_transmission_meanband', 'sorting_transmission_contrastband']:
                                band_idxs = partition_bands(monitors_data['wavelength'], [4.5e-7, 5.2e-7, 6.1e-7])
                                
                                plot_data[plot['name']+'_sweep'] = \
                                    generate_inband_val_sweep(monitors_data, plot['generate_plot_per_job'], 
                                                                           statistics='mean', add_opp_polarizations=True,
                                                                           band_idxs=band_idxs)
                                    
                            elif plot['name'] in ['Enorm_focal_plane_image']:
                                plot_data[plot['name']+'_spectrum'], plot_data[plot['name']+'_sweep'] = \
                                    generate_Enorm_focal_plane(monitors_data, plot['generate_plot_per_job'])
                                    
                            elif plot['name'] in ['device_cross_section'] and sweep_parameter_value_array[idx][0] == angle_theta:
                                plot_data[plot['name']+'_spectrum'], plot_data[plot['name']+'_sweep'] = \
                                    generate_device_cross_section_spectrum(monitors_data, plot['generate_plot_per_job'])
                            
                            elif plot['name'] in ['src_spill_power']:
                                plot_data[plot['name']+'_spectrum'], plot_data[plot['name']+'_sweep'] = \
                                    generate_source_spill_power_spectrum(monitors_data, plot['generate_plot_per_job'])
                                
                            elif plot['name'] in ['device_transmission']:
                                plot_data[plot['name']+'_spectrum'], plot_data[plot['name']+'_sweep'] = \
                                    generate_device_transmission_spectrum(monitors_data, plot['generate_plot_per_job'])
                                    
                            elif plot['name'] in ['device_reflection']:
                                plot_data[plot['name']+'_spectrum'], plot_data[plot['name']+'_sweep'] = \
                                    generate_overall_reflection_spectrum(monitors_data, plot['generate_plot_per_job'])
                            
                            elif plot['name'] in ['crosstalk_power']:
                                plot_data[plot['name']+'_spectrum'], plot_data[plot['name']+'_sweep'] = \
                                    generate_crosstalk_power_spectrum(monitors_data, plot['generate_plot_per_job'])
                            
                            elif plot['name'] in ['side_scattering_power']:
                                plot_data[plot['name']+'_spectrum'], plot_data[plot['name']+'_sweep'] = \
                                    generate_side_power_spectrum(monitors_data, plot['generate_plot_per_job'])
                                    
                            elif plot['name'] in ['focal_region_power']:
                                plot_data[plot['name']+'_spectrum'], plot_data[plot['name']+'_sweep'] = \
                                    generate_focal_power_spectrum(monitors_data, plot['generate_plot_per_job'])
                            
                            elif plot['name'] in ['oblique_scattering_power']:
                                plot_data[plot['name']+'_spectrum'], plot_data[plot['name']+'_sweep'] = \
                                    generate_oblique_scattering_power_spectrum(monitors_data, plot['generate_plot_per_job'])        
                                    
                            elif plot['name'] in ['focal_scattering_power']:
                                plot_data[plot['name']+'_spectrum'], plot_data[plot['name']+'_sweep'] = \
                                    generate_scatter_plane_power_spectrum(monitors_data, plot['generate_plot_per_job'])
                                    
                            elif plot['name'] in ['exit_power_distribution']:
                                plot_data[plot['name']+'_spectrum'], plot_data[plot['name']+'_sweep'] = \
                                    generate_exit_power_distribution_spectrum(monitors_data, plot['generate_plot_per_job'])
                            
                            elif plot['name'] in ['device_rta']:
                                plot_data[plot['name']+'_spectrum'], plot_data[plot['name']+'_sweep'] = \
                                    generate_device_rta_spectrum(monitors_data, plot['generate_plot_per_job'])
                                
                    plots_data[job_names[('sweep', xy_idx, idx)]] = plot_data
                        
                        
            print("Completed Step 3: Processed and saved relevant plot data.")
            sys.stdout.flush()
        
        # TODO: Undo the link between Step 3 and 4
        #     backup_all_vars()
            
            
        # #
        # # Step 4: Plot Sweeps across Jobs
        # #
        
        # if start_from_step == 0 or start_from_step == 4:
        #     if start_from_step == 4:
        #         load_backup_vars()
            
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
                            job_name = job_names[('sweep', xy_idx, idx)]
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

            # Purge the nodev job names
            job_names_check = job_names.copy()
            for job_idx, job_name in job_names_check.items():
                if 'nodev' in isolate_filename(job_name):
                    job_names.pop(job_idx)
            
            for job_idx, job_name in job_names.items():
                plot_data = plots_data[job_name]
                
                print('Plotting for Job: ' + isolate_filename(job_names[job_idx]))
                
                for plot in plots:
                    if plot['enabled']:
                        # continue
                        if plot['name'] in ['sorting_efficiency']:
                            plot_sorting_eff_spectrum(plot_data[plot['name']+'_spectrum'], job_idx)
                            # continue
                        elif plot['name'] in ['sorting_transmission']:
                            plot_sorting_transmission_spectrum(plot_data[plot['name']+'_spectrum'], job_idx)
                            
                        elif plot['name'] in ['Enorm_focal_plane_image']:
                            plot_Enorm_focal_plane_image_spectrum(plot_data[plot['name']+'_spectrum'], job_idx,
                                        plot_wavelengths = None,
                                        #plot_wavelengths = monitors_data['wavelength'][[n['index'] for n in plot_data['sorting_efficiency_sweep']]],
                                        #plot_wavelengths = [4.6e-7,5.2e-7, 6.1e-7]
                                        ignore_opp_polarization=False)
                                        # Second part gets the wavelengths of each spectra where sorting eff. peaks
                            #continue
                        
                        elif plot['name'] in ['device_cross_section']:
                            plot_device_cross_section_spectrum(plot_data[plot['name']+'_spectrum'], job_idx,
                                        plot_wavelengths = [4.6e-7,5.2e-7,6.1e-7],
                                        #plot_wavelengths = monitors_data['wavelength'][[n['index'] for n in plot_data['sorting_efficiency_sweep']]],
                                        ignore_opp_polarization=False)
                                        # Second part gets the wavelengths of each spectra where sorting eff. peaks
                            #continue
                        
                        elif plot['name'] in ['src_spill_power']:
                            plot_src_spill_power_spectrum(plot_data[plot['name']+'_spectrum'], job_idx)
                        
                        elif plot['name'] in ['device_transmission']:
                            plot_device_transmission_spectrum(plot_data[plot['name']+'_spectrum'], job_idx)
                            
                        elif plot['name'] in ['device_reflection']:
                            plot_overall_reflection_spectrum(plot_data[plot['name']+'_spectrum'], job_idx)
                        
                        elif plot['name'] in ['crosstalk_power']:
                            plot_crosstalk_power_spectrum(plot_data[plot['name']+'_spectrum'], job_idx)
                        
                        elif plot['name'] in ['side_scattering_power']:
                            plot_side_power_spectrum(plot_data[plot['name']+'_spectrum'], job_idx)
                        
                        elif plot['name'] in ['oblique_scattering_power']:
                            plot_oblique_scattering_spectrum(plot_data[plot['name']+'_spectrum'], job_idx)
                            
                        elif plot['name'] in ['focal_region_power']:
                            plot_focal_power_spectrum(plot_data[plot['name']+'_spectrum'], job_idx)
                        
                        elif plot['name'] in ['focal_scattering_power']:
                            plot_scatter_plane_power_spectrum(plot_data[plot['name']+'_spectrum'], job_idx)
                            
                        elif plot['name'] in ['exit_power_distribution']:
                            plot_exit_power_distribution_spectrum(plot_data[plot['name']+'_spectrum'], job_idx)
                            
                        elif plot['name'] in ['device_rta']:
                            plot_device_rta_spectrum(plot_data[plot['name']+'_spectrum'], job_idx)
                                
                        
            #
            # Step 4.2: Go through all the sweep plots and export each one
            #
            
            for plot in plots:
                if plot['enabled']:
                        
                    if plot['name'] in ['sorting_efficiency']:
                        # Here there is the most variability in what the user might wish to plot, so it will have to be hardcoded for the most part.
                        # plot_function_sweep(plots_data['sweep_plots'][plot['name']], 'th', plot_sorting_eff_sweep_1d.__name__)
                        # plot_sorting_eff_sweep(plots_data['sweep_plots'][plot['name']], 'phi')
                        plot_sorting_eff_sweep_1d(plots_data['sweep_plots'][plot['name']], [slice(None), 0])
                        
                    elif plot['name'] in ['sorting_transmission']:
                        plot_sorting_transmission_sweep_1d(plots_data['sweep_plots'][plot['name']], [slice(None), 0])
                        
                    elif plot['name'] in ['sorting_transmission_meanband']:
                        plot_sorting_meanband_transmission_sweep_1d(plots_data['sweep_plots'][plot['name']], [slice(None), 0], plot_stdDev=False)
                    
                    elif plot['name'] in ['sorting_transmission_contrastband']:
                        plot_sorting_contrastband_transmission_sweep_1d(plots_data['sweep_plots'][plot['name']], [slice(None), 0], 
                                                                        add_opp_polarizations=True, plot_stdDev=False)
                    
                    elif plot['name'] in ['src_spill_power']:
                        plot_src_spill_power_sweep_1d(plots_data['sweep_plots'][plot['name']], [slice(None), 0], plot_stdDev=True)
                        
                    elif plot['name'] in ['device_transmission']:
                        plot_device_transmission_sweep_1d(plots_data['sweep_plots'][plot['name']], [slice(None), 0], plot_stdDev=False)
                        
                    elif plot['name'] in ['device_reflection']:
                        plot_overall_reflection_sweep_1d(plots_data['sweep_plots'][plot['name']], [slice(None), 0], plot_stdDev=False)
                    
                    elif plot['name'] in ['side_scattering_power']:
                        plot_side_power_sweep_1d(plots_data['sweep_plots'][plot['name']], [slice(None), 0], plot_stdDev=False)
                    
                    elif plot['name'] in ['focal_region_power']:
                        plot_focal_power_sweep_1d(plots_data['sweep_plots'][plot['name']], [slice(None), 0], plot_stdDev=False)
                    
                    elif plot['name'] in ['oblique_scattering_power']:
                        plot_oblique_scattering_sweep_1d(plots_data['sweep_plots'][plot['name']], [slice(None), 0], plot_stdDev=False)
                        
                    elif plot['name'] in ['focal_scattering_power']:
                        plot_scatter_plane_power_sweep_1d(plots_data['sweep_plots'][plot['name']], [slice(None), 0], plot_stdDev=False)
                        
                    elif plot['name'] in ['exit_power_distribution']:
                        plot_exit_power_distribution_sweep_1d(plots_data['sweep_plots'][plot['name']], [slice(None), 0], plot_stdDev=False)
                            
                    elif plot['name'] in ['device_rta']:
                        plot_device_rta_sweep_1d(plots_data['sweep_plots'][plot['name']], [slice(None), 0], plot_stdDev = False)


            backup_var('plots_data', os.path.basename(python_src_directory))
            backup_all_vars()
            
                        
            print("Completed Step 4: Created and exported all plots and plot data.")
            sys.stdout.flush()
            
            endtime = time.time()
            print(f'Step 4 took {(endtime-startTime)/60} minutes.')
               
print('Reached end of file. Code completed.')

