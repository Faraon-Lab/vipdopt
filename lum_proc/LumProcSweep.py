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
    monitorbox_size_lateral_um = device_size_lateral_um + 2 * sidewall_thickness_um
    monitorbox_vertical_maximum_um = device_vertical_maximum_um + 0.2
    monitorbox_vertical_minimum_um = device_vertical_minimum_um - 0.2
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
    src_transmission_monitor['x span'] = (fdtd_region_size_lateral_um * 8/9) * 1e-6
    src_transmission_monitor['y'] = 0 * 1e-6
    src_transmission_monitor['y span'] = (fdtd_region_size_lateral_um * 8/9) * 1e-6
    src_transmission_monitor['z'] = (fdtd_hook.getnamed('forward_src_x','z') / 1e-6 - 0.5) * 1e-6
    src_transmission_monitor['override global monitor settings'] = 1
    src_transmission_monitor['use wavelength spacing'] = 1
    src_transmission_monitor['use source limits'] = 1
    src_transmission_monitor['frequency points'] = num_design_frequency_points
    fdtd_objects['src_transmission_monitor'] = src_transmission_monitor
    
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
            fdtd_hook.setnamed(sidewall_mesh['name'], 'dx', fdtd_hook.getnamed('device_mesh','dx') / 8)
        else:
            fdtd_hook.setnamed(sidewall_mesh['name'], 'dy', fdtd_hook.getnamed('device_mesh','dy') / 8)
        
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
        for key, val in globals()['forward_src_x'].items():
            fdtd_hook.setnamed('forward_src_x', key, val)
    

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


#! Step 2 Functions
def get_sourcepower():
    f = fdtd_hook.getresult('transmission_focal_monitor_','f')
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

def generate_sorting_eff_spectrum(monitors_data, produce_spectrum, statistics='peak'):
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
    for idx in range(0, num_adjoint_sources):
        if statistics in ['peak']:
            sweep_index, sweep_val = max(enumerate(f_vectors[idx]['var_values']), key=(lambda x: x[1]))
            sweep_stdev = 0
        else:
            sweep_val = np.average(f_vectors[idx]['var_values'])
            sweep_index = min(range(len(f_vectors[idx]['var_values'])), key=lambda i: abs(f_vectors[idx]['var_values'][i]-sweep_val))
            sweep_stdev = np.std(f_vectors[idx]['var_values'])
            
        sweep_values.append({'value': float(sweep_val),
                        'index': sweep_index,
                        'std_dev': sweep_stdev,
                        'statistics': statistics
                        })
    
    print('Picked peak efficiency points for each quadrant to pass to sorting efficiency variable sweep.')
    
    return output_plot, sweep_values
        
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
    for idx in range(0, 1):
        if statistics in ['peak']:
            sweep_index, sweep_val = max(enumerate(f_vectors[idx]['var_values']), key=(lambda x: x[1]))
            sweep_stdev = 0
        else:
            sweep_val = np.average(f_vectors[idx]['var_values'])
            sweep_index = min(range(len(f_vectors[idx]['var_values'])), key=lambda i: abs(f_vectors[idx]['var_values'][i]-sweep_val))
            sweep_stdev = np.std(f_vectors[idx]['var_values'])
            
        sweep_values.append({'value': float(sweep_val),
                        'index': sweep_index,
                        'std_dev': sweep_stdev,
                        'statistics': statistics
                        })
    
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
    
    print('Generated E-norm focal plane image plot.')
    
    # ## Plotting code to test
    # fig, ax = plt.subplots()
    # #plt.contour([np.squeeze(r_vectors[0]['var_values']), np.squeeze(r_vectors[1]['var_values'])], f_vectors[0]['var_values'][:,:,5])
    # plt.imshow(f_vectors[0]['var_values'][:,:,5])
    # plt.show(block=False)
    # plt.show()
    
    return output_plot, {}

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
    for idx in range(0, 1):
        if statistics in ['peak']:
            sweep_index, sweep_val = max(enumerate(f_vectors[idx]['var_values']), key=(lambda x: x[1]))
            sweep_stdev = 0
        else:
            sweep_val = np.average(f_vectors[idx]['var_values'])
            sweep_index = min(range(len(f_vectors[idx]['var_values'])), key=lambda i: abs(f_vectors[idx]['var_values'][i]-sweep_val))
            sweep_stdev = np.std(f_vectors[idx]['var_values'])
            
        sweep_values.append({'value': float(sweep_val),
                        'index': sweep_index,
                        'std_dev': sweep_stdev,
                        'statistics': statistics
                        })
    
    print('Picked reflection point to pass to variable sweep.')
    
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
    for idx in range(0, 4):
        if statistics in ['peak']:
            sweep_index, sweep_val = max(enumerate(f_vectors[idx]['var_values']), key=(lambda x: x[1]))
            sweep_stdev = 0
        else:
            sweep_val = np.average(f_vectors[idx]['var_values'])
            sweep_index = min(range(len(f_vectors[idx]['var_values'])), key=lambda i: abs(f_vectors[idx]['var_values'][i]-sweep_val))
            sweep_stdev = np.std(f_vectors[idx]['var_values'])
            
        sweep_values.append({'value': float(sweep_val),
                        'index': sweep_index,
                        'std_dev': sweep_stdev,
                        'statistics': statistics
                        })
    
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
    for idx in range(0, 1):
        if statistics in ['peak']:
            sweep_index, sweep_val = max(enumerate(f_vectors[idx]['var_values']), key=(lambda x: x[1]))
            sweep_stdev = 0
        else:
            sweep_val = np.average(f_vectors[idx]['var_values'])
            sweep_index = min(range(len(f_vectors[idx]['var_values'])), key=lambda i: abs(f_vectors[idx]['var_values'][i]-sweep_val))
            sweep_stdev = np.std(f_vectors[idx]['var_values'])
            
        sweep_values.append({'value': float(sweep_val),
                        'index': sweep_index,
                        'std_dev': sweep_stdev,
                        'statistics': statistics
                        })
    
    print('Picked focal region power point to pass to variable sweep.')
    
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
    for idx in range(0, 1):
        if statistics in ['peak']:
            sweep_index, sweep_val = max(enumerate(f_vectors[idx]['var_values']), key=(lambda x: x[1]))
            sweep_stdev = 0
        else:
            sweep_val = np.average(f_vectors[idx]['var_values'])
            sweep_index = min(range(len(f_vectors[idx]['var_values'])), key=lambda i: abs(f_vectors[idx]['var_values'][i]-sweep_val))
            sweep_stdev = np.std(f_vectors[idx]['var_values'])
            
        sweep_values.append({'value': float(sweep_val),
                        'index': sweep_index,
                        'std_dev': sweep_stdev,
                        'statistics': statistics
                        })
    
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
    
    try:
        scatter_power = monitors_data['scatter_plane_monitor']['P']
    except Exception as err:
        scatter_power = monitors_data['spill_plane_monitor']['P']
    focal_power = monitors_data['transmission_focal_monitor_']['P']
    exit_aperture_power = monitors_data['exit_aperture_monitor']['P']
    
    f_vectors.append({'var_name': 'Focal Region Power',
                      'var_values': focal_power/scatter_power
                    })
    
    f_vectors.append({'var_name': 'Oblique Scattering Power',
                      'var_values': (scatter_power - focal_power)/scatter_power
                    })
    
    if produce_spectrum:
        output_plot['r'] = r_vectors
        output_plot['f'] = f_vectors
        output_plot['title'] = 'Focal Plane Power Distribution Spectrum'
        output_plot['const_params'] = const_parameters
        
        print('Generated focal plane power distribution spectrum.')
    
    sweep_values = []
    for idx in range(0, len(f_vectors)):
        if statistics in ['peak']:
            sweep_index, sweep_val = max(enumerate(f_vectors[idx]['var_values']), key=(lambda x: x[1]))
            sweep_stdev = 0
        else:
            sweep_val = np.average(f_vectors[idx]['var_values'])
            sweep_index = min(range(len(f_vectors[idx]['var_values'])), key=lambda i: abs(f_vectors[idx]['var_values'][i]-sweep_val))
            sweep_stdev = np.std(f_vectors[idx]['var_values'])
            
        sweep_values.append({'value': float(sweep_val),
                        'index': sweep_index,
                        'std_dev': sweep_stdev,
                        'statistics': statistics
                        })
    
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
    
    input_power = monitors_data['incident_aperture_monitor']['P'] # monitors_data['sourcepower']
    
    R_power = np.reshape(1 - monitors_data['src_transmission_monitor']['T'],    (len(lambda_vector),1))
    R_power = R_power * monitors_data['sourcepower']
    
    side_powers = []
    side_directions = ['E','N','W','S']
    for idx in range(0, 4):
        side_powers.append(monitors_data['side_monitor_'+str(idx)]['P'])
    
    try:
        scatter_power = monitors_data['scatter_plane_monitor']['P']
    except Exception as err:
        scatter_power = monitors_data['spill_plane_monitor']['P']
        
    focal_power = monitors_data['transmission_focal_monitor_']['P']
    exit_aperture_power = monitors_data['exit_aperture_monitor']['P']
    
    power_sum = R_power + scatter_power
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
                      'var_values': (scatter_power - focal_power)/input_power   # (exit_aperture_power - focal_power)/input_power
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
    for idx in range(0, len(f_vectors)):
        if statistics in ['peak']:
            sweep_index, sweep_val = max(enumerate(f_vectors[idx]['var_values']), key=(lambda x: x[1]))
            sweep_stdev = 0
        else:
            sweep_val = np.average(f_vectors[idx]['var_values'])
            sweep_index = min(range(len(f_vectors[idx]['var_values'])), key=lambda i: abs(f_vectors[idx]['var_values'][i]-sweep_val))
            sweep_stdev = np.std(f_vectors[idx]['var_values'])
            
        sweep_values.append({'value': float(sweep_val),
                        'index': sweep_index,
                        'std_dev': sweep_stdev,
                        'statistics': statistics
                        })
    
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

# -- Spectrum Plot Functions (Per Job)

def plot_sorting_eff_spectrum(plot_data, job_idx):
    '''For the given job index, plots the sorting efficiency spectrum corresponding to that job.'''
    r_vectors = plot_data['r']
    f_vectors = plot_data['f']
    
    fig, ax = plt.subplots()
    plot_config = {'title': None,
                   'x_axis': {'label':'', 'limits':[]},
                   'y_axis': {'label':'', 'limits':[]},
                   'lines': []
                }
    
    plot_colors = ['blue', 'green', 'red', 'gray']
    plot_labels = ['Blue', 'Green (x-pol.)', 'Red', 'Green (y-pol.)']
    
    for plot_idx in range(0, len(f_vectors)):
        line_data = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'y_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'cutoff': None,
                     'color': None,
                     'alpha': 1.0,
                     'legend': None,
                     'marker_style': marker_style
                }
        
        line_data['x_axis']['values'] = r_vectors[0]['var_values']
        line_data['x_axis']['factor'] = 1e6
        line_data['y_axis']['values'] = f_vectors[plot_idx]['var_values']
        line_data['cutoff'] = slice(0, len(line_data['x_axis']['values'])-5)
        
        line_data['color'] = plot_colors[plot_idx]
        line_data['legend'] = plot_labels[plot_idx]
        
        plot_config['lines'].append(line_data)
        
    
    title_string = 'Spectrum in Each Quadrant'
    for sweep_param_value in list(sweep_parameters.values()):
        current_value = sweep_param_value['var_values'][job_idx[2][0]]
        optimized_value = sweep_param_value['var_values'][sweep_param_value['peakInd']]
        title_string += '\n' + sweep_param_value['var_name'] + f': Current Value {current_value}, Optimized for {optimized_value}'
    plot_config['title'] = title_string
    
    plot_config['x_axis']['label'] = 'Wavelength (um)'
    plot_config['y_axis']['label'] = 'Sorting Efficiency'
    
    
    
    fig, ax = enter_plot_data_1d(plot_config, fig, ax)
    
    
    plot_subfolder_name = 'sorting_efficiency_spectra'
    if not os.path.isdir(plot_directory_location + '/' + plot_subfolder_name):
        os.makedirs(plot_directory_location + '/' + plot_subfolder_name)
    plt.savefig(plot_directory_location + '/' + plot_subfolder_name\
        + '/' + isolate_filename(job_names[job_idx]).replace('.fsp', '') + '.png',
        bbox_inches='tight')
    print('Exported: Sorting Efficiency Spectrum')
    
    plt.close()
    
    return fig, ax

def plot_Enorm_focal_plane_image_spectrum(plot_data, job_idx, plot_wavelengths = None, ignore_opp_polarization = True):
    '''For the given job index, plots the E-norm f.p. image at specific input spectra, corresponding to that job.
    Note: Includes focal scatter region. '''
    
    r_vectors = plot_data['r']
    f_vectors = plot_data['f']
    lambda_vectors = plot_data['lambda']
    wl_vector = np.squeeze(lambda_vectors[0]['var_values']).tolist()
    
    if plot_wavelengths is None:
        # plot_wavelengths = [lambda_min_um, lambda_values_um[int(len(lambda_values_um/2))]lambda_max_um]
        spectral_band_midpoints = np.multiply(num_points_per_band, np.arange(0.5, num_bands + 0.5))         # This fails if num_points_per_band and num_bands are different than what was optimized
        plot_wavelengths = lambda_vectors[[int(g) for g in spectral_band_midpoints]]
        
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

def plot_device_transmission_spectrum(plot_data, job_idx):
    '''For the given job index, plots the device transmission spectrum corresponding to that job.'''
    r_vectors = plot_data['r']
    f_vectors = plot_data['f']
    
    fig, ax = plt.subplots()
    plot_config = {'title': None,
                   'x_axis': {'label':'', 'limits':[]},
                   'y_axis': {'label':'', 'limits':[]},
                   'lines': []
                }
    
    for plot_idx in range(0, len(f_vectors)):
        line_data = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'y_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'cutoff': None,
                     'color': None,
                     'alpha': 1.0,
                     'legend': None,
                     'marker_style': marker_style
                }
        
        line_data['x_axis']['values'] = r_vectors[0]['var_values']
        line_data['x_axis']['factor'] = 1e6
        line_data['y_axis']['values'] = f_vectors[plot_idx]['var_values']
        line_data['cutoff'] = slice(0, len(line_data['x_axis']['values'])-5)
        
        line_data['legend'] = f_vectors[plot_idx]['var_name']
        
        plot_config['lines'].append(line_data)
        
    
    title_string = 'Device Transmission'
    for sweep_param_value in list(sweep_parameters.values()):
        current_value = sweep_param_value['var_values'][job_idx[2][0]]
        optimized_value = sweep_param_value['var_values'][sweep_param_value['peakInd']]
        title_string += '\n' + sweep_param_value['var_name'] + f': Current Value {current_value}, Optimized for {optimized_value}'
    plot_config['title'] = title_string
    
    plot_config['x_axis']['label'] = 'Wavelength (um)'
    plot_config['y_axis']['label'] = 'Device Transmission'
    
    
    
    fig, ax = enter_plot_data_1d(plot_config, fig, ax)

    
    plot_subfolder_name = 'device_transmission_spectra'
    if not os.path.isdir(plot_directory_location + '/' + plot_subfolder_name):
        os.makedirs(plot_directory_location + '/' + plot_subfolder_name)
    plt.savefig(plot_directory_location + '/' + plot_subfolder_name\
        + '/' + isolate_filename(job_names[job_idx]).replace('.fsp', '') + '.png',
        bbox_inches='tight')
    print('Exported: Device Transmission Spectrum')
    
    plt.close()
    
    return fig, ax

def plot_overall_reflection_spectrum(plot_data, job_idx):
    '''For the given job index, plots the overall reflection spectrum corresponding to that job.'''
    r_vectors = plot_data['r']
    f_vectors = plot_data['f']
    
    fig, ax = plt.subplots()
    plot_config = {'title': None,
                   'x_axis': {'label':'', 'limits':[]},
                   'y_axis': {'label':'', 'limits':[]},
                   'lines': []
                }
    
    for plot_idx in range(0, len(f_vectors)):
        line_data = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'y_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'cutoff': None,
                     'color': None,
                     'alpha': 1.0,
                     'legend': None,
                     'marker_style': marker_style
                }
        
        line_data['x_axis']['values'] = r_vectors[0]['var_values']
        line_data['x_axis']['factor'] = 1e6
        line_data['y_axis']['values'] = f_vectors[plot_idx]['var_values']
        line_data['cutoff'] = slice(0, len(line_data['x_axis']['values'])-5)
        
        line_data['legend'] = f_vectors[plot_idx]['var_name']
        
        plot_config['lines'].append(line_data)
        
    
    title_string = 'Overall Device Reflection'
    for sweep_param_value in list(sweep_parameters.values()):
        current_value = sweep_param_value['var_values'][job_idx[2][0]]
        optimized_value = sweep_param_value['var_values'][sweep_param_value['peakInd']]
        title_string += '\n' + sweep_param_value['var_name'] + f': Current Value {current_value}, Optimized for {optimized_value}'
    plot_config['title'] = title_string
    
    plot_config['x_axis']['label'] = 'Wavelength (um)'
    plot_config['y_axis']['label'] = 'Overall Reflection'
    
    
    
    fig, ax = enter_plot_data_1d(plot_config, fig, ax)

    
    plot_subfolder_name = 'overall_reflection_spectra'
    if not os.path.isdir(plot_directory_location + '/' + plot_subfolder_name):
        os.makedirs(plot_directory_location + '/' + plot_subfolder_name)
    plt.savefig(plot_directory_location + '/' + plot_subfolder_name\
        + '/' + isolate_filename(job_names[job_idx]).replace('.fsp', '') + '.png',
        bbox_inches='tight')
    print('Exported: Overall Reflection Spectrum')
    
    plt.close()
    
    return fig, ax

def plot_side_power_spectrum(plot_data, job_idx):
    '''For the given job index, plots the device side scattering corresponding to that job.'''
    r_vectors = plot_data['r']
    f_vectors = plot_data['f']
    
    fig, ax = plt.subplots()
    plot_config = {'title': None,
                   'x_axis': {'label':'', 'limits':[]},
                   'y_axis': {'label':'', 'limits':[]},
                   'lines': []
                }
    
    plot_labels = ['Device Transmission']
    
    for plot_idx in range(0, len(f_vectors)):
        line_data = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'y_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'cutoff': None,
                     'color': None,
                     'alpha': 1.0,
                     'legend': None,
                     'marker_style': marker_style
                }
        
        line_data['x_axis']['values'] = r_vectors[0]['var_values']
        line_data['x_axis']['factor'] = 1e6
        line_data['y_axis']['values'] = f_vectors[plot_idx]['var_values']
        line_data['cutoff'] = slice(0, len(line_data['x_axis']['values'])-5)
        
        line_data['legend'] = f_vectors[plot_idx]['var_name']
        
        plot_config['lines'].append(line_data)
        
    
    title_string = 'Device Side Scattering'
    for sweep_param_value in list(sweep_parameters.values()):
        current_value = sweep_param_value['var_values'][job_idx[2][0]]
        optimized_value = sweep_param_value['var_values'][sweep_param_value['peakInd']]
        title_string += '\n' + sweep_param_value['var_name'] + f': Current Value {current_value}, Optimized for {optimized_value}'
    plot_config['title'] = title_string
    
    plot_config['x_axis']['label'] = 'Wavelength (um)'
    plot_config['y_axis']['label'] = 'Side Scattering Power'
    
    
    fig, ax = enter_plot_data_1d(plot_config, fig, ax)

    
    plot_subfolder_name = 'side_power_spectra'
    if not os.path.isdir(plot_directory_location + '/' + plot_subfolder_name):
        os.makedirs(plot_directory_location + '/' + plot_subfolder_name)
    plt.savefig(plot_directory_location + '/' + plot_subfolder_name\
        + '/' + isolate_filename(job_names[job_idx]).replace('.fsp', '') + '.png',
        bbox_inches='tight')
    print('Exported: Side Scattering Power Spectrum')
    
    plt.close()
    
    return fig, ax

def plot_focal_power_spectrum(plot_data, job_idx):
    '''For the given job index, plots the power falling on the desired focal region corresponding to that job.'''
    r_vectors = plot_data['r']
    f_vectors = plot_data['f']
    
    fig, ax = plt.subplots()
    plot_config = {'title': None,
                   'x_axis': {'label':'', 'limits':[]},
                   'y_axis': {'label':'', 'limits':[]},
                   'lines': []
                }
    
    plot_labels = ['Device Transmission']
    
    for plot_idx in range(0, len(f_vectors)):
        line_data = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'y_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'cutoff': None,
                     'color': None,
                     'alpha': 1.0,
                     'legend': None,
                     'marker_style': marker_style
                }
        
        line_data['x_axis']['values'] = r_vectors[0]['var_values']
        line_data['x_axis']['factor'] = 1e6
        line_data['y_axis']['values'] = f_vectors[plot_idx]['var_values']
        line_data['cutoff'] = slice(0, len(line_data['x_axis']['values'])-5)
        
        line_data['legend'] = f_vectors[plot_idx]['var_name']
        
        plot_config['lines'].append(line_data)
        
    
    title_string = 'Focal Region Power'
    for sweep_param_value in list(sweep_parameters.values()):
        current_value = sweep_param_value['var_values'][job_idx[2][0]]
        optimized_value = sweep_param_value['var_values'][sweep_param_value['peakInd']]
        title_string += '\n' + sweep_param_value['var_name'] + f': Current Value {current_value}, Optimized for {optimized_value}'
    plot_config['title'] = title_string
    
    plot_config['x_axis']['label'] = 'Wavelength (um)'
    plot_config['y_axis']['label'] = 'Focal Region Power'
    
    
    fig, ax = enter_plot_data_1d(plot_config, fig, ax)
    
    
    plot_subfolder_name = 'focal_region_power_spectra'
    if not os.path.isdir(plot_directory_location + '/' + plot_subfolder_name):
        os.makedirs(plot_directory_location + '/' + plot_subfolder_name)
    plt.savefig(plot_directory_location + '/' + plot_subfolder_name\
        + '/' + isolate_filename(job_names[job_idx]).replace('.fsp', '') + '.png',
        bbox_inches='tight')
    print('Exported: Focal Region Power Spectrum')
    
    plt.close()
    
    return fig, ax

def plot_scatter_plane_power_spectrum(plot_data, job_idx):
    '''For the given job index, plots the device transmission spectrum corresponding to that job.'''
    r_vectors = plot_data['r']
    f_vectors = plot_data['f']
    
    fig, ax = plt.subplots()
    plot_config = {'title': None,
                   'x_axis': {'label':'', 'limits':[]},
                   'y_axis': {'label':'', 'limits':[]},
                   'lines': []
                }
    
    plot_labels = ['Device Transmission']
    
    for plot_idx in range(0, len(f_vectors)):
        line_data = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'y_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'cutoff': None,
                     'color': None,
                     'alpha': 1.0,
                     'legend': None,
                     'marker_style': marker_style
                }
        
        line_data['x_axis']['values'] = r_vectors[0]['var_values']
        line_data['x_axis']['factor'] = 1e6
        line_data['y_axis']['values'] = f_vectors[plot_idx]['var_values']
        line_data['cutoff'] = slice(0, len(line_data['x_axis']['values'])-5)
        
        line_data['legend'] = f_vectors[plot_idx]['var_name']
        
        plot_config['lines'].append(line_data)
        
    
    title_string = 'Focal Scatter Plane Power'
    for sweep_param_value in list(sweep_parameters.values()):
        current_value = sweep_param_value['var_values'][job_idx[2][0]]
        optimized_value = sweep_param_value['var_values'][sweep_param_value['peakInd']]
        title_string += '\n' + sweep_param_value['var_name'] + f': Current Value {current_value}, Optimized for {optimized_value}'
    plot_config['title'] = title_string
    
    plot_config['x_axis']['label'] = 'Wavelength (um)'
    plot_config['y_axis']['label'] = 'Focal Scatter Plane Power'
    
    
    fig, ax = enter_plot_data_1d(plot_config, fig, ax)
    
    
    plot_subfolder_name = 'scatter_plane_power'
    if not os.path.isdir(plot_directory_location + '/' + plot_subfolder_name):
        os.makedirs(plot_directory_location + '/' + plot_subfolder_name)
    plt.savefig(plot_directory_location + '/' + plot_subfolder_name\
        + '/' + isolate_filename(job_names[job_idx]).replace('.fsp', '') + '.png',
        bbox_inches='tight')
    print('Exported: Scatter Plane Power Spectrum')
    
    plt.close()
    
    return fig, ax

def plot_exit_power_distribution_spectrum(plot_data, job_idx):
    '''For the given job index, plots the power spectrum for focal region power and oblique scattering power, corresponding to that job.
    The two are normalized against their sum.'''
    
    r_vectors = plot_data['r']
    f_vectors = plot_data['f']
    
    fig, ax = plt.subplots()
    plot_config = {'title': None,
                   'x_axis': {'label':'', 'limits':[]},
                   'y_axis': {'label':'', 'limits':[]},
                   'lines': []
                }
    
    plot_labels = ['Device Transmission']
    
    for plot_idx in range(0, len(f_vectors)):
        line_data = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'y_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'cutoff': None,
                     'color': None,
                     'alpha': 1.0,
                     'legend': None,
                     'marker_style': marker_style
                }
        
        line_data['x_axis']['values'] = r_vectors[0]['var_values']
        line_data['x_axis']['factor'] = 1e6
        line_data['y_axis']['values'] = f_vectors[plot_idx]['var_values']
        line_data['cutoff'] = slice(0, len(line_data['x_axis']['values'])-5)
        
        line_data['legend'] = f_vectors[plot_idx]['var_name']
        
        plot_config['lines'].append(line_data)
        
    
    title_string = 'Power Distribution at Focal Plane'
    for sweep_param_value in list(sweep_parameters.values()):
        current_value = sweep_param_value['var_values'][job_idx[2][0]]
        optimized_value = sweep_param_value['var_values'][sweep_param_value['peakInd']]
        title_string += '\n' + sweep_param_value['var_name'] + f': Current Value {current_value}, Optimized for {optimized_value}'
    plot_config['title'] = title_string
    
    plot_config['x_axis']['label'] = 'Wavelength (um)'
    plot_config['y_axis']['label'] = 'Normalized Power'
    
    
    fig, ax = enter_plot_data_1d(plot_config, fig, ax)
    
    
    plot_subfolder_name = 'exit_power_distribution_spectra'
    if not os.path.isdir(plot_directory_location + '/' + plot_subfolder_name):
        os.makedirs(plot_directory_location + '/' + plot_subfolder_name)
    plt.savefig(plot_directory_location + '/' + plot_subfolder_name\
        + '/' + isolate_filename(job_names[job_idx]).replace('.fsp', '') + '.png',
        bbox_inches='tight')
    print('Exported: Exit Power Distribution Spectrum')
    
    plt.close()
    
    return fig, ax

def plot_device_rta_spectrum(plot_data, job_idx):
    '''For the given job index, plots the device RTA and all power components in and out, corresponding to that job.
    The normalization factor is the power through the device input aperture.'''
    
    r_vectors = plot_data['r']
    f_vectors = plot_data['f']
    
    fig, ax = plt.subplots()
    plot_config = {'title': None,
                   'x_axis': {'label':'', 'limits':[]},
                   'y_axis': {'label':'', 'limits':[]},
                   'lines': []
                }
    
    plot_labels = ['Device Transmission']
    
    for plot_idx in range(0, len(f_vectors)):
        line_data = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'y_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'cutoff': None,
                     'color': None,
                     'alpha': 1.0,
                     'legend': None,
                     'marker_style': marker_style
                }
        
        line_data['x_axis']['values'] = r_vectors[0]['var_values']
        line_data['x_axis']['factor'] = 1e6
        line_data['y_axis']['values'] = f_vectors[plot_idx]['var_values']
        line_data['cutoff'] = slice(0, len(line_data['x_axis']['values'])-5)
        
        line_data['legend'] = f_vectors[plot_idx]['var_name']
        
        plot_config['lines'].append(line_data)
        
    
    title_string = 'Device RTA'
    for sweep_param_value in list(sweep_parameters.values()):
        current_value = sweep_param_value['var_values'][job_idx[2][0]]
        optimized_value = sweep_param_value['var_values'][sweep_param_value['peakInd']]
        title_string += '\n' + sweep_param_value['var_name'] + f': Current Value {current_value}, Optimized for {optimized_value}'
    plot_config['title'] = title_string
    
    plot_config['x_axis']['label'] = 'Wavelength (um)'
    plot_config['y_axis']['label'] = 'Normalized Power'
    
    
    fig, ax = enter_plot_data_1d(plot_config, fig, ax)
    
    
    plot_subfolder_name = 'device_rta_spectra'
    if not os.path.isdir(plot_directory_location + '/' + plot_subfolder_name):
        os.makedirs(plot_directory_location + '/' + plot_subfolder_name)
    plt.savefig(plot_directory_location + '/' + plot_subfolder_name\
        + '/' + isolate_filename(job_names[job_idx]).replace('.fsp', '') + '.png',
        bbox_inches='tight')
    print('Exported: Device RTA Spectrum')
    
    plt.close()
    
    return fig, ax

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
    
    # Whichever entry is a slice means that variable in r_vectors is the x-axis of this plot
    r_vector_value_idx = [index for (index , item) in enumerate(slice_coords) if isinstance(item, slice)][0]
    r_vectors = list(plot_data['r'].values())[r_vector_value_idx]
    f_vectors = plot_data['f']              # This is a list of N-D arrays, containing the function values for each parameter combination
    
    fig, ax = plt.subplots()
    plot_config = {'title': None,
                   'x_axis': {'label':'', 'limits':[]},
                   'y_axis': {'label':'', 'limits':[]},
                   'lines': []
                }
    
    plot_colors = ['blue', 'green', 'red', 'gray']
    plot_labels = ['Blue', 'Green (x-pol.)', 'Red', 'Green (y-pol.)']
    normalize_against_max = False
    
    for plot_idx in range(0, len(f_vectors)-0):
        line_data = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'y_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'cutoff': None,
                     'color': None,
                     'alpha': 1.0,
                     'legend': None,
                     'marker_style': marker_style
                }
        
        
        y_plot_data = f_vectors[plot_idx]['var_values']
        y_plot_data = np.reshape(y_plot_data, (len(y_plot_data), 1))
        y_plot_data = y_plot_data[tuple(slice_coords)]
        normalization_factor = np.max(y_plot_data) if normalize_against_max else 1
        line_data['y_axis']['values'] = y_plot_data / normalization_factor
        line_data['x_axis']['values'] = r_vectors['var_values']
        line_data['cutoff'] = slice(0, len(line_data['x_axis']['values'])-5)
        
        line_data['color'] = plot_colors[plot_idx]
        line_data['legend'] = plot_labels[plot_idx]
        
        plot_config['lines'].append(line_data)
        
        
        if plot_stdDev and f_vectors[plot_idx]['statistics'] in ['mean']:
            line_data_2 = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'y_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'cutoff': None,
                     'color': None,
                     'alpha': 0.3,
                     'legend': '_nolegend_',
                     'marker_style': dict(linestyle='-', linewidth=1.0)
                }
        
            line_data_2['x_axis']['values'] = r_vectors['var_values']
            line_data_2['y_axis']['values'] = (y_plot_data + f_vectors[plot_idx]['var_stdevs']) / normalization_factor
            line_data_2['cutoff'] = slice(0, len(line_data_2['x_axis']['values']))
            
            colors_so_far = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            line_data_2['color'] = colors_so_far[-1]
            
            plot_config['lines'].append(line_data_2)
            
            line_data_3 = copy.deepcopy(line_data_2)
            line_data_3['y_axis']['values'] = (y_plot_data - f_vectors[plot_idx]['var_stdevs']) / normalization_factor
            plot_config['lines'].append(line_data_3)
            
    title_string = 'Sorting Efficiency - Parameter Range:'
    for sweep_param_value in list(sweep_parameters.values()):
        optimized_value = sweep_param_value['var_values'][sweep_param_value['peakInd']]
        title_string += '\n' + sweep_param_value['var_name'] + f': Optimized at {optimized_value}'
    plot_config['title'] = title_string
    
    sweep_variable_idx = [index for (index, item) in enumerate(slice_coords) if type(item) is slice][0]
    plot_config['x_axis']['label'] = list(sweep_parameters.values())[sweep_variable_idx]['var_name']
    plot_config['y_axis']['label'] = 'Sorting Efficiency'
    
    
    
    fig, ax = enter_plot_data_1d(plot_config, fig, ax)
    
    plot_subfolder_name = ''
    if not os.path.isdir(plot_directory_location + '/' + plot_subfolder_name):
        os.makedirs(plot_directory_location + '/' + plot_subfolder_name)
    plt.savefig(plot_directory_location + '/' + plot_subfolder_name\
        + '/' + 'sorting_eff_sweep_' + create_parameter_filename_string(slice_coords) + '.png',
        bbox_inches='tight')
    print('Exported: Sorting Efficiency Sweep: ' + create_parameter_filename_string(slice_coords))
    
    return fig, ax

def plot_device_transmission_sweep_1d(plot_data, slice_coords, plot_stdDev = True):
    '''Produces a 1D device transmission plot using given coordinates to slice the N-D array of data contained in plot_data.
    X-axis i.e. sweep variable is given by the index of the entry of type 'slice' in the slice_coords.'''
    
    # Whichever entry is a slice means that variable in r_vectors is the x-axis of this plot
    r_vector_value_idx = [index for (index , item) in enumerate(slice_coords) if isinstance(item, slice)][0]
    r_vectors = list(plot_data['r'].values())[r_vector_value_idx]
    f_vectors = plot_data['f']              # This is a list of N-D arrays, containing the function values for each parameter combination
    
    fig, ax = plt.subplots()
    plot_config = {'title': None,
                   'x_axis': {'label':'', 'limits':[]},
                   'y_axis': {'label':'', 'limits':[]},
                   'lines': []
                }
    
    plot_labels = ['Device Transmission']
    normalize_against_max = False
    
    plot_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    
    for plot_idx in range(0, len(f_vectors)):
        line_data = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
                        'y_axis': {'values': None, 'factor': 1, 'offset': 0},
                        'cutoff': None,
                        'color': None,
                        'alpha': 1.0,
                        'legend': None,
                        'marker_style': marker_style
                }
        
        
        y_plot_data = f_vectors[plot_idx]['var_values']
        y_plot_data = np.reshape(y_plot_data, (len(y_plot_data), 1))
        y_plot_data = y_plot_data[tuple(slice_coords)]
        
        normalization_factor = np.max(y_plot_data) if normalize_against_max else 1
        line_data['y_axis']['values'] = y_plot_data / normalization_factor
        line_data['x_axis']['values'] = r_vectors['var_values']
        line_data['cutoff'] = slice(0, len(line_data['x_axis']['values'])-5)
        
        line_data['legend'] = plot_labels[plot_idx]
        
        plot_config['lines'].append(line_data)
        
        
        if plot_stdDev and f_vectors[plot_idx]['statistics'] in ['mean']:
            line_data_2 = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'y_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'cutoff': None,
                     'color': None,
                     'alpha': 0.1,
                     'legend': '_nolegend_',
                     'marker_style': dict(linestyle='-', linewidth=1.0)
                }
        
            line_data_2['x_axis']['values'] = r_vectors['var_values']
            line_data_2['y_axis']['values'] = (y_plot_data + f_vectors[plot_idx]['var_stdevs']) / normalization_factor
            line_data_2['cutoff'] = slice(0, len(line_data_2['x_axis']['values']))
            
            line_data_2['color'] = plot_colors[plot_idx]
            
            plot_config['lines'].append(line_data_2)
            
            line_data_3 = copy.deepcopy(line_data_2)
            line_data_3['y_axis']['values'] = (y_plot_data - f_vectors[plot_idx]['var_stdevs']) / normalization_factor
            plot_config['lines'].append(line_data_3)
            
    title_string = 'Device Transmission - Parameter Range:'
    for sweep_param_value in list(sweep_parameters.values()):
        optimized_value = sweep_param_value['var_values'][sweep_param_value['peakInd']]
        title_string += '\n' + sweep_param_value['var_name'] + f': Optimized at {optimized_value}'
    plot_config['title'] = title_string
    
    sweep_variable_idx = [index for (index, item) in enumerate(slice_coords) if type(item) is slice][0]
    plot_config['x_axis']['label'] = list(sweep_parameters.values())[sweep_variable_idx]['var_name']
    plot_config['y_axis']['label'] = 'Device Transmission'
    
    
    fig, ax = enter_plot_data_1d(plot_config, fig, ax)
    
    
    plot_subfolder_name = ''
    if not os.path.isdir(plot_directory_location + '/' + plot_subfolder_name):
        os.makedirs(plot_directory_location + '/' + plot_subfolder_name)
    plt.savefig(plot_directory_location + '/' + plot_subfolder_name\
        + '/' + 'dev_transmission_sweep_' + create_parameter_filename_string(slice_coords) + '.png',
        bbox_inches='tight')
    print('Exported: Device Transmission Sweep: ' + create_parameter_filename_string(slice_coords))
    
    return fig, ax

def plot_overall_reflection_sweep_1d(plot_data, slice_coords, plot_stdDev = True):
    '''Produces a 1D FDTD-region reflection plot using given coordinates to slice the N-D array of data contained in plot_data.
    X-axis i.e. sweep variable is given by the index of the entry of type 'slice' in the slice_coords.'''
    
    # Whichever entry is a slice means that variable in r_vectors is the x-axis of this plot
    r_vector_value_idx = [index for (index , item) in enumerate(slice_coords) if isinstance(item, slice)][0]
    r_vectors = list(plot_data['r'].values())[r_vector_value_idx]
    f_vectors = plot_data['f']              # This is a list of N-D arrays, containing the function values for each parameter combination
    
    fig, ax = plt.subplots()
    plot_config = {'title': None,
                   'x_axis': {'label':'', 'limits':[]},
                   'y_axis': {'label':'', 'limits':[]},
                   'lines': []
                }
    
    plot_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plot_labels = ['Overall Reflection']
    normalize_against_max = False
    
    for plot_idx in range(0, len(f_vectors)):
        line_data = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
                        'y_axis': {'values': None, 'factor': 1, 'offset': 0},
                        'cutoff': None,
                        'color': None,
                        'alpha': 1.0,
                        'legend': None,
                        'marker_style': marker_style
                }
        
        
        y_plot_data = f_vectors[plot_idx]['var_values']
        y_plot_data = np.reshape(y_plot_data, (len(y_plot_data), 1))
        y_plot_data = y_plot_data[tuple(slice_coords)]
        
        normalization_factor = np.max(y_plot_data) if normalize_against_max else 1
        line_data['y_axis']['values'] = y_plot_data / normalization_factor
        line_data['x_axis']['values'] = r_vectors['var_values']
        line_data['cutoff'] = slice(0, len(line_data['x_axis']['values'])-5)
        
        line_data['legend'] = plot_labels[plot_idx]
        
        plot_config['lines'].append(line_data)
        
        
        if plot_stdDev and f_vectors[plot_idx]['statistics'] in ['mean']:
            line_data_2 = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'y_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'cutoff': None,
                     'color': None,
                     'alpha': 0.1,
                     'legend': '_nolegend_',
                     'marker_style': dict(linestyle='-', linewidth=1.0)
                }
        
            line_data_2['x_axis']['values'] = r_vectors['var_values']
            line_data_2['y_axis']['values'] = (y_plot_data + f_vectors[plot_idx]['var_stdevs']) / normalization_factor
            line_data_2['cutoff'] = slice(0, len(line_data_2['x_axis']['values']))
            
            line_data_2['color'] = plot_colors[plot_idx]
            
            plot_config['lines'].append(line_data_2)
            
            line_data_3 = copy.deepcopy(line_data_2)
            line_data_3['y_axis']['values'] = (y_plot_data - f_vectors[plot_idx]['var_stdevs']) / normalization_factor
            plot_config['lines'].append(line_data_3)
            
    title_string = 'Overall Device Reflection - Parameter Range:'
    for sweep_param_value in list(sweep_parameters.values()):
        optimized_value = sweep_param_value['var_values'][sweep_param_value['peakInd']]
        title_string += '\n' + sweep_param_value['var_name'] + f': Optimized at {optimized_value}'
    plot_config['title'] = title_string
    
    sweep_variable_idx = [index for (index, item) in enumerate(slice_coords) if type(item) is slice][0]
    plot_config['x_axis']['label'] = list(sweep_parameters.values())[sweep_variable_idx]['var_name']
    plot_config['y_axis']['label'] = 'Overall Reflection'
    
    
    fig, ax = enter_plot_data_1d(plot_config, fig, ax)
    
    
    plot_subfolder_name = ''
    if not os.path.isdir(plot_directory_location + '/' + plot_subfolder_name):
        os.makedirs(plot_directory_location + '/' + plot_subfolder_name)
    plt.savefig(plot_directory_location + '/' + plot_subfolder_name\
        + '/' + 'overall_ref_sweep_' + create_parameter_filename_string(slice_coords) + '.png',
        bbox_inches='tight')
    print('Exported: Overall Reflection Sweep: ' + create_parameter_filename_string(slice_coords))
    
    return fig, ax

def plot_side_power_sweep_1d(plot_data, slice_coords, plot_stdDev = True):
    '''Produces a 1D side scattering power plot using given coordinates to slice the N-D array of data contained in plot_data.
    X-axis i.e. sweep variable is given by the index of the entry of type 'slice' in the slice_coords.'''
    
    # Whichever entry is a slice means that variable in r_vectors is the x-axis of this plot
    r_vector_value_idx = [index for (index , item) in enumerate(slice_coords) if isinstance(item, slice)][0]
    r_vectors = list(plot_data['r'].values())[r_vector_value_idx]
    f_vectors = plot_data['f']              # This is a list of N-D arrays, containing the function values for each parameter combination
    
    fig, ax = plt.subplots()
    plot_config = {'title': None,
                   'x_axis': {'label':'', 'limits':[]},
                   'y_axis': {'label':'', 'limits':[]},
                   'lines': []
                }
    plot_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plot_labels = ['Side Scattering E', 'Side Scattering N', 'Side Scattering W', 'Side Scattering S']
    normalize_against_max = False
    
    for plot_idx in range(0, len(f_vectors)):
        line_data = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
                        'y_axis': {'values': None, 'factor': 1, 'offset': 0},
                        'cutoff': None,
                        'color': None,
                        'alpha': 1.0,
                        'legend': None,
                        'marker_style': marker_style
                }
        
        
        y_plot_data = f_vectors[plot_idx]['var_values']
        y_plot_data = np.reshape(y_plot_data, (len(y_plot_data), 1))
        y_plot_data = y_plot_data[tuple(slice_coords)]
        
        normalization_factor = np.max(y_plot_data) if normalize_against_max else 1
        line_data['y_axis']['values'] = y_plot_data / normalization_factor
        line_data['x_axis']['values'] = r_vectors['var_values']
        line_data['cutoff'] = slice(0, len(line_data['x_axis']['values'])-5)
        
        line_data['legend'] = plot_labels[plot_idx]
        
        plot_config['lines'].append(line_data)
        
        
        if plot_stdDev and f_vectors[plot_idx]['statistics'] in ['mean']:
            line_data_2 = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'y_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'cutoff': None,
                     'color': None,
                     'alpha': 0.1,
                     'legend': '_nolegend_',
                     'marker_style': dict(linestyle='-', linewidth=1.0)
                }
        
            line_data_2['x_axis']['values'] = r_vectors['var_values']
            line_data_2['y_axis']['values'] = (y_plot_data + f_vectors[plot_idx]['var_stdevs']) / normalization_factor
            line_data_2['cutoff'] = slice(0, len(line_data_2['x_axis']['values']))
            
            line_data_2['color'] = plot_colors[plot_idx]
            
            plot_config['lines'].append(line_data_2)
            
            line_data_3 = copy.deepcopy(line_data_2)
            line_data_3['y_axis']['values'] = (y_plot_data - f_vectors[plot_idx]['var_stdevs']) / normalization_factor
            plot_config['lines'].append(line_data_3)
            
    title_string = 'Side Scattering - Parameter Range:'
    for sweep_param_value in list(sweep_parameters.values()):
        optimized_value = sweep_param_value['var_values'][sweep_param_value['peakInd']]
        title_string += '\n' + sweep_param_value['var_name'] + f': Optimized at {optimized_value}'
    plot_config['title'] = title_string
    
    sweep_variable_idx = [index for (index, item) in enumerate(slice_coords) if type(item) is slice][0]
    plot_config['x_axis']['label'] = list(sweep_parameters.values())[sweep_variable_idx]['var_name']
    plot_config['y_axis']['label'] = 'Side Scattering Power'
    
    
    fig, ax = enter_plot_data_1d(plot_config, fig, ax)
    
    plot_subfolder_name = ''
    if not os.path.isdir(plot_directory_location + '/' + plot_subfolder_name):
        os.makedirs(plot_directory_location + '/' + plot_subfolder_name)
    plt.savefig(plot_directory_location + '/' + plot_subfolder_name\
        + '/' + 'side_power_sweep_' + create_parameter_filename_string(slice_coords) + '.png',
        bbox_inches='tight')
    print('Exported: Side Power Sweep: ' + create_parameter_filename_string(slice_coords))
    
    return fig, ax

def plot_focal_power_sweep_1d(plot_data, slice_coords, plot_stdDev = True):
    '''Produces a 1D focal region power plot using given coordinates to slice the N-D array of data contained in plot_data.
    X-axis i.e. sweep variable is given by the index of the entry of type 'slice' in the slice_coords.'''
    
    # Whichever entry is a slice means that variable in r_vectors is the x-axis of this plot
    r_vector_value_idx = [index for (index , item) in enumerate(slice_coords) if isinstance(item, slice)][0]
    r_vectors = list(plot_data['r'].values())[r_vector_value_idx]
    f_vectors = plot_data['f']              # This is a list of N-D arrays, containing the function values for each parameter combination
    
    fig, ax = plt.subplots()
    plot_config = {'title': None,
                   'x_axis': {'label':'', 'limits':[]},
                   'y_axis': {'label':'', 'limits':[]},
                   'lines': []
                }
    
    plot_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plot_labels = ['Focal Region Power']
    normalize_against_max = False
    
    for plot_idx in range(0, len(f_vectors)):
        line_data = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
                        'y_axis': {'values': None, 'factor': 1, 'offset': 0},
                        'cutoff': None,
                        'color': None,
                        'alpha': 1.0,
                        'legend': None,
                        'marker_style': marker_style
                }
        
        
        y_plot_data = f_vectors[plot_idx]['var_values']
        y_plot_data = np.reshape(y_plot_data, (len(y_plot_data), 1))
        y_plot_data = y_plot_data[tuple(slice_coords)]
        
        normalization_factor = np.max(y_plot_data) if normalize_against_max else 1
        line_data['y_axis']['values'] = y_plot_data / normalization_factor
        line_data['x_axis']['values'] = r_vectors['var_values']
        line_data['cutoff'] = slice(0, len(line_data['x_axis']['values'])-5)
        
        line_data['legend'] = plot_labels[plot_idx]
        
        plot_config['lines'].append(line_data)
        
        
        if plot_stdDev and f_vectors[plot_idx]['statistics'] in ['mean']:
            line_data_2 = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'y_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'cutoff': None,
                     'color': None,
                     'alpha': 0.1,
                     'legend': '_nolegend_',
                     'marker_style': dict(linestyle='-', linewidth=1.0)
                }
        
            line_data_2['x_axis']['values'] = r_vectors['var_values']
            line_data_2['y_axis']['values'] = (y_plot_data + f_vectors[plot_idx]['var_stdevs']) / normalization_factor
            line_data_2['cutoff'] = slice(0, len(line_data_2['x_axis']['values']))
            
            line_data_2['color'] = plot_colors[plot_idx]
            
            plot_config['lines'].append(line_data_2)
            
            line_data_3 = copy.deepcopy(line_data_2)
            line_data_3['y_axis']['values'] = (y_plot_data - f_vectors[plot_idx]['var_stdevs']) / normalization_factor
            plot_config['lines'].append(line_data_3)
            
    title_string = 'Focal Region Power - Parameter Range:'
    for sweep_param_value in list(sweep_parameters.values()):
        optimized_value = sweep_param_value['var_values'][sweep_param_value['peakInd']]
        title_string += '\n' + sweep_param_value['var_name'] + f': Optimized at {optimized_value}'
    plot_config['title'] = title_string
    
    sweep_variable_idx = [index for (index, item) in enumerate(slice_coords) if type(item) is slice][0]
    plot_config['x_axis']['label'] = list(sweep_parameters.values())[sweep_variable_idx]['var_name']
    plot_config['y_axis']['label'] = 'Focal Region Power'
    
    
    fig, ax = enter_plot_data_1d(plot_config, fig, ax)
    
    plot_subfolder_name = ''
    if not os.path.isdir(plot_directory_location + '/' + plot_subfolder_name):
        os.makedirs(plot_directory_location + '/' + plot_subfolder_name)
    plt.savefig(plot_directory_location + '/' + plot_subfolder_name\
        + '/' + 'focal_power_sweep_' + create_parameter_filename_string(slice_coords) + '.png',
        bbox_inches='tight')
    print('Exported: Focal Power Sweep: ' + create_parameter_filename_string(slice_coords))
    
    return fig, ax

def plot_scatter_plane_power_sweep_1d(plot_data, slice_coords, plot_stdDev = True):
    '''Produces a 1D scatter plane region power plot using given coordinates to slice the N-D array of data contained in plot_data.
    X-axis i.e. sweep variable is given by the index of the entry of type 'slice' in the slice_coords.'''
    
    # Whichever entry is a slice means that variable in r_vectors is the x-axis of this plot
    r_vector_value_idx = [index for (index , item) in enumerate(slice_coords) if isinstance(item, slice)][0]
    r_vectors = list(plot_data['r'].values())[r_vector_value_idx]
    f_vectors = plot_data['f']              # This is a list of N-D arrays, containing the function values for each parameter combination
    
    fig, ax = plt.subplots()
    plot_config = {'title': None,
                   'x_axis': {'label':'', 'limits':[]},
                   'y_axis': {'label':'', 'limits':[]},
                   'lines': []
                }
    plot_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plot_labels = ['Scatter Plane Power']
    normalize_against_max = False
    
    for plot_idx in range(0, len(f_vectors)):
        line_data = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
                        'y_axis': {'values': None, 'factor': 1, 'offset': 0},
                        'cutoff': None,
                        'color': None,
                        'alpha': 1.0,
                        'legend': None,
                        'marker_style': marker_style
                }
        
        
        y_plot_data = f_vectors[plot_idx]['var_values']
        y_plot_data = np.reshape(y_plot_data, (len(y_plot_data), 1))
        y_plot_data = y_plot_data[tuple(slice_coords)]
        
        normalization_factor = np.max(y_plot_data) if normalize_against_max else 1
        line_data['y_axis']['values'] = y_plot_data / normalization_factor
        line_data['x_axis']['values'] = r_vectors['var_values']
        line_data['cutoff'] = slice(0, len(line_data['x_axis']['values'])-5)
        
        line_data['legend'] = plot_labels[plot_idx]
        
        plot_config['lines'].append(line_data)
        
        
        if plot_stdDev and f_vectors[plot_idx]['statistics'] in ['mean']:
            line_data_2 = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'y_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'cutoff': None,
                     'color': None,
                     'alpha': 0.1,
                     'legend': '_nolegend_',
                     'marker_style': dict(linestyle='-', linewidth=1.0)
                }
        
            line_data_2['x_axis']['values'] = r_vectors['var_values']
            line_data_2['y_axis']['values'] = (y_plot_data + f_vectors[plot_idx]['var_stdevs']) / normalization_factor
            line_data_2['cutoff'] = slice(0, len(line_data_2['x_axis']['values']))
            
            line_data_2['color'] = plot_colors[plot_idx]
            
            plot_config['lines'].append(line_data_2)
            
            line_data_3 = copy.deepcopy(line_data_2)
            line_data_3['y_axis']['values'] = (y_plot_data - f_vectors[plot_idx]['var_stdevs']) / normalization_factor
            plot_config['lines'].append(line_data_3)
            
    title_string = 'Focal Plane Scattering Power - Parameter Range:'
    for sweep_param_value in list(sweep_parameters.values()):
        optimized_value = sweep_param_value['var_values'][sweep_param_value['peakInd']]
        title_string += '\n' + sweep_param_value['var_name'] + f': Optimized at {optimized_value}'
    plot_config['title'] = title_string
    
    sweep_variable_idx = [index for (index, item) in enumerate(slice_coords) if type(item) is slice][0]
    plot_config['x_axis']['label'] = list(sweep_parameters.values())[sweep_variable_idx]['var_name']
    plot_config['y_axis']['label'] = 'Focal Plane Scattering Power'
    
    
    fig, ax = enter_plot_data_1d(plot_config, fig, ax)
    
    plot_subfolder_name = ''
    if not os.path.isdir(plot_directory_location + '/' + plot_subfolder_name):
        os.makedirs(plot_directory_location + '/' + plot_subfolder_name)
    plt.savefig(plot_directory_location + '/' + plot_subfolder_name\
        + '/' + 'focal_power_sweep_' + create_parameter_filename_string(slice_coords) + '.png',
        bbox_inches='tight')
    print('Exported: Focal Power Sweep: ' + create_parameter_filename_string(slice_coords))
    
    return fig, ax

def plot_exit_power_distribution_sweep_1d(plot_data, slice_coords, plot_stdDev = True):
    '''Produces a 1D power plot of focal plane power distribution using given coordinates to slice the N-D array 
    of data contained in plot_data.
    X-axis i.e. sweep variable is given by the index of the entry of type 'slice' in the slice_coords.'''
    
    # Whichever entry is a slice means that variable in r_vectors is the x-axis of this plot
    r_vector_value_idx = [index for (index , item) in enumerate(slice_coords) if isinstance(item, slice)][0]
    r_vectors = list(plot_data['r'].values())[r_vector_value_idx]
    f_vectors = plot_data['f']              # This is a list of N-D arrays, containing the function values for each parameter combination
    
    fig, ax = plt.subplots()
    plot_config = {'title': None,
                   'x_axis': {'label':'', 'limits':[]},
                   'y_axis': {'label':'', 'limits':[]},
                   'lines': []
                }
    plot_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plot_labels = ['Focal Region Power', 'Oblique Scattering']
    normalize_against_max = False
    
    for plot_idx in range(0, len(f_vectors)):
        line_data = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
                        'y_axis': {'values': None, 'factor': 1, 'offset': 0},
                        'cutoff': None,
                        'color': None,
                        'alpha': 1.0,
                        'legend': None,
                        'marker_style': marker_style
                }
        
        
        y_plot_data = f_vectors[plot_idx]['var_values']
        y_plot_data = np.reshape(y_plot_data, (len(y_plot_data), 1))
        y_plot_data = y_plot_data[tuple(slice_coords)]
        
        normalization_factor = np.max(y_plot_data) if normalize_against_max else 1
        line_data['y_axis']['values'] = y_plot_data / normalization_factor
        line_data['x_axis']['values'] = r_vectors['var_values']
        line_data['cutoff'] = slice(0, len(line_data['x_axis']['values'])-5)
        
        line_data['legend'] = plot_labels[plot_idx]
        
        plot_config['lines'].append(line_data)
        
        
        if plot_stdDev and f_vectors[plot_idx]['statistics'] in ['mean']:
            line_data_2 = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'y_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'cutoff': None,
                     'color': None,
                     'alpha': 0.1,
                     'legend': '_nolegend_',
                     'marker_style': dict(linestyle='-', linewidth=1.0)
                }
        
            line_data_2['x_axis']['values'] = r_vectors['var_values']
            line_data_2['y_axis']['values'] = (y_plot_data + f_vectors[plot_idx]['var_stdevs']) / normalization_factor
            line_data_2['cutoff'] = slice(0, len(line_data_2['x_axis']['values']))
            
            line_data_2['color'] = plot_colors[plot_idx]
            
            plot_config['lines'].append(line_data_2)
            
            line_data_3 = copy.deepcopy(line_data_2)
            line_data_3['y_axis']['values'] = (y_plot_data - f_vectors[plot_idx]['var_stdevs']) / normalization_factor
            plot_config['lines'].append(line_data_3)
            
    title_string = 'Power at Focal Plane - Parameter Range:'
    for sweep_param_value in list(sweep_parameters.values()):
        optimized_value = sweep_param_value['var_values'][sweep_param_value['peakInd']]
        title_string += '\n' + sweep_param_value['var_name'] + f': Optimized at {optimized_value}'
    plot_config['title'] = title_string
    
    sweep_variable_idx = [index for (index, item) in enumerate(slice_coords) if type(item) is slice][0]
    plot_config['x_axis']['label'] = list(sweep_parameters.values())[sweep_variable_idx]['var_name']
    plot_config['y_axis']['label'] = 'Normalized Focal Plane Power'
    
    
    fig, ax = enter_plot_data_1d(plot_config, fig, ax)
    
    plot_subfolder_name = ''
    if not os.path.isdir(plot_directory_location + '/' + plot_subfolder_name):
        os.makedirs(plot_directory_location + '/' + plot_subfolder_name)
    plt.savefig(plot_directory_location + '/' + plot_subfolder_name\
        + '/' + 'exit_power_dist_sweep_' + create_parameter_filename_string(slice_coords) + '.png',
        bbox_inches='tight')
    print('Exported: Exit Power Distribution Sweep: ' + create_parameter_filename_string(slice_coords))
    
    return fig, ax

def plot_device_rta_sweep_1d(plot_data, slice_coords, plot_stdDev = True):
    '''Produces a 1D power plot of device RTA using given coordinates to slice the N-D array 
    of data contained in plot_data.
    X-axis i.e. sweep variable is given by the index of the entry of type 'slice' in the slice_coords.'''
    
    # Whichever entry is a slice means that variable in r_vectors is the x-axis of this plot
    r_vector_value_idx = [index for (index , item) in enumerate(slice_coords) if isinstance(item, slice)][0]
    r_vectors = list(plot_data['r'].values())[r_vector_value_idx]
    f_vectors = plot_data['f']              # This is a list of N-D arrays, containing the function values for each parameter combination
    
    fig, ax = plt.subplots()
    plot_config = {'title': None,
                   'x_axis': {'label':'', 'limits':[]},
                   'y_axis': {'label':'', 'limits':[]},
                   'lines': []
                }
    plot_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plot_labels = ['Device Reflection', 'Side Scattering E', 'Side Scattering N', 'Side Scattering W', 'Side Scattering S',
                   'Focal Region Power', 'Oblique Scattering Power', 'Absorbed Power']
    normalize_against_max = False
    
    for plot_idx in range(0, len(f_vectors)):
        line_data = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
                        'y_axis': {'values': None, 'factor': 1, 'offset': 0},
                        'cutoff': None,
                        'color': None,
                        'alpha': 1.0,
                        'legend': None,
                        'marker_style': marker_style
                }
        
        
        y_plot_data = f_vectors[plot_idx]['var_values']
        y_plot_data = np.reshape(y_plot_data, (len(y_plot_data), 1))
        y_plot_data = y_plot_data[tuple(slice_coords)]
        
        normalization_factor = np.max(y_plot_data) if normalize_against_max else 1
        line_data['y_axis']['values'] = y_plot_data / normalization_factor
        line_data['x_axis']['values'] = r_vectors['var_values']
        line_data['cutoff'] = slice(0, len(line_data['x_axis']['values'])-5)
        
        line_data['legend'] = plot_labels[plot_idx]
        
        plot_config['lines'].append(line_data)
        
        
        if plot_stdDev and f_vectors[plot_idx]['statistics'] in ['mean']:
            line_data_2 = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'y_axis': {'values': None, 'factor': 1, 'offset': 0},
                     'cutoff': None,
                     'color': None,
                     'alpha': 0.1,
                     'legend': '_nolegend_',
                     'marker_style': dict(linestyle='-', linewidth=1.0)
                }
        
            line_data_2['x_axis']['values'] = r_vectors['var_values']
            line_data_2['y_axis']['values'] = (y_plot_data + f_vectors[plot_idx]['var_stdevs']) / normalization_factor
            line_data_2['cutoff'] = slice(0, len(line_data_2['x_axis']['values']))
            
            line_data_2['color'] = plot_colors[plot_idx]
            
            plot_config['lines'].append(line_data_2)
            
            line_data_3 = copy.deepcopy(line_data_2)
            line_data_3['y_axis']['values'] = (y_plot_data - f_vectors[plot_idx]['var_stdevs']) / normalization_factor
            plot_config['lines'].append(line_data_3)
            
    title_string = 'Device RTA - Parameter Range:'
    for sweep_param_value in list(sweep_parameters.values()):
        optimized_value = sweep_param_value['var_values'][sweep_param_value['peakInd']]
        title_string += '\n' + sweep_param_value['var_name'] + f': Optimized at {optimized_value}'
    plot_config['title'] = title_string
    
    sweep_variable_idx = [index for (index, item) in enumerate(slice_coords) if type(item) is slice][0]
    plot_config['x_axis']['label'] = list(sweep_parameters.values())[sweep_variable_idx]['var_name']
    plot_config['y_axis']['label'] = 'Normalized Power'
    
    
    fig, ax = enter_plot_data_1d(plot_config, fig, ax)
    
    plot_subfolder_name = ''
    if not os.path.isdir(plot_directory_location + '/' + plot_subfolder_name):
        os.makedirs(plot_directory_location + '/' + plot_subfolder_name)
    plt.savefig(plot_directory_location + '/' + plot_subfolder_name\
        + '/' + 'device_rta_sweep_' + create_parameter_filename_string(slice_coords) + '.png',
        bbox_inches='tight')
    print('Exported: Device RTA Sweep: ' + create_parameter_filename_string(slice_coords))
    
    return fig, ax

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
                        
                    
                for xy_idx in range(0, 1): # range(0,2) if considering source with both polarizations
                    disable_all_sources()
                    disable_objects(disable_object_list)

                    fdtd_hook.select(forward_sources[xy_idx]['name'])
                    fdtd_hook.set('enabled', 1)
                    
                    job_name = 'sweep_job_' + \
                        parameter_filename_string + '.fsp'
                    fdtd_hook.save(projects_directory_location +
                                    "/optimization.fsp")
                    print('Saved out ' + job_name)
                    job_names[('sweep', xy_idx, idx)
                                ] = add_job(job_name, jobs_queue)

                    
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
                        fdtd_hook.load(
                            convert_root_folder(job_names[('sweep',xy_idx,idx)],
                                                projects_directory_location)
                        )
                        
                    monitors_data = {}
                    # Add wavelength and sourcepower information, common to every monitor in a job
                    monitors_data['wavelength'] = np.divide(3e8, fdtd_hook.getresult('transmission_focal_monitor_','f'))
                    monitors_data['sourcepower'] = get_sourcepower()
                    
                    # Go through each individual monitor and extract respective data
                    for monitor in monitors:
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
        
        
        # TODO: Break up this part and make sure they are truly separate.
        # TODO: The lumapi functions in the get_xx_spectrum functions all need to be removed and replaced with references to the monitor dictionaries instead.
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
                                    generate_sorting_eff_spectrum(monitors_data, plot['generate_plot_per_job'])
                                    
                            elif plot['name'] in ['Enorm_focal_plane_image']:
                                plot_data[plot['name']+'_spectrum'], plot_data[plot['name']+'_sweep'] = \
                                    generate_Enorm_focal_plane(monitors_data, plot['generate_plot_per_job'])
                                
                            elif plot['name'] in ['device_transmission']:
                                plot_data[plot['name']+'_spectrum'], plot_data[plot['name']+'_sweep'] = \
                                    generate_device_transmission_spectrum(monitors_data, plot['generate_plot_per_job'])
                                    
                            elif plot['name'] in ['device_reflection']:
                                plot_data[plot['name']+'_spectrum'], plot_data[plot['name']+'_sweep'] = \
                                    generate_overall_reflection_spectrum(monitors_data, plot['generate_plot_per_job'])
                                    
                            elif plot['name'] in ['side_scattering_power']:
                                plot_data[plot['name']+'_spectrum'], plot_data[plot['name']+'_sweep'] = \
                                    generate_side_power_spectrum(monitors_data, plot['generate_plot_per_job'])
                                    
                            elif plot['name'] in ['focal_region_power']:
                                plot_data[plot['name']+'_spectrum'], plot_data[plot['name']+'_sweep'] = \
                                    generate_focal_power_spectrum(monitors_data, plot['generate_plot_per_job'])
                                    
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
            
            for job_idx, job_name in job_names.items():
                plot_data = plots_data[job_name]
                
                print('Plotting for Job: ' + isolate_filename(job_names[job_idx]))
                
                for plot in plots:
                    if plot['enabled']:
                        continue
                        # if plot['name'] in ['sorting_efficiency']:
                        #     plot_sorting_eff_spectrum(plot_data[plot['name']+'_spectrum'], job_idx)
                            
                        # elif plot['name'] in ['Enorm_focal_plane_image']:
                        #     plot_Enorm_focal_plane_image_spectrum(plot_data[plot['name']+'_spectrum'], job_idx,
                        #                 plot_wavelengths = monitors_data['wavelength'][[n['index'] for n in plot_data['sorting_efficiency_sweep']]])
                        #                 # Second part gets the wavelengths of each spectra where sorting eff. peaks
                        #     #continue
                        
                        # elif plot['name'] in ['device_transmission']:
                        #     plot_device_transmission_spectrum(plot_data[plot['name']+'_spectrum'], job_idx)
                            
                        # elif plot['name'] in ['device_reflection']:
                        #     plot_overall_reflection_spectrum(plot_data[plot['name']+'_spectrum'], job_idx)
                            
                        # elif plot['name'] in ['side_scattering_power']:
                        #     plot_side_power_spectrum(plot_data[plot['name']+'_spectrum'], job_idx)
                        
                        # elif plot['name'] in ['focal_region_power']:
                        #     plot_focal_power_spectrum(plot_data[plot['name']+'_spectrum'], job_idx)
                            
                        # elif plot['name'] in ['focal_scattering_power']:
                        #     plot_scatter_plane_power_spectrum(plot_data[plot['name']+'_spectrum'], job_idx)
                            
                        # elif plot['name'] in ['exit_power_distribution']:
                        #     plot_exit_power_distribution_spectrum(plot_data[plot['name']+'_spectrum'], job_idx)
                            
                        # elif plot['name'] in ['device_rta']:
                        #     plot_device_rta_spectrum(plot_data[plot['name']+'_spectrum'], job_idx)
                                
                        
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
                        
                    elif plot['name'] in ['device_transmission']:
                        plot_device_transmission_sweep_1d(plots_data['sweep_plots'][plot['name']], [slice(None), 0], plot_stdDev=False)
                        
                    elif plot['name'] in ['device_reflection']:
                        plot_overall_reflection_sweep_1d(plots_data['sweep_plots'][plot['name']], [slice(None), 0], plot_stdDev=False)
                        
                    elif plot['name'] in ['side_scattering_power']:
                        plot_side_power_sweep_1d(plots_data['sweep_plots'][plot['name']], [slice(None), 0], plot_stdDev=False)
                    
                    elif plot['name'] in ['focal_region_power']:
                        plot_focal_power_sweep_1d(plots_data['sweep_plots'][plot['name']], [slice(None), 0], plot_stdDev=False)
                        
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
            
            
print('Reached end of file. Code completed.')

endtime = time.time()
print(f'{(endtime-startTime)/60} minutes')