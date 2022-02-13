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
import h5py
import numpy as np
import time

import queue
import subprocess
import platform
import re

print("All modules loaded.")
sys.stdout.flush()

def backup_all_vars():
    '''Performs the equivalent of MATLAB's save(), albeit with some workarounds. Lumapi SimObjects cannot be saved this way.'''
    global my_shelf
    my_shelf = shelve.open(shelf_fn,'n')
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
    
def load_backup_vars():
    '''Restores session variables from a save_state file. Debug purposes.'''
    #! This will overwrite all current session variables!
    # except for the following:
    do_not_overwrite = [running_on_local_machine,
                        lumapi_filepath,
                        start_from_step,
                        projects_directory_location,
                        projects_directory_location_init]
    
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
# Create FDTD hook
#
fdtd_hook = lumapi.FDTD()


#
# Create project folder and save out the parameter file for documentation for this optimization
#
python_src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))

if not os.path.isdir(projects_directory_location):
	os.mkdir(projects_directory_location)

log_file = open( projects_directory_location + "/log.txt", 'w' )
log_file.write( "---Log for Lumerical Processing Sweep---\n" )
log_file.close()

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
    # No checks are done to see if there are discrepancies.
    device_size_lateral_um = fdtd_hook.getnamed('design_import','x span') / 1e-6
    device_vertical_maximum_um = fdtd_hook.getnamed('design_import','z max') / 1e-6
    device_vertical_minimum_um = fdtd_hook.getnamed('design_import','z min') / 1e-6
    fdtd_region_size_lateral_um = fdtd_hook.getnamed('FDTD','x span') / 1e-6
    fdtd_region_maximum_vertical_um = fdtd_hook.getnamed('FDTD','z max') / 1e-6
    fdtd_region_minimum_vertical_um = fdtd_hook.getnamed('FDTD','z min') / 1e-6
    fdtd_region_size_vertical_um = fdtd_region_maximum_vertical_um + abs(fdtd_region_minimum_vertical_um)
    monitorbox_size_lateral_um = device_size_lateral_um + 2 * sidewall_thickness_um
    monitorbox_vertical_maximum_um = device_vertical_maximum_um + 0.2
    monitorbox_vertical_minimum_um = device_vertical_minimum_um - 0.2
    adjoint_vertical_um = fdtd_hook.getnamed('transmission_focal_monitor_', 'z') / 1e-6

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
    
    for sm_idx in range(0,4):
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
    
    spill_plane_monitor = {}
    spill_plane_monitor['x'] = 0 * 1e-6
    spill_plane_monitor['x span'] = ((fdtd_hook.getnamed('transmission_focal_monitor_','x span') /2 * 4) / 1e-6) * 1e-6
    spill_plane_monitor['y'] = 0 * 1e-6
    spill_plane_monitor['y span'] = ((fdtd_hook.getnamed('transmission_focal_monitor_','y span') /2 * 4) / 1e-6) * 1e-6
    spill_plane_monitor['z'] = adjoint_vertical_um * 1e-6
    spill_plane_monitor['override global monitor settings'] = 1
    spill_plane_monitor['use wavelength spacing'] = 1
    spill_plane_monitor['use source limits'] = 1
    spill_plane_monitor['frequency points'] = num_design_frequency_points
    fdtd_objects['spill_plane_monitor'] = spill_plane_monitor
    
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
    
    for sidewall_idx in range(0,4):
        sidewall_mesh = {}
        sidewall_mesh['name'] = 'mesh_sidewall_' + str(sidewall_idx)
        sidewall_mesh['override x mesh'] = fdtd_hook.getnamed(sidewall_mesh['name'], 'override x mesh')
        if sidewall_mesh['override x mesh']:
            fdtd_hook.setnamed(sidewall_mesh['name'], 'dx', fdtd_hook.getnamed('device_mesh','dx') / 4)
        else:
            fdtd_hook.setnamed(sidewall_mesh['name'], 'dy', fdtd_hook.getnamed('device_mesh','dy') / 4)
        
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
            # fdtd_hook.addmesh(name=sidewall_mesh['name'])
            pass
        

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
        
    # TODO: The following assumes that the source is exactly at the interface between air and the topmost substrate.
    # TODO: Write code to take into account the case where the source is located in between or within a top substrate
    # TODO: Will have to explicitly read the z height of the source

    src_angle_incidence = np.radians(src_angle_incidence)
    snellConst = float(indices[-1])*np.sin(src_angle_incidence)
    input_theta = np.degrees(np.arcsin(snellConst/float(indices[0])))

    r_offset = 0
    for cnt, h_i in enumerate(heights):
        s_i = snellConst/indices[cnt]
        r_offset = r_offset + h_i*s_i/np.sqrt(1-s_i**2)

    return [input_theta, r_offset]

def change_source_theta(theta_value, phi_value):
    fdtd_hook.select('forward_src_x')
    
    # Structure the objects list such that the last entry is the last medium above the device.
    input_theta, r_offset = snellRefrOffset(theta_value, objects_above_device)
    
    fdtd_hook.set('angle theta', input_theta) # degrees
    fdtd_hook.set('angle phi', phi_value) # degrees
    fdtd_hook.set('x', r_offset*np.cos(np.radians(phi_value)) * 1e-6)
    fdtd_hook.set('y', r_offset*np.sin(np.radians(phi_value)) * 1e-6)
    
    print(f'Theta set to {input_theta}, Phi set to {phi_value}.')

def change_source_beam_radius(param_value):
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
    
def change_fdtd_region_size(param_value):
    print('3')
    
def change_sidewall_thickness(param_value):
    print('3')


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

def get_transmission_magnitude(monitor_name):
    read_T = fdtd_hook.getresult(monitor_name, 'T')
    return np.abs(read_T['T'])



#
# Set up some numpy arrays to handle all the data we will pull out of the simulation.
#

forward_e_fields = {}
focal_data = {}
quadrant_efield_data = {}
quadrant_transmission_data = {}

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
    current_job_creation_idx = (0,0,0)
    
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
                current_job_creation_idx = idx
                print(f'Creating Job: Current parameter index is {current_job_creation_idx}')
                
                for t_idx, p_idx in enumerate(idx):
                    
                    variable = list(sweep_parameters.values())[t_idx]
                    variable_name = variable['var_name']
                    variable_value = variable['var_values'][p_idx]
                    
                    sweep_parameter_value_array[idx]
                
                
                    fdtd_hook.switchtolayout()
                    
                    # depending on what variable is,
                    # edit Lumerical
                    
                    if variable_name in ['angle_theta', 'angle_phi']:
                        change_source_theta(variable_value, angle_phi)
                    elif variable_name in ['beam_radius_um']:
                        change_source_beam_radius(variable_value)
                    
                    
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
        # Step 2: Grab the various fields
        #
        
        if start_from_step == 0 or start_from_step == 2:
            if start_from_step == 2:
                load_backup_vars()
                from NTTArrParameters import *
                
            print("Beginning Step 2: Computing Figure of Merit")
            sys.stdout.flush()
            
            # Initialize shape and keys of dictionaries for later storage
            for xy_idx in range(0, 2):
                focal_data[xy_names[xy_idx]] = [
                    None for idx in range(0, num_focal_spots)]
                quadrant_efield_data[xy_names[xy_idx]] = [
                    None for idx in range(0, num_focal_spots)]
                quadrant_transmission_data[xy_names[xy_idx]] = [
                    None for idx in range(0, num_focal_spots)]
        
            for dispersive_range_idx in range(0, num_dispersive_ranges):
                for xy_idx in range(0, 2):
                    if start_from_step == 0:
                        fdtd_hook.load(
                            job_names[('forward', xy_idx, dispersive_range_idx)])
                    else:
                        # Extract filename, directly declare current directory address
                        fdtd_hook.load(
                            projects_directory_location + '/' + 
                            job_names[('forward', xy_idx, dispersive_range_idx)].rsplit('/',1)[1]
                        )

                    fwd_e_fields = get_efield(transmission_monitor_focal['name'])

                    if (dispersive_range_idx == 0):
                        forward_e_fields[xy_names[xy_idx]] = fwd_e_fields
                        
                    for adj_src_idx in dispersive_range_to_adjoint_src_map[dispersive_range_idx]:
                        focal_data[xy_names[xy_idx]][adj_src_idx] = get_efield(
                            focal_monitors[adj_src_idx]['name'])

                        quadrant_efield_data[xy_names[xy_idx]][adj_src_idx] = get_efield(
                            transmission_monitors[adj_src_idx]['name'])
                        
                        quadrant_transmission_data[xy_names[xy_idx]][adj_src_idx] = get_transmission_magnitude(
                            transmission_monitors[adj_src_idx]['name'])
        
            np.save(projects_directory_location +
                    "/focal_plane_efields.npy", forward_e_fields)
            np.save(projects_directory_location +
                    "/focal_data.npy", focal_data)
            np.save(projects_directory_location + 
                    "/quadrant_efield_data.npy", quadrant_efield_data)
            np.save(projects_directory_location + 
                    "/quadrant_transmission_data.npy", quadrant_transmission_data)
                        
            print("Completed Step 2: Saved out figures of merit.")
            sys.stdout.flush()
            
            backup_all_vars()


print('Reached end of file. Code completed.')