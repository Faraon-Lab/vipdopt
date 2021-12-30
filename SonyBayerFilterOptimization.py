import os
import shutil
import sys

# Add filepath to default path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from SonyBayerFilterParameters import *
import SonyBayerFilter

from scipy import interpolate

# Lumerical hook Python library API
import imp
imp.load_source( "lumapi", "/central/home/gdrobert/Develompent/lumerical/v212/api/python/lumapi.py")
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


def index_from_permittivity(permittivity_):
    """Checks all permittivity values are real and then takes square root to give index."""
    assert np.all(np.imag(permittivity_) ==
                  0), 'Not expecting complex index values right now!'

    return np.sqrt(permittivity_)


class DevicePermittivityModel():
    def average_permittivity(self, wl_range_um_):
        return max_device_permittivity


def softplus(x_in):
    return np.log(1 + np.exp(x_in))


def softplus_prime(x_in):
    return (1. / (1 + np.exp(-x_in)))


def get_slurm_node_list(slurm_job_env_variable=None):
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


num_nodes_available = int(sys.argv[1])
num_cpus_per_node = 8
cluster_hostnames = get_slurm_node_list()


#
# Create FDTD hook
#
fdtd_hook = lumapi.FDTD()


#
# Create project folder and save out the parameter file for documentation for this optimization
#
python_src_directory = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '.'))

# Initial Project File Directory
projects_directory_location_init = "/central/groups/Faraon_Computing/ian/sony"
projects_directory_location_init += "/" + project_name_init

# Final Output Project File Directory
projects_directory_location = "/central/groups/Faraon_Computing/ian/sony"
projects_directory_location += "/" + project_name

if not os.path.isdir(projects_directory_location):
    os.mkdir(projects_directory_location)


log_file = open(projects_directory_location + "/log.txt", 'w')
log_file.write("Log\n")
log_file.close()

# Create new file called optimization.fsp and save
fdtd_hook.newproject()
fdtd_hook.save(projects_directory_location + "/optimization")
# Copy Python code files to the output project directory
shutil.copy2(python_src_directory + "/SonyBayerFilterParameters.py",
             projects_directory_location + "/SonyBayerFilterParameters.py")
shutil.copy2(python_src_directory + "/SonyBayerFilter.py",
             projects_directory_location + "/SonyBayerFilter.py")
shutil.copy2(python_src_directory + "/SonyBayerFilterOptimization.py",
             projects_directory_location + "/SonyBayerFilterOptimization.py")


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

device_mesh = fdtd_hook.addmesh()
device_mesh['name'] = 'device_mesh'
device_mesh['x'] = 0
device_mesh['x span'] = fdtd_region_size_lateral_um * 1e-6
device_mesh['y'] = 0
device_mesh['y span'] = fdtd_region_size_lateral_um * 1e-6
device_mesh['z max'] = (device_vertical_maximum_um + 0.5) * 1e-6
device_mesh['z min'] = (device_vertical_minimum_um - 0.5) * 1e-6
device_mesh['dx'] = mesh_spacing_um * 1e-6
device_mesh['dy'] = mesh_spacing_um * 1e-6
device_mesh['dz'] = mesh_spacing_um * 1e-6

#
# General polarized source information
#
xy_pol_rotations = [0, 90]
xy_names = ['x', 'y']

#
# Add a TFSF plane wave forward source at normal incidence
# for both x- and y-polarization
#
forward_sources = []


def gaussRefrOffset(src_angle_incidence, heights, indices):
    # structured such that n0 is the input interface and n_end is the last above the device
    # height_0 should be 0

    src_angle_incidence = np.radians(src_angle_incidence)
    snellConst = indices[-1]*np.sin(src_angle_incidence)
    input_theta = np.degrees(np.arcsin(snellConst/indices[0]))

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

    # adjustedRIParams = gaussRefrOffset(src_angle_incidence, [0, src_hgt_Si, src_hgt_polymer], [1, n_Si, n_polymer])
    adjustedRIParams = gaussRefrOffset(src_angle_incidence, [0], [1])

    forward_src = fdtd_hook.addgaussian()
    forward_src['name'] = 'forward_src_'+xy_names[xy_idx]
    forward_src['injection axis'] = 'z-axis'
    forward_src['direction'] = 'Backward'
    forward_src['polarization angle'] = xy_pol_rotations[xy_idx]  # degrees
    forward_src['angle theta'] = adjustedRIParams[0]  # degrees
    forward_src['angle phi'] = src_phi_incidence    # degrees

    forward_src['x'] = adjustedRIParams[1] * \
        np.cos(src_phi_incidence*np.pi/180) * 1e-6
    forward_src['x span'] = lateral_aperture_um * 1e-6 * (4/3)
    forward_src['y'] = adjustedRIParams[1] * \
        np.sin(src_phi_incidence*np.pi/180) * 1e-6
    forward_src['y span'] = lateral_aperture_um * 1e-6 * (4/3)
    forward_src['z'] = (src_maximum_vertical_um + src_hgt_Si) * 1e-6
    forward_src['wavelength start'] = lambda_min_um * 1e-6
    forward_src['wavelength stop'] = lambda_max_um * 1e-6

    forward_src["frequency dependent profile"] = 1
    forward_src["number of field profile samples"] = 150
    forward_src['beam parameters'] = 'Beam size and divergence angle'
    forward_src['divergence angle'] = 0.5 * \
        (lambda_min_um+lambda_max_um)*1e-6/(np.pi*src_beam_rad)*(180/np.pi)
    forward_src['beam radius wz'] = src_beam_rad

    forward_sources.append(forward_src)

#
# Place dipole adjoint sources at the focal plane that can ring in both
# x-axis and y-axis
#
adjoint_sources = []

for adj_src_idx in range(0, num_adjoint_sources):
    adjoint_sources.append([])
    for xy_idx in range(0, 2):
        adj_src = fdtd_hook.adddipole()
        adj_src['name'] = 'adj_src_' + str(adj_src_idx) + xy_names[xy_idx]
        adj_src['x'] = adjoint_x_positions_um[adj_src_idx] * 1e-6
        adj_src['y'] = adjoint_y_positions_um[adj_src_idx] * 1e-6
        adj_src['z'] = adjoint_vertical_um * 1e-6
        adj_src['theta'] = 90
        adj_src['phi'] = xy_pol_rotations[xy_idx]
        adj_src['wavelength start'] = lambda_min_um * 1e-6
        adj_src['wavelength stop'] = lambda_max_um * 1e-6

        adjoint_sources[adj_src_idx].append(adj_src)

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


#
# Add device region and create device permittivity
#
design_import = fdtd_hook.addimport()
design_import['name'] = 'design_import'
design_import['x span'] = device_size_lateral_um * 1e-6
design_import['y span'] = device_size_lateral_um * 1e-6
design_import['z max'] = device_vertical_maximum_um * 1e-6
design_import['z min'] = device_vertical_minimum_um * 1e-6

# Left off here
bayer_filter_size_voxels = np.array(
    [device_voxels_lateral, device_voxels_lateral, device_voxels_vertical])
bayer_filter = SonyBayerFilter.SonyBayerFilter(
    bayer_filter_size_voxels,
    [0.0, 1.0],
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

bayer_filter_region = np.zeros(
    (device_voxels_lateral, device_voxels_lateral, device_voxels_vertical, 3))
for x_idx in range(0, device_voxels_lateral):
    for y_idx in range(0, device_voxels_lateral):
        for z_idx in range(0, device_voxels_vertical):
            bayer_filter_region[x_idx, y_idx, z_idx, :] = [
                bayer_filter_region_x[x_idx], bayer_filter_region_y[y_idx], bayer_filter_region_z[z_idx]]

gradient_region_x = 1e-6 * np.linspace(-0.5 * device_size_lateral_um,
                                       0.5 * device_size_lateral_um, device_voxels_simulation_mesh_lateral)
gradient_region_y = 1e-6 * np.linspace(-0.5 * device_size_lateral_um,
                                       0.5 * device_size_lateral_um, device_voxels_simulation_mesh_lateral)
gradient_region_z = 1e-6 * np.linspace(device_vertical_minimum_um,
                                       device_vertical_maximum_um, device_voxels_simulation_mesh_vertical)

#
# Disable all sources in the simulation, so that we can selectively turn single sources on at a time
#


def disable_all_sources():
    lumapi.evalScript(fdtd_hook.handle, 'switchtolayout;')

    for xy_idx in range(0, 2):
        fdtd_hook.select(forward_sources[xy_idx]['name'])
        fdtd_hook.set('enabled', 0)

    for adj_src_idx in range(0, num_adjoint_sources):
        for xy_idx in range(0, 2):
            fdtd_hook.select(adjoint_sources[adj_src_idx][xy_idx]['name'])
            fdtd_hook.set('enabled', 0)


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
quadrant_transmission_data = {}

figure_of_merit_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
figure_of_merit_by_wl_evolution = np.zeros(
    (num_epochs, num_iterations_per_epoch, 4))
step_size_evolution = np.zeros((num_epochs, num_iterations_per_epoch))
average_design_variable_change_evolution = np.zeros(
    (num_epochs, num_iterations_per_epoch))
max_design_variable_change_evolution = np.zeros(
    (num_epochs, num_iterations_per_epoch))

transmission_by_focal_spot_evolution = np.zeros(
    (num_epochs, num_iterations_per_epoch, 4))
transmission_by_wl_evolution = np.zeros(
    (num_epochs, num_iterations_per_epoch, num_design_frequency_points))
intensity_fom_by_wavelength_evolution = np.zeros(
    (num_epochs, num_iterations_per_epoch, num_design_frequency_points))

step_size_start = 1.

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
            if queue_in.qsize() > 0:
                small_queue.put(queue_in.get())

        run_jobs_inner(small_queue)


def run_jobs_inner(queue_in):
    processes = []
    job_idx = 0
    while not queue_in.empty():
        get_job_path = queue_in.get()

        process = subprocess.Popen(
            [
                '/home/gdrobert/Develompent/adjoint_lumerical/inverse_design/run_proc.sh',
                cluster_hostnames[job_idx],
                get_job_path
            ]
        )
        processes.append(process)

        job_idx += 1

    completed_jobs = [0 for i in range(0, len(processes))]
    while np.sum(completed_jobs) < len(processes):
        for job_idx in range(0, len(processes)):
            if completed_jobs[job_idx] == 0:

                poll_result = processes[job_idx].poll()
                if not(poll_result is None):
                    completed_jobs[job_idx] = 1

        time.sleep(1)


dispersion_model = DevicePermittivityModel()

restart = False

if restart:
    cur_design_variable = np.load(
        projects_directory_location + "/cur_design_variable.npy")
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
        projects_directory_location + "/figure_of_merit_by_wl.npy")
    transmission_by_focal_spot_evolution = np.load(
        projects_directory_location + "/transmission_by_focal_spot.npy")
    transmission_by_wl_evolution = np.load(
        projects_directory_location + "/transmission_by_wl.npy")


#
# Run the optimization
#
start_epoch = 0
for epoch in range(start_epoch, num_epochs):
    bayer_filter.update_filters(epoch)
    bayer_filter.update_permittivity()

    fdtd_hook.switchtolayout()

    start_iter = 0
    if epoch == start_epoch:
        start_iter = 0
        bayer_filter.current_iteration = start_epoch * \
            num_iterations_per_epoch + start_iter
    for iteration in range(start_iter, num_iterations_per_epoch):
        print("Working on epoch " + str(epoch) +
              " and iteration " + str(iteration))

        job_names = {}

        fdtd_hook.switchtolayout()
        cur_density = bayer_filter.get_permittivity()

        #
        # Step 1: Run the forward optimization for both x- and y-polarized plane waves.  Run a different optimization for each color band so
        # that you can input a different index into each of those cubes.  Further note that the polymer slab needs to also change index when
        # the cube index changes.
        #
        for dispersive_range_idx in range(0, num_dispersive_ranges):
            dispersive_max_permittivity = dispersion_model.average_permittivity(
                dispersive_ranges_um[dispersive_range_idx])
            disperesive_max_index = index_from_permittivity(
                dispersive_max_permittivity)

            fdtd_hook.switchtolayout()

            cur_permittivity = min_device_permittivity + \
                (dispersive_max_permittivity - min_device_permittivity) * cur_density
            cur_index = index_from_permittivity(cur_permittivity)

            fdtd_hook.select('design_import')
            fdtd_hook.importnk2(cur_index, bayer_filter_region_x,
                                bayer_filter_region_y, bayer_filter_region_z)

            for xy_idx in range(0, 2):
                disable_all_sources()

                fdtd_hook.select(forward_sources[xy_idx]['name'])
                fdtd_hook.set('enabled', 1)

                job_name = 'forward_job_' + \
                    str(xy_idx) + "_" + str(dispersive_range_idx) + '.fsp'
                fdtd_hook.save(projects_directory_location +
                               "/optimization.fsp")
                job_names[('forward', xy_idx, dispersive_range_idx)
                          ] = add_job(job_name, jobs_queue)

            for adj_src_idx in range(0, num_adjoint_sources):
                for xy_idx in range(0, 2):
                    disable_all_sources()

                    fdtd_hook.select(
                        adjoint_sources[adj_src_idx][xy_idx]['name'])
                    fdtd_hook.set('enabled', 1)

                    job_name = 'adjoint_job_' + \
                        str(adj_src_idx) + '_' + str(xy_idx) + \
                        '_' + str(dispersive_range_idx) + '.fsp'
                    fdtd_hook.save(
                        projects_directory_location + "/optimization.fsp")
                    job_names[('adjoint', adj_src_idx, xy_idx, dispersive_range_idx)] = add_job(
                        job_name, jobs_queue)

        run_jobs(jobs_queue)

        #
        # Step 2: Compute the figure of merit
        #
        for xy_idx in range(0, 2):
            focal_data[xy_names[xy_idx]] = [
                None for idx in range(0, num_focal_spots)]
            quadrant_transmission_data[xy_names[xy_idx]] = [
                None for idx in range(0, num_focal_spots)]

        for dispersive_range_idx in range(0, num_dispersive_ranges):
            for xy_idx in range(0, 2):
                fdtd_hook.load(
                    job_names[('forward', xy_idx, dispersive_range_idx)])

                fwd_e_fields = get_efield(design_efield_monitor['name'])

                if (dispersive_range_idx == 0):

                    forward_e_fields[xy_names[xy_idx]] = fwd_e_fields
                else:
                    get_dispersive_range_buckets = dispersive_range_to_adjoint_src_map[
                        dispersive_range_idx]

                    for bucket_idx in range(0, len(get_dispersive_range_buckets)):

                        get_wavelength_bucket = spectral_focal_plane_map[
                            get_dispersive_range_buckets[bucket_idx]]

                        for spectral_idx in range(get_wavelength_bucket[0], get_wavelength_bucket[1]):
                            forward_e_fields[xy_names[xy_idx]][:, :, :, :,
                                                               spectral_idx] = fwd_e_fields[:, :, :, :, spectral_idx]

                for adj_src_idx in dispersive_range_to_adjoint_src_map[dispersive_range_idx]:
                    focal_data[xy_names[xy_idx]][adj_src_idx] = get_efield(
                        focal_monitors[adj_src_idx]['name'])

                    quadrant_transmission_data[xy_names[xy_idx]][adj_src_idx] = get_transmission_magnitude(
                        transmission_monitors[adj_src_idx]['name'])

        figure_of_merit_per_focal_spot = []
        transmission_per_quadrant = []

        transmission_by_wavelength = np.zeros(num_design_frequency_points)
        intensity_fom_by_wavelength = np.zeros(num_design_frequency_points)

        for focal_idx in range(0, num_focal_spots):
            compute_fom = 0
            compute_transmission_fom = 0

            polarizations = polarizations_focal_plane_map[focal_idx]

            for polarization_idx in range(0, len(polarizations)):
                get_focal_data = focal_data[polarizations[polarization_idx]]
                get_quad_transmission_data = quadrant_transmission_data[
                    polarizations[polarization_idx]]

                max_intensity_weighting = max_intensity_by_wavelength[spectral_focal_plane_map[
                    focal_idx][0]: spectral_focal_plane_map[focal_idx][1]: 1]
                total_weighting = max_intensity_weighting / \
                    weight_focal_plane_map_performance_weighting[focal_idx]
                one_over_weight_focal_plane_map_performance_weighting = 1. / \
                    weight_focal_plane_map_performance_weighting[focal_idx]

                for spectral_idx in range(0, total_weighting.shape[0]):

                    weight_spectral_rejection = 1.0

                    if do_rejection:
                        weight_spectral_rejection = spectral_focal_plane_map_directional_weights[
                            focal_idx][spectral_idx]

                    compute_transmission_fom += weight_spectral_rejection * \
                        get_quad_transmission_data[focal_idx][spectral_focal_plane_map[focal_idx]
                                                              [0] + spectral_idx] / one_over_weight_focal_plane_map_performance_weighting
                    compute_fom += weight_spectral_rejection * np.sum(
                        (
                            np.abs(get_focal_data[focal_idx][:, 0, 0, 0, spectral_focal_plane_map[focal_idx][0] + spectral_idx])**2 /
                            total_weighting[spectral_idx]
                        )
                    )

                    intensity_fom_by_wavelength[spectral_focal_plane_map[focal_idx][0] + spectral_idx] += weight_spectral_rejection * np.sum(
                        (
                            np.abs(get_focal_data[focal_idx][:, 0, 0, 0, spectral_focal_plane_map[focal_idx][0] + spectral_idx])**2 /
                            total_weighting[spectral_idx]
                        )
                    )
                    transmission_by_wavelength[spectral_focal_plane_map[focal_idx][0] + spectral_idx] += weight_spectral_rejection * \
                        get_quad_transmission_data[focal_idx][spectral_focal_plane_map[focal_idx]
                                                              [0] + spectral_idx] / one_over_weight_focal_plane_map_performance_weighting

            figure_of_merit_per_focal_spot.append(compute_fom)
            transmission_per_quadrant.append(compute_transmission_fom)

        if do_rejection:
            for idx in range(0, len(figure_of_merit_per_focal_spot)):
                figure_of_merit_per_focal_spot[idx] = softplus(
                    figure_of_merit_per_focal_spot[idx])
            for idx in range(0, len(transmission_per_quadrant)):
                transmission_per_quadrant[idx] = softplus(
                    transmission_per_quadrant[idx])

            for idx in range(0, len(intensity_fom_by_wavelength)):
                intensity_fom_by_wavelength[idx] = softplus(
                    intensity_fom_by_wavelength[idx])
            for idx in range(0, len(transmission_by_wavelength)):
                transmission_by_wavelength[idx] = softplus(
                    transmission_by_wavelength[idx])

        figure_of_merit_per_focal_spot = np.array(
            figure_of_merit_per_focal_spot)
        transmission_per_quadrant = np.array(transmission_per_quadrant)

        performance_weighting = (
            2. / num_focal_spots) - figure_of_merit_per_focal_spot**2 / np.sum(figure_of_merit_per_focal_spot**2)
        if weight_by_quadrant_transmission:
            performance_weighting = (
                2. / num_focal_spots) - transmission_per_quadrant**2 / np.sum(transmission_per_quadrant**2)

        if weight_by_individual_wavelength:
            performance_weighting = (2. / num_design_frequency_points) - \
                intensity_fom_by_wavelength**2 / \
                np.sum(intensity_fom_by_wavelength**2)
            if weight_by_quadrant_transmission:
                performance_weighting = (2. / num_design_frequency_points) - \
                    transmission_by_wavelength**2 / \
                    np.sum(transmission_by_wavelength**2)

        if np.min(performance_weighting) < 0:
            performance_weighting -= np.min(performance_weighting)
            performance_weighting /= np.sum(performance_weighting)

        figure_of_merit = np.sum(figure_of_merit_per_focal_spot)
        figure_of_merit_evolution[epoch, iteration] = figure_of_merit
        figure_of_merit_by_wl_evolution[epoch,
                                        iteration] = figure_of_merit_per_focal_spot

        transmission_by_focal_spot_evolution[epoch,
                                             iteration] = transmission_per_quadrant
        transmission_by_wl_evolution[epoch,
                                     iteration] = transmission_by_wavelength
        intensity_fom_by_wavelength_evolution[epoch,
                                              iteration] = intensity_fom_by_wavelength

        np.save(projects_directory_location +
                "/figure_of_merit.npy", figure_of_merit_evolution)
        np.save(projects_directory_location +
                "/figure_of_merit_by_wl.npy", figure_of_merit_by_wl_evolution)
        np.save(projects_directory_location + "/transmission_by_focal_spot.npy",
                transmission_by_focal_spot_evolution)
        np.save(projects_directory_location +
                "/transmission_by_wl.npy", transmission_by_wl_evolution)
        np.save(projects_directory_location + "/intensity_fom_by_wavelength.npy",
                intensity_fom_by_wavelength_evolution)

        #
        # Step 3: Run all the adjoint optimizations for both x- and y-polarized adjoint sources and use the results to compute the
        # gradients for x- and y-polarized forward sources.
        #
        cur_permittivity_shape = cur_permittivity.shape
        field_shape = [device_voxels_simulation_mesh_lateral,
                       device_voxels_simulation_mesh_lateral, device_voxels_simulation_mesh_vertical]

        xy_polarized_gradients = [np.zeros(
            field_shape, dtype=np.complex), np.zeros(field_shape, dtype=np.complex)]

        for adj_src_idx in range(0, num_adjoint_sources):
            polarizations = polarizations_focal_plane_map[adj_src_idx]
            spectral_indices = spectral_focal_plane_map[adj_src_idx]

            gradient_performance_weight = 0

            if not weight_by_individual_wavelength:
                gradient_performance_weight = performance_weighting[adj_src_idx]

                if do_rejection:
                    if weight_by_quadrant_transmission:
                        gradient_performance_weight *= softplus_prime(
                            transmission_per_quadrant[adj_src_idx])
                    else:
                        gradient_performance_weight *= softplus_prime(
                            figure_of_merit_per_focal_spot[adj_src_idx])

            lookup_dispersive_range_idx = adjoint_src_to_dispersive_range_map[adj_src_idx]

            adjoint_e_fields = []
            for xy_idx in range(0, 2):
                #
                # This is ok because we are only going to be using the spectral idx corresponding to this range in the summation below
                #
                fdtd_hook.load(
                    job_names[('adjoint', adj_src_idx, xy_idx, lookup_dispersive_range_idx)])

                adjoint_e_fields.append(
                    get_efield(design_efield_monitor['name']))

            for pol_idx in range(0, len(polarizations)):
                pol_name = polarizations[pol_idx]
                get_focal_data = focal_data[pol_name]
                pol_name_to_idx = polarization_name_to_idx[pol_name]

                for xy_idx in range(0, 2):
                    source_weight = np.squeeze(np.conj(
                        get_focal_data[adj_src_idx][xy_idx, 0, 0, 0, spectral_indices[0]: spectral_indices[1]: 1]))

                    #
                    # todo: do you need to change this if you weight by transmission? the figure of merit still has this weight, the transmission is just
                    # guiding additional weights on top of this
                    #
                    max_intensity_weighting = max_intensity_by_wavelength[
                        spectral_indices[0]: spectral_indices[1]: 1]

                    total_weighting = max_intensity_weighting / \
                        weight_focal_plane_map[focal_idx]

                    #
                    # We need to properly account here for the current real and imaginary index
                    #
                    dispersive_max_permittivity = dispersion_model.average_permittivity(
                        dispersive_ranges_um[lookup_dispersive_range_idx])
                    delta_real_permittivity = np.real(
                        dispersive_max_permittivity - min_device_permittivity)
                    delta_imag_permittivity = np.imag(
                        dispersive_max_permittivity - min_device_permittivity)

                    for spectral_idx in range(0, len(source_weight)):
                        if weight_by_individual_wavelength:
                            gradient_performance_weight = performance_weighting[
                                spectral_indices[0] + spectral_idx]

                            if do_rejection:
                                if weight_by_quadrant_transmission:
                                    gradient_performance_weight *= softplus_prime(
                                        transmission_by_wavelength[adj_src_idx])
                                else:
                                    gradient_performance_weight *= softplus_prime(
                                        intensity_fom_by_wavelength[adj_src_idx])

                        if do_rejection:
                            gradient_performance_weight *= spectral_focal_plane_map_directional_weights[
                                adj_src_idx][spectral_idx]

                        gradient_component = np.sum(
                            (source_weight[spectral_idx] * gradient_performance_weight / total_weighting[spectral_idx]) *
                            adjoint_e_fields[xy_idx][:, :, :, :, spectral_indices[0] + spectral_idx] *
                            forward_e_fields[pol_name][:, :, :, :,
                                                       spectral_indices[0] + spectral_idx],
                            axis=0)

                        real_part_gradient = np.real(gradient_component)
                        imag_part_gradient = -np.imag(gradient_component)

                        get_grad_density = delta_real_permittivity * real_part_gradient + \
                            delta_imag_permittivity * imag_part_gradient

                        xy_polarized_gradients[pol_name_to_idx] += get_grad_density

        #
        # Step 4: Step the design variable.
        #
        device_gradient = 2 * \
            (xy_polarized_gradients[0] + xy_polarized_gradients[1])

        device_gradient_interpolated = interpolate.interpn(
            (gradient_region_x, gradient_region_y, gradient_region_z), device_gradient, bayer_filter_region, method='linear')

        device_gradient = device_gradient_interpolated.copy()
        design_gradient = bayer_filter.backpropagate(device_gradient)

        max_change_design = epoch_start_design_change_max
        min_change_design = epoch_start_design_change_min

        if num_iterations_per_epoch > 1:

            max_change_design = (
                epoch_end_design_change_max +
                (num_iterations_per_epoch - 1 - iteration) *
                (epoch_range_design_change_max / (num_iterations_per_epoch - 1))
            )

            min_change_design = (
                epoch_end_design_change_min +
                (num_iterations_per_epoch - 1 - iteration) *
                (epoch_range_design_change_min / (num_iterations_per_epoch - 1))
            )

        cur_design_variable = bayer_filter.get_design_variable()

        if use_fixed_step_size:
            step_size = fixed_step_size
        else:
            step_size = step_size_start

            check_last = False
            last = 0

            while True:
                proposed_design_variable = cur_design_variable + step_size * design_gradient
                proposed_design_variable = np.maximum(
                    np.minimum(
                        proposed_design_variable,
                        1.0),
                    0.0)

                difference = np.abs(
                    proposed_design_variable - cur_design_variable)
                max_difference = np.max(difference)

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

        last_design_variable = cur_design_variable.copy()
        bayer_filter.step(-device_gradient, step_size)
        cur_design_variable = bayer_filter.get_design_variable()
        cur_design = bayer_filter.get_permittivity()

        average_design_variable_change = np.mean(
            np.abs(cur_design_variable - last_design_variable))
        max_design_variable_change = np.max(
            np.abs(cur_design_variable - last_design_variable))

        step_size_evolution[epoch][iteration] = step_size
        average_design_variable_change_evolution[epoch][iteration] = average_design_variable_change
        max_design_variable_change_evolution[epoch][iteration] = max_design_variable_change

        np.save(projects_directory_location +
                '/device_gradient.npy', device_gradient)
        np.save(projects_directory_location +
                '/design_gradient.npy', design_gradient)
        np.save(projects_directory_location +
                "/step_size_evolution.npy", step_size_evolution)
        np.save(projects_directory_location + "/average_design_change_evolution.npy",
                average_design_variable_change_evolution)
        np.save(projects_directory_location + "/max_design_change_evolution.npy",
                max_design_variable_change_evolution)
        np.save(projects_directory_location +
                "/cur_design_variable.npy", cur_design_variable)
        np.save(projects_directory_location + "/cur_design.npy", cur_design)
        np.save(projects_directory_location + "/cur_design_variable_epoch_" +
                str(epoch) + ".npy", cur_design_variable)
