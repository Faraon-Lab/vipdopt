import sys
import os
import copy
import numpy as np
import logging
import time

# Custom Classes and Imports
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())
import configs.yaml_to_params as cfg
# Gets all parameters from config file - store all those variables within the namespace. Editing cfg edits it for all modules accessing it
# See https://docs.python.org/3/faq/programming.html#how-do-i-share-global-variables-across-modules
from utils import utility

# NOTE: It seems most expedient to straight up define everything in this class, i.e. let it be a "script" for not just the Lumerical
# objects, but also the forward sources, adjoint sources, how each FoM is set up etc.

def construct_sim_objects():
	'''Create dictionary to duplicate FDTD SimObjects as dictionaries for saving out'''
	# This dictionary captures the states of all FDTD objects in the simulation, and has the advantage that it can be saved out or exported.
	# The dictionary will be our master record, and is continually updated for every change in FDTD.
	sim_objects = {}
	sim_objects['SIM_INFO'] = {
				'name': 'optimization',
				'path': '',
				'simulator_name': 'LumericalFDTD',
				'coordinates': None
	}

	#! This script should describe all objects without any sort of partitioning for E-field stitching later on. 
	#! The Simulator class will be performing partitioning.

	# Set up the FDTD region and mesh.
	# We are using dict-like access: see https://support.lumerical.com/hc/en-us/articles/360041873053-Session-management-Python-API
	# The line fdtd_objects['FDTD'] = fdtd means that both sides of the equation point to the same object.
	# See https://stackoverflow.com/a/10205681 and https://nedbatchelder.com/text/names.html

	fdtd = {}
	fdtd['name'] = 'FDTD'
	fdtd['type'] = 'FDTD'
	fdtd['dimension'] = '2D'
	fdtd['x span'] = cfg.pv.fdtd_region_size_lateral_um * 1e-6
	# fdtd['y span'] = cfg.pv.fdtd_region_size_lateral_um * 1e-6
	fdtd['y max'] = cfg.pv.fdtd_region_maximum_vertical_um * 1e-6
	fdtd['y min'] = cfg.pv.fdtd_region_minimum_vertical_um * 1e-6
	# fdtd['z max'] = cfg.pv.fdtd_region_maximum_vertical_um * 1e-6
	# fdtd['z min'] = cfg.pv.fdtd_region_minimum_vertical_um * 1e-6
	fdtd['simulation time'] = cfg.cv.fdtd_simulation_time_fs * 1e-15
	fdtd['index'] = cfg.cv.background_index
	# fdtd = fdtd_update_object(fdtd_hook, fdtd, create_object=True)
	sim_objects['FDTD'] = fdtd

	# Add a Gaussian wave forward source at angled incidence
	# (Deprecated) Add a TFSF plane wave forward source at normal incidence
	forward_sources = []

	for xy_idx in range(0, 2):
		if cfg.cv.use_gaussian_sources:
			
			forward_src = {}
			forward_src['name'] = 'forward_src_' + cfg.cv.xy_names[xy_idx]
			forward_src['type'] = 'GaussianSource'			
			forward_src['attached_monitor'] = xy_idx
			forward_src['injection axis'] = 'y-axis'
			forward_src['angle theta'] = cfg.pv.source_angle_theta_deg
			forward_src['angle phi'] = cfg.cv.source_angle_phi_deg
			forward_src['polarization angle'] = cfg.cv.xy_phi_rotations[xy_idx]
			forward_src['direction'] = 'Backward'
			forward_src['x span'] = 2 * cfg.pv.fdtd_region_size_lateral_um * 1e-6
			forward_src['y'] = cfg.pv.src_maximum_vertical_um * 1e-6
			# forward_src['y span'] = 2 * cfg.pv.fdtd_region_size_lateral_um * 1e-6
			# forward_src['z'] = cfg.pv.src_maximum_vertical_um * 1e-6

			# forward_src['angle theta'], shift_x_center = snellRefrOffset(source_angle_theta_deg, objects_above_device)	# this code takes substrates and objects above device into account
			shift_x_center = np.abs( cfg.pv.device_vertical_maximum_um - cfg.pv.src_maximum_vertical_um ) * np.tan( cfg.pv.source_angle_theta_rad )
			forward_src['x'] = shift_x_center * 1e-6

			forward_src['wavelength start'] = cfg.cv.lambda_min_um * 1e-6
			forward_src['wavelength stop'] = cfg.cv.lambda_max_um * 1e-6

			forward_src['waist radius w0'] = cfg.pv.gaussian_waist_radius_um * 1e-6
			forward_src['distance from waist'] = ( cfg.pv.device_vertical_maximum_um - cfg.pv.src_maximum_vertical_um ) * 1e-6

		else:
			forward_src = {}
			forward_src['name'] = 'forward_src_' + cfg.cv.xy_names[xy_idx]
			forward_src['type'] = 'TFSFSource'
			# forward_src['angle phi'] = 0
			forward_src['polarization angle'] = cfg.cv.xy_phi_rotations[xy_idx]
			forward_src['direction'] = 'Backward'
			forward_src['x span'] = cfg.pv.lateral_aperture_um * 1e-6
			forward_src['y span'] = cfg.pv.lateral_aperture_um * 1e-6
			forward_src['z max'] = cfg.pv.src_maximum_vertical_um * 1e-6
			forward_src['z min'] = cfg.cv.src_minimum_vertical_um * 1e-6
			forward_src['wavelength start'] = cfg.cv.lambda_min_um * 1e-6
			forward_src['wavelength stop'] = cfg.cv.lambda_max_um * 1e-6

		#forward_src = fdtd_update_object(fdtd_hook, forward_src, create_object=True)
		forward_sources.append(forward_src)
		sim_objects[forward_src['name']] = forward_src
	sim_objects['forward_sources'] = forward_sources


	# Place dipole adjoint sources at the focal plane that can ring in both x-axis and y-axis
	adjoint_sources = []

	for adj_src_idx in range(0, cfg.pv.num_adjoint_sources):
		# adjoint_sources.append([])
		for xy_idx in range(0, 2):
			adj_src = {}
			adj_src['name'] = f'adj_src_{adj_src_idx}' + cfg.cv.xy_names[xy_idx]
			adj_src['type'] = 'DipoleSource'
			adj_src['attached_monitor'] = adj_src_idx
			adj_src['x'] = cfg.pv.adjoint_x_positions_um[adj_src_idx] * 1e-6
			# adj_src['y'] = cfg.pv.adjoint_y_positions_um[adj_src_idx] * 1e-6
			adj_src['y'] = cfg.pv.adjoint_vertical_um * 1e-6
			# adj_src['z'] = cfg.pv.adjoint_vertical_um * 1e-6
			adj_src['theta'] = cfg.cv.xy_adjtheta_rotations[xy_idx]	# NOTE: It says phi_rotations but it needs to be theta in 2D
			adj_src['phi'] = 0
			# todo: Assign an explicit polarization to the adjoint sources
			adj_src['wavelength start'] = cfg.cv.lambda_min_um * 1e-6
			adj_src['wavelength stop'] = cfg.cv.lambda_max_um * 1e-6
			
			# adj_src = fdtd_update_object(fdtd_hook, adj_src, create_object=True)
			sim_objects[adj_src['name']] = adj_src
			# adjoint_sources[adj_src_idx].append(adj_src)
			adjoint_sources.append(adj_src)
	sim_objects['adjoint_sources'] = adjoint_sources
	#! todo: change this around. numpy array????

	# Set up adjoint point monitors to get electric field strength at focus spots.  This will allow us to
	# compute the figure of merit as well as weight the adjoint simulations properly in calculation of the
	# gradient.
	adjoint_monitors = []

	for adj_src in range(0, cfg.pv.num_adjoint_sources):
		adjoint_monitor = {}
		adjoint_monitor['name'] = f'focal_monitor_{adj_src}'
		adjoint_monitor['type'] = 'DFTMonitor'
		adjoint_monitor['power_monitor'] = True
		adjoint_monitor['monitor type'] = 'point'
		adjoint_monitor['x'] = cfg.pv.adjoint_x_positions_um[adj_src] * 1e-6
		# adjoint_monitor['y'] = cfg.pv.adjoint_y_positions_um[adj_src] * 1e-6
		adjoint_monitor['y'] = cfg.pv.adjoint_vertical_um * 1e-6
		# adjoint_monitor['z'] = cfg.pv.adjoint_vertical_um * 1e-6
		adjoint_monitor['override global monitor settings'] = 1
		adjoint_monitor['use wavelength spacing'] = 1
		adjoint_monitor['use source limits'] = 1
		adjoint_monitor['frequency points'] = cfg.pv.num_design_frequency_points
		
		# focal_monitor = fdtd_update_object(fdtd_hook, focal_monitor, create_object=True)
		sim_objects[adjoint_monitor['name']] = adjoint_monitor
		adjoint_monitors.append(adjoint_monitor)
	sim_objects['adjoint_monitors'] = adjoint_monitors

	transmission_monitors = []

	for adj_src in range(0, cfg.pv.num_adjoint_sources):
		transmission_monitor = {}
		transmission_monitor['name'] = f'transmission_monitor_{adj_src}'
		transmission_monitor['type'] = 'DFTMonitor'
		transmission_monitor['power_monitor'] = True
		transmission_monitor['monitor type'] = 'Linear X'
		transmission_monitor['x'] = cfg.pv.adjoint_x_positions_um[adj_src] * 1e-6
		transmission_monitor['x span'] = 0.5 * cfg.cv.device_size_lateral_um * 1e-6
		# transmission_monitor['y'] = cfg.pv.adjoint_y_positions_um[adj_src] * 1e-6
		# transmission_monitor['y span'] = 0.5 * cfg.cv.device_size_lateral_um * 1e-6
		transmission_monitor['y'] = cfg.pv.adjoint_vertical_um * 1e-6
		# transmission_monitor['z'] = cfg.pv.adjoint_vertical_um * 1e-6
		transmission_monitor['override global monitor settings'] = 1
		transmission_monitor['use wavelength spacing'] = 1
		transmission_monitor['use source limits'] = 1
		transmission_monitor['frequency points'] = cfg.pv.num_design_frequency_points

		# transmission_monitor = fdtd_update_object(fdtd_hook, transmission_monitor, create_object=True)
		sim_objects[transmission_monitor['name']] = transmission_monitor
		transmission_monitors.append(transmission_monitor)
	sim_objects['transmission_monitors'] = transmission_monitors

	transmission_monitor_focal = {}
	transmission_monitor_focal['name'] = 'transmission_focal_monitor_'
	transmission_monitor_focal['type'] = 'DFTMonitor'
	transmission_monitor_focal['power_monitor'] = True
	transmission_monitor_focal['monitor type'] = 'Linear X'
	transmission_monitor_focal['x'] = 0 * 1e-6
	transmission_monitor_focal['x span'] = cfg.cv.device_size_lateral_um * 1e-6
	# transmission_monitor_focal['y'] = 0 * 1e-6
	# transmission_monitor_focal['y span'] = cfg.cv.device_size_lateral_um * 1e-6
	transmission_monitor_focal['y'] = cfg.pv.adjoint_vertical_um * 1e-6
	# transmission_monitor_focal['z'] = cfg.pv.adjoint_vertical_um * 1e-6
	transmission_monitor_focal['override global monitor settings'] = 1
	transmission_monitor_focal['use wavelength spacing'] = 1
	transmission_monitor_focal['use source limits'] = 1
	transmission_monitor_focal['frequency points'] = cfg.pv.num_design_frequency_points

	# transmission_monitor_focal = fdtd_update_object(fdtd_hook, transmission_monitor_focal, create_object=True)
	sim_objects['transmission_monitor_focal'] = transmission_monitor_focal

	# TODO: PDAF Sources

	# Install Aperture that blocks off source
	if cfg.cv.use_source_aperture:
		source_block = {}
		source_block['name'] = 'PEC_screen'
		source_block['type'] = 'Rectangle'
		source_block['x'] = 0 * 1e-6
		source_block['x span'] = 1.1*4/3*1.2 * cfg.cv.device_size_lateral_um * 1e-6
		# source_block['y'] = 0 * 1e-6
		# source_block['y span'] = 1.1*4/3*1.2 * cfg.cv.device_size_lateral_um * 1e-6
		source_block['y min'] = (cfg.pv.device_vertical_maximum_um + 3 * cfg.cv.mesh_spacing_um) * 1e-6
		source_block['y max'] = ( cfg.pv.device_vertical_maximum_um + 3 * cfg.cv.mesh_spacing_um + cfg.pv.pec_aperture_thickness_um ) * 1e-6
		# source_block['z min'] = (cfg.pv.device_vertical_maximum_um + 3 * cfg.cv.mesh_spacing_um) * 1e-6
		# source_block['z max'] = ( cfg.pv.device_vertical_maximum_um + 3 * cfg.cv.mesh_spacing_um + cfg.pv.pec_aperture_thickness_um ) * 1e-6
		source_block['material'] = 'PEC (Perfect Electrical Conductor)'
	
		# source_block = fdtd_update_object(fdtd_hook, source_block, create_object=True)
		sim_objects['source_block'] = source_block
		
		source_aperture = {}
		source_aperture['name'] = 'source_aperture'
		source_aperture['type'] = 'Rectangle'
		source_aperture['x'] = 0 * 1e-6
		source_aperture['x span'] = cfg.cv.device_size_lateral_um * 1e-6
		# source_aperture['y'] = 0 * 1e-6
		# source_aperture['y span'] = cfg.cv.device_size_lateral_um * 1e-6
		source_aperture['y min'] = (cfg.pv.device_vertical_maximum_um + 3 * cfg.cv.mesh_spacing_um) * 1e-6
		source_aperture['y max'] = (cfg.pv.device_vertical_maximum_um + 3 * cfg.cv.mesh_spacing_um + cfg.pv.pec_aperture_thickness_um) * 1e-6
		# source_aperture['z min'] = (cfg.pv.device_vertical_maximum_um + 3 * cfg.cv.mesh_spacing_um) * 1e-6
		# source_aperture['z max'] = (cfg.pv.device_vertical_maximum_um + 3 * cfg.cv.mesh_spacing_um + cfg.pv.pec_aperture_thickness_um) * 1e-6
		source_aperture['index'] = cfg.cv.background_index
	
		# source_aperture = fdtd_update_object(fdtd_hook, source_aperture, create_object=True)
		sim_objects['source_aperture'] = source_aperture


	# # Set up sidewalls on the side to try and attenuate crosstalk
	# device_sidewalls = []

	# for dev_sidewall_idx in range(0, cfg.cv.num_sidewalls):
	# 	device_sidewall = {}
	# 	device_sidewall['name'] = 'sidewall_' + str(dev_sidewall_idx)
	# 	device_sidewall['type'] = 'Rectangle'
	# 	device_sidewall['x'] = cfg.pv.sidewall_x_positions_um[dev_sidewall_idx] * 1e-6
	# 	device_sidewall['x span'] = cfg.pv.sidewall_xspan_positions_um[dev_sidewall_idx] * 1e-6
	# 	device_sidewall['y'] = cfg.pv.sidewall_y_positions_um[dev_sidewall_idx] * 1e-6
	# 	device_sidewall['y span'] = cfg.pv.sidewall_yspan_positions_um[dev_sidewall_idx] * 1e-6
	# 	if cfg.cv.sidewall_extend_focalplane:
	# 		device_sidewall['z min'] = cfg.pv.adjoint_vertical_um * 1e-6
	# 	else:   
	# 		device_sidewall['z min'] = cfg.cv.sidewall_vertical_minimum_um * 1e-6
	# 	device_sidewall['z max'] = cfg.pv.device_vertical_maximum_um * 1e-6
	# 	device_sidewall['material'] = cfg.cv.sidewall_material
	
	# 	# device_sidewall = fdtd_update_object(fdtd_hook, device_sidewall, create_object=True)
	# 	sim_objects[device_sidewall['name']] = device_sidewall
	# 	device_sidewalls.append(device_sidewall)
	# sim_objects['device_sidewalls'] = device_sidewalls


	# # Apply finer mesh regions restricted to sidewalls
	# device_sidewall_meshes = []

	# for dev_sidewall_idx in range(0, cfg.cv.num_sidewalls):
	# 	device_sidewall_mesh = {}
	# 	device_sidewall_mesh['name'] = f'mesh_sidewall_{dev_sidewall_idx}'
	# 	device_sidewall_mesh['type'] = 'Mesh'
	# 	device_sidewall_mesh['set maximum mesh step'] = 1
	# 	device_sidewall_mesh['override x mesh'] = 0
	# 	device_sidewall_mesh['override y mesh'] = 0
	# 	device_sidewall_mesh['override z mesh'] = 0
	# 	device_sidewall_mesh['based on a structure'] = 1
	# 	device_sidewall_mesh['structure'] = device_sidewalls[dev_sidewall_idx]['name']
	# 	if device_sidewalls[dev_sidewall_idx]['x span'] < device_sidewalls[dev_sidewall_idx]['y span']:
	# 		device_sidewall_mesh['override x mesh'] = 1
	# 		device_sidewall_mesh['dx'] = cfg.cv.mesh_spacing_um * 1e-6 #(sidewall_thickness_um / 5) * 1e-6
	# 	else:
	# 		device_sidewall_mesh['override y mesh'] = 1
	# 		device_sidewall_mesh['dy'] = cfg.cv.mesh_spacing_um * 1e-6 #(sidewall_thickness_um / 5) * 1e-6
		
	# 	# device_sidewall_mesh = fdtd_update_object(fdtd_hook, device_sidewall_mesh, create_object=True)
	# 	sim_objects[device_sidewall_mesh['name']] = device_sidewall_mesh
	# 	device_sidewall_meshes.append(device_sidewall_mesh)
	# sim_objects['device_sidewall_meshes'] = device_sidewall_meshes

	return sim_objects


def construct_device_regions():
	'''Create dictionary to duplicate device regions as dictionaries for saving out'''
	# This dictionary captures the states of all FDTD objects in the simulation, and has the advantage that it can be saved out or exported.
	# The dictionary will be our master record, and is continually updated for every change in FDTD.
	device_regions = {}
	device_regions['DEVICE_INFO'] = {
				'name': 'SonyBayerFilter',
				'path': '',
				'coordinates': {'x': np.linspace(-0.5 * cfg.pv.device_size_lateral_bordered_um, 0.5 * cfg.pv.device_size_lateral_bordered_um, cfg.pv.device_voxels_lateral_bordered),
								# 'y': np.linspace(-0.5 * cfg.pv.device_size_lateral_bordered_um, 0.5 * cfg.pv.device_size_lateral_bordered_um, cfg.pv.device_voxels_lateral_bordered),
								# 'z': np.linspace(cfg.cv.device_vertical_minimum_um, cfg.pv.device_vertical_maximum_um, cfg.pv.device_voxels_vertical)
								'y': np.linspace(cfg.cv.device_vertical_minimum_um, cfg.pv.device_vertical_maximum_um, cfg.pv.device_voxels_vertical),
								# 'z': np.linspace(-3 * cfg.cv.mesh_spacing_um * 1e-6, 3 * cfg.cv.mesh_spacing_um * 1e-6, 6)
								'z': np.linspace(-0.5 * cfg.cv.mesh_spacing_um * 1e-6, 0.5 * cfg.cv.mesh_spacing_um * 1e-6, 3)
							},					# Coordinates of each voxel can be regular or irregular. Gives the most freedom for feature generation down the road
				# 'size_voxels': np.array( [cfg.pv.device_voxels_lateral_bordered, cfg.pv.device_voxels_lateral_bordered, cfg.pv.device_voxels_vertical] ),
				'size_voxels': np.array( [cfg.pv.device_voxels_lateral_bordered, cfg.pv.device_voxels_vertical, 3] ),
				'feature_dimensions': 1,
				'permittivity_constraints': [cfg.pv.min_device_permittivity, cfg.pv.max_device_permittivity]
		# other_arguments
	}

	#! This script should describe all objects without any sort of partitioning for E-field stitching later on. 
	#! The Simulator class will be performing partitioning.

	# Add device region and create device permittivity
	design_import = {}
	design_import['name'] = 'design_import'
	design_import['type'] = 'Import'
	design_import['dev_id'] = 0				# ID keeps track of which device this belongs to.
	design_import['x span'] = cfg.pv.device_size_lateral_bordered_um * 1e-6
	# design_import['y span'] = cfg.pv.device_size_lateral_bordered_um * 1e-6
	design_import['y max'] = cfg.pv.device_vertical_maximum_um * 1e-6
	design_import['y min'] = cfg.cv.device_vertical_minimum_um * 1e-6
	# design_import['z max'] = cfg.pv.device_vertical_maximum_um * 1e-6
	# design_import['z min'] = cfg.cv.device_vertical_minimum_um * 1e-6

	# design_import = fdtd_update_object(fdtd_hook, design_import, create_object=True)
	device_regions['design_import'] = design_import

	device_mesh = {}
	device_mesh['name'] = 'device_mesh'
	device_mesh['type'] = 'Mesh'
	device_mesh['dev_id'] = 0
	device_mesh['x'] = 0
	device_mesh['x span'] = cfg.pv.fdtd_region_size_lateral_um * 1e-6
	# device_mesh['y'] = 0
	# device_mesh['y span'] = cfg.pv.fdtd_region_size_lateral_um * 1e-6
	device_mesh['y max'] = ( cfg.pv.device_vertical_maximum_um + 0.5 ) * 1e-6
	device_mesh['y min'] = ( cfg.cv.device_vertical_minimum_um - 0.5 ) * 1e-6
	# device_mesh['z max'] = ( cfg.pv.device_vertical_maximum_um + 0.5 ) * 1e-6
	# device_mesh['z min'] = ( cfg.cv.device_vertical_minimum_um - 0.5 ) * 1e-6
	device_mesh['dx'] = cfg.cv.mesh_spacing_um * 1e-6
	device_mesh['dy'] = cfg.cv.mesh_spacing_um * 1e-6
	device_mesh['dz'] = cfg.cv.mesh_spacing_um * 1e-6
 
	# device_mesh = fdtd_update_object(fdtd_hook, device_mesh, create_object=True)
	device_regions['device_mesh'] = device_mesh

	# Add device index monitor
	design_index_monitor = {}
	design_index_monitor['name'] = 'design_index_monitor'
	design_index_monitor['type'] = 'IndexMonitor'
	design_index_monitor['monitor type'] = '2D Z-Normal'
	design_index_monitor['dev_id'] = 0
	design_index_monitor['x span'] = cfg.pv.device_size_lateral_bordered_um * 1e-6
	# design_index_monitor['y span'] = cfg.pv.device_size_lateral_bordered_um * 1e-6
	design_index_monitor['y max'] = cfg.pv.device_vertical_maximum_um * 1e-6
	design_index_monitor['y min'] = cfg.cv.device_vertical_minimum_um * 1e-6
	# design_index_monitor['z max'] = cfg.pv.device_vertical_maximum_um * 1e-6
	# design_index_monitor['z min'] = cfg.cv.device_vertical_minimum_um * 1e-6
	design_index_monitor['spatial interpolation'] = 'nearest mesh cell'

	# design_index_monitor = fdtd_update_object(fdtd_hook, design_index_monitor, create_object=True)
	device_regions['design_index_monitor'] = design_index_monitor

	# Set up the volumetric electric field monitor inside the design region.  We will need this to compute the adjoint gradient
	design_efield_monitor = {}
	design_efield_monitor['name'] = 'design_efield_monitor'
	design_efield_monitor['type'] = 'DFTMonitor'
	design_efield_monitor['power_monitor'] = False
	design_efield_monitor['monitor type'] = '2D Z-Normal'
	design_efield_monitor['dev_id'] = 0
	design_efield_monitor['x span'] = cfg.pv.device_size_lateral_bordered_um * 1e-6
	# design_efield_monitor['y span'] = cfg.pv.device_size_lateral_bordered_um * 1e-6
	design_efield_monitor['y max'] = cfg.pv.device_vertical_maximum_um * 1e-6
	design_efield_monitor['y min'] = cfg.cv.device_vertical_minimum_um * 1e-6
	# design_efield_monitor['z max'] = cfg.pv.device_vertical_maximum_um * 1e-6
	# design_efield_monitor['z min'] = cfg.cv.device_vertical_minimum_um * 1e-6
	design_efield_monitor['override global monitor settings'] = 1
	design_efield_monitor['use wavelength spacing'] = 1
	design_efield_monitor['use source limits'] = 1
	design_efield_monitor['frequency points'] = cfg.pv.num_design_frequency_points
	design_efield_monitor['output Hx'] = 0
	design_efield_monitor['output Hy'] = 0
	design_efield_monitor['output Hz'] = 0

	# design_efield_monitor = fdtd_update_object(fdtd_hook, design_efield_monitor, create_object=True)
	device_regions['design_efield_monitor'] = design_efield_monitor

	# Put in list format in case there are multiple devices
	device_regions = [device_regions]
	return device_regions


def construct_fwd_srcs_and_monitors(sim_objects):
	forward_sources = sim_objects.get('forward_sources')
	forward_monitors = sim_objects.get('forward_monitors')
	#! They should be matched to each other in the list arrangement

	if forward_monitors is None:		# catch the case where there are no monitors
		forward_monitors = [None] * len(forward_sources)
	return forward_sources, forward_monitors

def construct_adj_srcs_and_monitors(sim_objects):
	adj_sources = [value for key, value in sim_objects.items() if 'adj_src' in key.lower()]

	adj_sources = sim_objects.get('adjoint_sources')
	adj_monitors = sim_objects.get('adjoint_monitors')
	#! They should be matched to each other in the list arrangement
 
	# todo: also turn adj_monitors to nothing? Probably not
	if cfg.cv.use_autograd:
		adj_sources = []
	
	if adj_monitors is None:			# catch the case where there are no monitors
		adj_monitors = [None] * len(adj_sources)
 
	return adj_sources, adj_monitors

#* Determine Figures of Merit
f_bin_all = np.array(range(len(cfg.pv.lambda_values_um)))
f_bin_1 = np.array_split(f_bin_all, 2)[0]
f_bin_2 = np.array_split(f_bin_all, 2)[1]
fom_dict = [{'fwd': [0], 'adj': [0], 'freq_idx_opt': f_bin_all, 'freq_idx_restricted_opt': []},
			{'fwd': [0], 'adj': [2], 'freq_idx_opt': f_bin_all, 'freq_idx_restricted_opt': []}
			]

#* Determine Associated Weights
# Overall Weights for each FoM
weights = np.array( [1] * len(fom_dict) ) # [1.5, 1]		

# Wavelength-dependent behaviour of each FoM (e.g. spectral sorting)
# spectral_weights_by_fom = np.zeros((len(fom_dict), cfg.pv.num_design_frequency_points))
spectral_weights_by_fom = np.zeros((cfg.cv.num_bands, cfg.pv.num_design_frequency_points))
# Each wavelength band needs a left, right, and peak (usually center).
wl_band_bounds = {'left': [], 'peak': [], 'right': []}

def assign_bands(wl_band_bounds, lambda_values_um, num_bands):
	# Reminder that wl_band_bounds is a dictionary {'left': [], 'peak': [], 'right': []}
	
	# Naive method: Split lambda_values_um into num_bands and return lefts, centers, and rights accordingly.
	# i.e. assume all bands are equally spread across all of lambda_values_um without missing any values

	for key in wl_band_bounds.keys():				# Reassign empty lists to be numpy arrays
		wl_band_bounds[key] = np.zeros(num_bands)
	
	wl_bands = np.array_split(lambda_values_um, num_bands)
	# wl_bands = np.array_split(lambda_values_um, [3,7,12,14])  # https://stackoverflow.com/a/67294512 
	# e.g. would give a list of arrays with length 3, 4, 5, 2, and N-(3+4+5+2)

	for band_idx, band in enumerate(wl_bands):
		wl_band_bounds['left'][band_idx] = band[0]
		wl_band_bounds['right'][band_idx] = band[-1]
		wl_band_bounds['peak'][band_idx] = band[(len(band)-1)//2]

	wl_band_bounds['peak'] = wl_band_bounds['left']
	
	return wl_band_bounds

def determine_spectral_weights(spectral_weights_by_fom, wl_band_bound_idxs, mode='identity', *args, **kwargs):
	
	# Right now we have it set up to send specific wavelength bands to specific FoMs
	# This can be thought of as creating a desired transmission spectra for each FoM
	# The bounds of each band are controlled by wl_band_bound_idxs


	for fom_idx, fom in enumerate(spectral_weights_by_fom):
		# We can choose different modes here
		if mode in ['identity']:	# 1 everywhere
			fom[:] = 1

		elif mode in ['hat']:		# 1 everywhere, 0 otherwise
			fom[ wl_band_bound_idxs['left'][fom_idx] : wl_band_bound_idxs['right'][fom_idx]+1 ] = 1
  
		elif mode in ['gaussian']:	# gaussians centered on peaks
			scaling_exp = - (4/7) / np.log( 0.5 )
			band_peak = wl_band_bound_idxs['peak'][fom_idx]
			band_width = wl_band_bound_idxs['left'][fom_idx] - wl_band_bound_idxs['right'][fom_idx]
			wl_idxs = range(spectral_weights_by_fom.shape[-1])
			fom[:] = np.exp( -( wl_idxs - band_peak)**2 / ( scaling_exp * band_width )**2 )

	# # Plotting code to check weighting shapes
	# import matplotlib.pyplot as plt
	# plt.vlines(wl_band_bound_idxs['left'], 0,1, 'b','--')
	# plt.vlines(wl_band_bound_idxs['right'], 0,1, 'r','--')
	# plt.vlines(wl_band_bound_idxs['peak'], 0,1, 'k','-')
	# for fom in spectral_weights_by_fom:
	# 	plt.plot(fom)
	# plt.show()
	# print(3)

	# # Spectral and polarization selectivity information
	# # todo: we re-implemented this but in EnvironmentConstructor.py (i.e. where the FoM is determined)
	# if cd.weight_by_individual_wavelength:
	# 	weight_focal_plane_map_performance_weighting = [ 1.0, 1.0, 1.0, 1.0 ]
	# 	weight_focal_plane_map = [ 1.0, 1.0, 1.0, 1.0 ]
	# 	weight_individual_wavelengths = np.ones( len( lambda_values_um ) )
	# 	weight_individual_wavelengths_by_quad = np.ones((4,num_design_frequency_points))
	
	# # Determine the wavelengths that will be directed to each focal area
	# spectral_focal_plane_map = [
	# 	[0, cd.num_points_per_band],
	# 	[cd.num_points_per_band, 2 * cd.num_points_per_band],
	# 	[2 * cd.num_points_per_band, 3 * cd.num_points_per_band],
	# 	[cd.num_points_per_band, 2 * cd.num_points_per_band]
	# ]

	# # Find indices of desired peak locations in lambda_values_um
	# desired_peak_location_per_band = [ np.argmin( ( lambda_values_um - cd.desired_peaks_per_band_um[ idx ] )**2 ) 
	# 									for idx in range( 0, len( cd.desired_peaks_per_band_um ) ) 
	# 								]

	# # for idx in range( 0, len( desired_peak_location_per_band ) ):
	# # 	get_location = desired_peak_location_per_band[ idx ]
	# # 	half_width = num_points_per_band // 2
	# # 	assert ( get_location - half_width ) >= 0, "Not enough optimization room in short wavelength region!"
	# # 	assert ( get_location + half_width + 1 ) <= num_design_frequency_points, "Not enough optimization room in long wavelength region!"

	# # spectral_focal_plane_map = [
	# # 	[ desired_peak_location_per_band[ 0 ] - ( num_points_per_band // 2 ), desired_peak_location_per_band[ 0 ] + 1 + ( num_points_per_band // 2 ) ],
	# # 	[ desired_peak_location_per_band[ 1 ] - ( num_points_per_band // 2 ), desired_peak_location_per_band[ 1 ] + 1 + ( num_points_per_band // 2 ) ],
	# # 	[ desired_peak_location_per_band[ 2 ] - ( num_points_per_band // 2 ), desired_peak_location_per_band[ 2 ] + 1 + ( num_points_per_band // 2 ) ],
	# # 	[ desired_peak_location_per_band[ 1 ] - ( num_points_per_band // 2 ), desired_peak_location_per_band[ 1 ] + 1 + ( num_points_per_band // 2 ) ]
	# # ]


	# if cd.explicit_band_centering:
	# 	# Determine the wavelengths that will be directed to each focal area
	# 	spectral_focal_plane_map = [
	# 		[0, num_design_frequency_points],
	# 		[0, num_design_frequency_points],
	# 		[0, num_design_frequency_points],
	# 		[0, num_design_frequency_points]
	# 	]


	# 	bandwidth_left_by_quad = [ 
	# 		( desired_peak_location_per_band[ 1 ] - desired_peak_location_per_band[ 0 ] ) // 2,
	# 		( desired_peak_location_per_band[ 1 ] - desired_peak_location_per_band[ 0 ] ) // 2,
	# 		( desired_peak_location_per_band[ 2 ] - desired_peak_location_per_band[ 1 ] ) // 2,
	# 		( desired_peak_location_per_band[ 1 ] - desired_peak_location_per_band[ 0 ] ) // 2
	# 	]

	# 	bandwidth_right_by_quad = [ 
	# 		( desired_peak_location_per_band[ 1 ] - desired_peak_location_per_band[ 0 ] ) // 2,
	# 		( desired_peak_location_per_band[ 2 ] - desired_peak_location_per_band[ 1 ] ) // 2,
	# 		( desired_peak_location_per_band[ 2 ] - desired_peak_location_per_band[ 1 ] ) // 2,
	# 		( desired_peak_location_per_band[ 2 ] - desired_peak_location_per_band[ 1 ] ) // 2
	# 	]

	# 	bandwidth_left_by_quad = [ bandwidth_left_by_quad[ idx ] / 1.75 for idx in range( 0, 4 ) ]
	# 	bandwidth_right_by_quad = [ bandwidth_right_by_quad[ idx ] / 1.75 for idx in range( 0, 4 ) ]

	# 	quad_to_band = [ 0, 1, 2, 1 ]
	# 	if cd.shuffle_green:
	# 		quad_to_band = [ 0, 1, 1, 2 ]

	# 	weight_individual_wavelengths_by_quad = np.zeros((4,num_design_frequency_points))
	# 	for quad_idx in range( 0, 4 ):
	# 		bandwidth_left = bandwidth_left_by_quad[ quad_to_band[ quad_idx ] ]
	# 		bandwidth_right = bandwidth_right_by_quad[ quad_to_band[ quad_idx ] ]

	# 		band_center = desired_peak_location_per_band[ quad_to_band[ quad_idx ] ]

	# 		for wl_idx in range( 0, num_design_frequency_points ):

	# 			choose_width = bandwidth_right
	# 			if wl_idx < band_center:
	# 				choose_width = bandwidth_left


	# 			scaling_exp = -1. / np.log( 0.5 )
	# 			weight_individual_wavelengths_by_quad[ quad_idx, wl_idx ] = np.exp( -( wl_idx - band_center )**2 / ( scaling_exp * choose_width**2 ) )


	# 	# import matplotlib.pyplot as plt
	# 	# colors = [ 'b', 'g', 'r', 'm' ]
	# 	# for quad_idx in range( 0, 4 ):

	# 	# 	plt.plot( lambda_values_um, weight_individual_wavelengths_by_quad[ quad_idx ], color=colors[ quad_idx ] )
	# 	# plt.show()


	# if cd.add_infrared:
	# 	wl_idx_ir_start = 0
	# 	for wl_idx in range( 0, len( lambda_values_um ) ):
	# 		if lambda_values_um[ wl_idx ] > infrared_center_um:
	# 			wl_idx_ir_start = wl_idx - 1
	# 			break

	# 	# Determine the wavelengths that will be directed to each focal area
	# 	spectral_focal_plane_map = [ 
	# 		[ 0, 2 + ( num_points_per_band // 2 ) ],
	# 		[ 2 + ( num_points_per_band // 2 ), 4 + num_points_per_band ],
	# 		[ 4 + num_points_per_band, 6 + num_points_per_band + ( num_points_per_band // 2 ) ],
	# 		[ wl_idx_ir_start, ( wl_idx_ir_start + 3 ) ]
	# 	]

	# 	ir_band_equalization_weights = [
	# 		1. / ( spectral_focal_plane_map[ idx ][ 1 ] - spectral_focal_plane_map[ idx ][ 0 ] ) for idx in range( 0, len( spectral_focal_plane_map ) ) ]

	# 	weight_individual_wavelengths = np.ones( len( lambda_values_um ) )
	# 	for band_idx in range( 0, len( spectral_focal_plane_map ) ):

	# 		for wl_idx in range( spectral_focal_plane_map[ band_idx ][ 0 ], spectral_focal_plane_map[ band_idx ][ 1 ] ):
	# 			weight_individual_wavelengths[ wl_idx ] = ir_band_equalization_weights[ band_idx ]


	# if cd.do_rejection:

	# 	# def weight_accept( transmission ):
	# 	# 	return transmission
	# 	# def weight_reject( transmission ):
	# 	# 	return ( 1 - transmission )

	# 	# performance_weighting_functions = [
	# 	# 	[ weight_accept for idx in range( 0, num_points_per_band ) ] + [ weight_reject for idx in range( 0, 2 * num_points_per_band ) ],
	# 	# 	[ weight_reject for idx in range( 0, num_points_per_band ) ] + [ weight_accept for idx in range( 0, num_points_per_band ) ] + [ weight_reject for idx in range( 0, num_points_per_band ) ],
	# 	# 	[ weight_reject for idx in range( 0, 2 * num_points_per_band ) ] + [ weight_accept for idx in range( 0, num_points_per_band ) ],
	# 	# 	[ weight_reject for idx in range( 0, num_points_per_band ) ] + [ weight_accept for idx in range( 0, num_points_per_band ) ] + [ weight_reject for idx in range( 0, num_points_per_band ) ]
	# 	# ]

	# 	# Determine the wavelengths that will be directed to each focal area
	# 	spectral_focal_plane_map = [
	# 		[0, num_design_frequency_points],
	# 		[0, num_design_frequency_points],
	# 		[0, num_design_frequency_points],
	# 		[0, num_design_frequency_points]
	# 	]

	# 	desired_band_width_um = 0.04#0.08#0.12#0.18


	# 	wl_per_step_um = lambda_values_um[ 1 ] - lambda_values_um[ 0 ]

	# 	#
	# 	# weights will be 1 at the center and drop to 0 at the edge of the band
	# 	#

	# 	weighting_by_band = np.zeros( ( num_bands, num_design_frequency_points ) )

	# 	for band_idx in range( 0, num_bands ):
	# 		wl_center_um = desired_peaks_per_band_um[ band_idx ]

	# 		for wl_idx in range( 0, num_design_frequency_points ):
	# 			wl_um = lambda_values_um[ wl_idx ]
	# 			weight = -0.5 + 1.5 * np.exp( -( ( wl_um - wl_center_um )**2 / ( desired_band_width_um**2 ) ) )
	# 			# weight = -1 + 2 * np.exp( -( ( wl_um - wl_center_um )**2 / ( desired_band_width_um**2 ) ) )
	# 			# weight = -2 + 3 * np.exp( -( ( wl_um - wl_center_um )**2 / ( desired_band_width_um**2 ) ) )

	# 			weighting_by_band[ band_idx, wl_idx ] = weight

	# 	#
	# 	# todo: we need to sort out this weighting scheme and how it works with the performance weighting scheme.
	# 	# we have seen this problem before, but the problem is that this weighting in the FoM will end up bumping
	# 	# up the apparent performance (in the FoM calculation), but as this apparent performance increases, the
	# 	# performance weighting scheme will counteract it and give a lower performance weight.  This, then, fights
	# 	# against the weight the gradient gets from these FoM bumps.  
	# 	#
	# 	spectral_focal_plane_map_directional_weights = np.zeros( ( 4, num_design_frequency_points ) )
	# 	spectral_focal_plane_map_directional_weights[ 0, : ] = weighting_by_band[ 0 ]
	# 	spectral_focal_plane_map_directional_weights[ 1, : ] = weighting_by_band[ 1 ]
	# 	spectral_focal_plane_map_directional_weights[ 2, : ] = weighting_by_band[ 2 ]
	# 	spectral_focal_plane_map_directional_weights[ 3, : ] = weighting_by_band[ 1 ]

	# 	# import matplotlib.pyplot as plt
	# 	# plt.plot( lambda_values_um, spectral_focal_plane_map_directional_weights[ 0 ], color='b', linewidth=2 )
	# 	# plt.plot( lambda_values_um, spectral_focal_plane_map_directional_weights[ 1 ], color='g', linewidth=2 )
	# 	# plt.plot( lambda_values_um, spectral_focal_plane_map_directional_weights[ 2 ], color='r', linewidth=2 )
	# 	# plt.plot( lambda_values_um, spectral_focal_plane_map_directional_weights[ 3 ], color='m', linewidth=2, linestyle='--' )
	# 	# plt.axhline( y=0, linestyle=':', color='k' )
	# 	# plt.show()




	# 	# half_points = num_points_per_band // 2
	# 	# quarter_points = num_points_per_band // 4

	# 	# Determine the wavelengths that will be directed to each focal area
	# 	# spectral_focal_plane_map = [
	# 	# 	[0, num_points_per_band + half_points ],
	# 	# 	[num_points_per_band - half_points, 2 * num_points_per_band + half_points ],
	# 	# 	[2 * num_points_per_band - half_points, 3 * num_points_per_band],
	# 	# 	[num_points_per_band - half_points, 2 * num_points_per_band + half_points ]
	# 	# ]

	# 	# total_sizes = [ spectral_focal_plane_map[ idx ][ 1 ] - spectral_focal_plane_map[ idx ][ 0 ] for idx in range( 0, len( spectral_focal_plane_map ) ) ]

	# 	# reject_sizes = [ half_points, 2 * half_points, half_points, 2 * half_points ]
	# 	# accept_sizes = [ total_sizes[ idx ] - reject_sizes[ idx ] for idx in range( 0, len( total_sizes ) ) ]

	# 	# reject_weights = [ -1 for idx in range( 0, len( reject_sizes ) ) ]
	# 	# accept_weights = [ 1 for idx in range( 0, len( accept_sizes ) ) ]

	# 	# spectral_focal_plane_map_directional_weights = [
	# 	# 	[ accept_weights[ 0 ] for idx in range( 0, num_points_per_band ) ] +
	# 	# 	[ reject_weights[ 0 ] for idx in range( num_points_per_band, num_points_per_band + half_points ) ],
			
	# 	# 	[ reject_weights[ 1 ] for idx in range( num_points_per_band - half_points, num_points_per_band ) ] +
	# 	# 	[ accept_weights[ 1 ] for idx in range( num_points_per_band, 2 * num_points_per_band ) ] +
	# 	# 	[ reject_weights[ 1 ] for idx in range( 2 * num_points_per_band, 2 * num_points_per_band + half_points ) ],

	# 	# 	[ reject_weights[ 2 ] for idx in range( 2 * num_points_per_band - half_points, 2 * num_points_per_band ) ] +
	# 	# 	[ accept_weights[ 2 ] for idx in range( 2 * num_points_per_band, 3 * num_points_per_band - 4 ) ] +
	# 	# 	[ reject_weights[ 2 ] for idx in range( 3 * num_points_per_band - 4, 3 * num_points_per_band ) ],

	# 	# 	[ reject_weights[ 3 ] for idx in range( num_points_per_band - half_points, num_points_per_band ) ] +
	# 	# 	[ accept_weights[ 3 ] for idx in range( num_points_per_band, 2 * num_points_per_band ) ] +
	# 	# 	[ reject_weights[ 3 ] for idx in range( 2 * num_points_per_band, 2 * num_points_per_band + half_points ) ],
	# 	# ]	
 
	return spectral_weights_by_fom

wl_band_bounds = assign_bands(wl_band_bounds, cfg.pv.lambda_values_um, cfg.cv.num_bands)
logging.info(f'Desired peak locations per band are: {wl_band_bounds["peak"]}')

# Convert to the nearest matching indices of lambda_values_um
wl_band_bound_idxs = copy.deepcopy(wl_band_bounds)
for key, val in wl_band_bounds.items():
	wl_band_bound_idxs[key] = np.searchsorted(cfg.pv.lambda_values_um, val)

spectral_weights_by_fom = determine_spectral_weights(spectral_weights_by_fom, wl_band_bound_idxs, mode='gaussian') # args, kwargs



def calculate_performance_weighting(fom_list):
	'''All gradients are combined with a weighted average in Eq.(3), with weights chosen according to Eq.(2) such that
	all figures of merit seek the same efficiency. In these equations, FoM represents the current value of a figure of 
	merit, N is the total number of figures of merit, and wi represents the weight applied to its respective merit function's
	gradient. The maximum operator is used to ensure the weights are never negative, thus ignoring the gradient of 
	high-performing figures of merit rather than forcing the figure of merit to decrease. The 2/N factor is used to ensure all
	weights conveniently sum to 1 unless some weights were negative before the maximum operation. Although the maximum operator
	is non-differentiable, this function is used only to apply the gradient rather than to compute it. Therefore, it does not 
	affect the applicability of the adjoint method.
	
	Taken from: https://doi.org/10.1038/s41598-021-88785-5'''

	performance_weighting = (2. / len(fom_list)) - fom_list**2 / np.sum(fom_list**2)

	# Zero-shift and renormalize
	if np.min( performance_weighting ) < 0:
		performance_weighting -= np.min(performance_weighting)
		performance_weighting /= np.sum(performance_weighting)

	return performance_weighting

def overall_combine_function( fom_objs, weights, property_str):
	'''calculate figure of merit / adjoint gradient as a weighted sum, the instructions of which are written here'''
	#! The spectral parts must be handled separately, i.e. here.
	# todo: insert a spectral weight vector.
	#! TODO: What about dispersion? Is that an additional dimension on top of everything?

	quantity_numbers = 0

	# Renormalize weights to sum to 1 (across the FoM dimension)
	#! Might not interact well with negative gradients for restriction
	# [20240216 Ian] But I think normalization in this way should still apply only to the weights, to keep their ratios correct
	weights /= np.sum(weights, 0)

	for f_idx, f in enumerate(fom_objs):
		# TODO:
		# First decide: how many entries do you want in the list of true_fom??
		# Second: weights[f_idx] needs to be a spectral vector of some description
		# Or some slice of the wavelength must be pre-determined!

		quantity_numbers += weights[f_idx] * np.squeeze(getattr(f, property_str))

	return quantity_numbers

def overall_figure_of_merit( indiv_foms, weights ):
	'''calculate figure of merit according to weights and FoM instructions, which are written here'''

	figures_of_merit_numbers = overall_combine_function( indiv_foms, weights, 'true_fom' )
	return figures_of_merit_numbers

def overall_adjoint_gradient( indiv_foms, weights ):
	'''Calculate gradient according to weights and individual adjoints, the instructions will be written here'''	
	# if nothing is different, then this should take the exact same form as the function overall_figure_of_merit()
	# but there might be some subtleties

	#! DEBUG: Check the proportions of the individual FoM gradients. One could be too small relative to the others.
	#! This becomes crucial when a functionality isn't working or when the performance weighting needs to be adjusted.
	for f_idx, f in enumerate(indiv_foms):
		logging.info(f'Average absolute gradient for FoM {f_idx} is {np.mean(np.abs(f.gradient))}.')

	final_weights = weights

	# Calculate performance weighting function here and multiply by the existing weights.
	# Performance weighting is an adjustment factor to the gradient computation so that each focal area's FoM
	# seeks the same efficiency. See Eq. 2, https://doi.org/10.1038/s41598-021-88785-5
	# calculate_performance_weighting() takes a list of numbers so we must assemble this accordingly.
	process_fom = np.array([])
	for f in indiv_foms:
		# Sum over wavelength
		process_fom = np.append(process_fom, np.sum(f.true_fom, -1))	#! these array dimensions might cause problems
		# todo: add options to adjust performance weighting by wavelength
	performance_weighting = calculate_performance_weighting(process_fom)
	final_weights *= performance_weighting[..., np.newaxis]

	device_gradient = overall_combine_function( indiv_foms, final_weights, 'gradient' )
	return device_gradient

if __name__ == "__main__":

	# todo: probably the way forward is to define everything in construct_sim_objects().
	# todo: Should assign things like sim_objects['device_monitors'] = design_index_monitor, design_Efield_monitor
	# todo: then, the other methods should be pulling from those keys.
	# todo: e.g. construct_device_regions should just return sim_objects['device_monitors']['design_index_monitor'] or something.

	sim_objects = construct_sim_objects()
	design_regions = construct_device_regions()
	forward_srcs, forward_monitors = construct_fwd_srcs_and_monitors(sim_objects)
	adjoint_srcs, adj_monitors = construct_adj_srcs_and_monitors(sim_objects)
	print(3)