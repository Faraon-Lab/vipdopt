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
	fdtd['x span'] = cfg.pv.fdtd_region_size_lateral_um * 1e-6
	fdtd['y span'] = cfg.pv.fdtd_region_size_lateral_um * 1e-6
	fdtd['z max'] = cfg.pv.fdtd_region_maximum_vertical_um * 1e-6
	fdtd['z min'] = cfg.pv.fdtd_region_minimum_vertical_um * 1e-6
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
			forward_src['angle theta'] = cfg.pv.source_angle_theta_deg
			forward_src['angle phi'] = cfg.cv.source_angle_phi_deg
			forward_src['polarization angle'] = cfg.cv.xy_phi_rotations[xy_idx]
			forward_src['direction'] = 'Backward'
			forward_src['x span'] = 2 * cfg.pv.fdtd_region_size_lateral_um * 1e-6
			forward_src['y span'] = 2 * cfg.pv.fdtd_region_size_lateral_um * 1e-6
			forward_src['z'] = cfg.pv.src_maximum_vertical_um * 1e-6

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
		adjoint_sources.append([])
		for xy_idx in range(0, 2):
			adj_src = {}
			adj_src['name'] = f'adj_src_{adj_src_idx}' + cfg.cv.xy_names[xy_idx]
			adj_src['type'] = 'DipoleSource'
			adj_src['x'] = cfg.pv.adjoint_x_positions_um[adj_src_idx] * 1e-6
			adj_src['y'] = cfg.pv.adjoint_y_positions_um[adj_src_idx] * 1e-6
			adj_src['z'] = cfg.pv.adjoint_vertical_um * 1e-6
			adj_src['theta'] = 90
			adj_src['phi'] = cfg.cv.xy_phi_rotations[xy_idx]
			adj_src['wavelength start'] = cfg.cv.lambda_min_um * 1e-6
			adj_src['wavelength stop'] = cfg.cv.lambda_max_um * 1e-6
			
			# adj_src = fdtd_update_object(fdtd_hook, adj_src, create_object=True)
			sim_objects[adj_src['name']] = adj_src
			adjoint_sources[adj_src_idx].append(adj_src)
	sim_objects['adjoint_sources'] = adjoint_sources

	# Set up adjoint point monitors to get electric field strength at focus spots.  This will allow us to
	# compute the figure of merit as well as weight the adjoint simulations properly in calculation of the
	# gradient.
	focal_monitors = []

	for adj_src in range(0, cfg.pv.num_adjoint_sources):
		focal_monitor = {}
		focal_monitor['name'] = f'focal_monitor_{adj_src}'
		focal_monitor['type'] = 'DFTMonitor'
		focal_monitor['power_monitor'] = True
		focal_monitor['monitor type'] = 'point'
		focal_monitor['x'] = cfg.pv.adjoint_x_positions_um[adj_src] * 1e-6
		focal_monitor['y'] = cfg.pv.adjoint_y_positions_um[adj_src] * 1e-6
		focal_monitor['z'] = cfg.pv.adjoint_vertical_um * 1e-6
		focal_monitor['override global monitor settings'] = 1
		focal_monitor['use wavelength spacing'] = 1
		focal_monitor['use source limits'] = 1
		focal_monitor['frequency points'] = cfg.pv.num_design_frequency_points
		
		# focal_monitor = fdtd_update_object(fdtd_hook, focal_monitor, create_object=True)
		sim_objects[focal_monitor['name']] = focal_monitor
		focal_monitors.append(focal_monitor)
	sim_objects['focal_monitors'] = focal_monitors

	transmission_monitors = []

	for adj_src in range(0, cfg.pv.num_adjoint_sources):
		transmission_monitor = {}
		transmission_monitor['name'] = f'transmission_monitor_{adj_src}'
		transmission_monitor['type'] = 'DFTMonitor'
		transmission_monitor['power_monitor'] = True
		transmission_monitor['monitor type'] = '2D Z-normal'
		transmission_monitor['x'] = cfg.pv.adjoint_x_positions_um[adj_src] * 1e-6
		transmission_monitor['x span'] = 0.5 * cfg.cv.device_size_lateral_um * 1e-6
		transmission_monitor['y'] = cfg.pv.adjoint_y_positions_um[adj_src] * 1e-6
		transmission_monitor['y span'] = 0.5 * cfg.cv.device_size_lateral_um * 1e-6
		transmission_monitor['z'] = cfg.pv.adjoint_vertical_um * 1e-6
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
	transmission_monitor_focal['monitor type'] = '2D Z-normal'
	transmission_monitor_focal['x'] = 0 * 1e-6
	transmission_monitor_focal['x span'] = cfg.cv.device_size_lateral_um * 1e-6
	transmission_monitor_focal['y'] = 0 * 1e-6
	transmission_monitor_focal['y span'] = cfg.cv.device_size_lateral_um * 1e-6
	transmission_monitor_focal['z'] = cfg.pv.adjoint_vertical_um * 1e-6
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
		source_block['y'] = 0 * 1e-6
		source_block['y span'] = 1.1*4/3*1.2 * cfg.cv.device_size_lateral_um * 1e-6
		source_block['z min'] = (cfg.pv.device_vertical_maximum_um + 3 * cfg.cv.mesh_spacing_um) * 1e-6
		source_block['z max'] = ( cfg.pv.device_vertical_maximum_um + 3 * cfg.cv.mesh_spacing_um + cfg.pv.pec_aperture_thickness_um ) * 1e-6
		source_block['material'] = 'PEC (Perfect Electrical Conductor)'
	
		# source_block = fdtd_update_object(fdtd_hook, source_block, create_object=True)
		sim_objects['source_block'] = source_block
		
		source_aperture = {}
		source_aperture['name'] = 'source_aperture'
		source_aperture['type'] = 'Rectangle'
		source_aperture['x'] = 0 * 1e-6
		source_aperture['x span'] = cfg.cv.device_size_lateral_um * 1e-6
		source_aperture['y'] = 0 * 1e-6
		source_aperture['y span'] = cfg.cv.device_size_lateral_um * 1e-6
		source_aperture['z min'] = (cfg.pv.device_vertical_maximum_um + 3 * cfg.cv.mesh_spacing_um) * 1e-6
		source_aperture['z max'] = (cfg.pv.device_vertical_maximum_um + 3 * cfg.cv.mesh_spacing_um + cfg.pv.pec_aperture_thickness_um) * 1e-6
		source_aperture['index'] = cfg.cv.background_index
	
		# source_aperture = fdtd_update_object(fdtd_hook, source_aperture, create_object=True)
		sim_objects['source_aperture'] = source_aperture


	# Set up sidewalls on the side to try and attenuate crosstalk
	device_sidewalls = []

	for dev_sidewall_idx in range(0, cfg.cv.num_sidewalls):
		device_sidewall = {}
		device_sidewall['name'] = 'sidewall_' + str(dev_sidewall_idx)
		device_sidewall['type'] = 'Rectangle'
		device_sidewall['x'] = cfg.pv.sidewall_x_positions_um[dev_sidewall_idx] * 1e-6
		device_sidewall['x span'] = cfg.pv.sidewall_xspan_positions_um[dev_sidewall_idx] * 1e-6
		device_sidewall['y'] = cfg.pv.sidewall_y_positions_um[dev_sidewall_idx] * 1e-6
		device_sidewall['y span'] = cfg.pv.sidewall_yspan_positions_um[dev_sidewall_idx] * 1e-6
		if cfg.cv.sidewall_extend_focalplane:
			device_sidewall['z min'] = cfg.pv.adjoint_vertical_um * 1e-6
		else:   
			device_sidewall['z min'] = cfg.cv.sidewall_vertical_minimum_um * 1e-6
		device_sidewall['z max'] = cfg.pv.device_vertical_maximum_um * 1e-6
		device_sidewall['material'] = cfg.cv.sidewall_material
	
		# device_sidewall = fdtd_update_object(fdtd_hook, device_sidewall, create_object=True)
		sim_objects[device_sidewall['name']] = device_sidewall
		device_sidewalls.append(device_sidewall)
	sim_objects['device_sidewalls'] = device_sidewalls


	# Apply finer mesh regions restricted to sidewalls
	device_sidewall_meshes = []

	for dev_sidewall_idx in range(0, cfg.cv.num_sidewalls):
		device_sidewall_mesh = {}
		device_sidewall_mesh['name'] = f'mesh_sidewall_{dev_sidewall_idx}'
		device_sidewall_mesh['type'] = 'Mesh'
		device_sidewall_mesh['set maximum mesh step'] = 1
		device_sidewall_mesh['override x mesh'] = 0
		device_sidewall_mesh['override y mesh'] = 0
		device_sidewall_mesh['override z mesh'] = 0
		device_sidewall_mesh['based on a structure'] = 1
		device_sidewall_mesh['structure'] = device_sidewalls[dev_sidewall_idx]['name']
		if device_sidewalls[dev_sidewall_idx]['x span'] < device_sidewalls[dev_sidewall_idx]['y span']:
			device_sidewall_mesh['override x mesh'] = 1
			device_sidewall_mesh['dx'] = cfg.cv.mesh_spacing_um * 1e-6 #(sidewall_thickness_um / 5) * 1e-6
		else:
			device_sidewall_mesh['override y mesh'] = 1
			device_sidewall_mesh['dy'] = cfg.cv.mesh_spacing_um * 1e-6 #(sidewall_thickness_um / 5) * 1e-6
		
		# device_sidewall_mesh = fdtd_update_object(fdtd_hook, device_sidewall_mesh, create_object=True)
		sim_objects[device_sidewall_mesh['name']] = device_sidewall_mesh
		device_sidewall_meshes.append(device_sidewall_mesh)
	sim_objects['device_sidewall_meshes'] = device_sidewall_meshes

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
								'y': np.linspace(-0.5 * cfg.pv.device_size_lateral_bordered_um, 0.5 * cfg.pv.device_size_lateral_bordered_um, cfg.pv.device_voxels_lateral_bordered),
								'z': np.linspace(cfg.cv.device_vertical_minimum_um, cfg.pv.device_vertical_maximum_um, cfg.pv.device_voxels_vertical)
							},					# Coordinates of each voxel can be regular or irregular. Gives the most freedom for feature generation down the road
				'size_voxels': np.array( [cfg.pv.device_voxels_lateral_bordered, cfg.pv.device_voxels_lateral_bordered, cfg.pv.device_voxels_vertical] ),
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
	design_import['y span'] = cfg.pv.device_size_lateral_bordered_um * 1e-6
	design_import['z max'] = cfg.pv.device_vertical_maximum_um * 1e-6
	design_import['z min'] = cfg.cv.device_vertical_minimum_um * 1e-6

	# design_import = fdtd_update_object(fdtd_hook, design_import, create_object=True)
	device_regions['design_import'] = design_import

	device_mesh = {}
	device_mesh['name'] = 'device_mesh'
	device_mesh['type'] = 'Mesh'
	device_mesh['dev_id'] = 0
	device_mesh['x'] = 0
	device_mesh['x span'] = cfg.pv.fdtd_region_size_lateral_um * 1e-6
	device_mesh['y'] = 0
	device_mesh['y span'] = cfg.pv.fdtd_region_size_lateral_um * 1e-6
	device_mesh['z max'] = ( cfg.pv.device_vertical_maximum_um + 0.5 ) * 1e-6
	device_mesh['z min'] = ( cfg.cv.device_vertical_minimum_um - 0.5 ) * 1e-6
	device_mesh['dx'] = cfg.cv.mesh_spacing_um * 1e-6
	device_mesh['dy'] = cfg.cv.mesh_spacing_um * 1e-6
	device_mesh['dz'] = cfg.cv.mesh_spacing_um * 1e-6
 
	# device_mesh = fdtd_update_object(fdtd_hook, device_mesh, create_object=True)
	device_regions['device_mesh'] = device_mesh

	# Add device index monitor
	design_index_monitor = {}
	design_index_monitor['name'] = 'design_index_monitor'
	design_index_monitor['type'] = 'IndexMonitor'
	design_index_monitor['monitor type'] = '3D'
	design_index_monitor['dev_id'] = 0
	design_index_monitor['x span'] = cfg.pv.device_size_lateral_bordered_um * 1e-6
	design_index_monitor['y span'] = cfg.pv.device_size_lateral_bordered_um * 1e-6
	design_index_monitor['z max'] = cfg.pv.device_vertical_maximum_um * 1e-6
	design_index_monitor['z min'] = cfg.cv.device_vertical_minimum_um * 1e-6
	design_index_monitor['spatial interpolation'] = 'nearest mesh cell'

	# design_index_monitor = fdtd_update_object(fdtd_hook, design_index_monitor, create_object=True)
	device_regions['design_index_monitor'] = design_index_monitor

	# Set up the volumetric electric field monitor inside the design region.  We will need this to compute the adjoint gradient
	design_efield_monitor = {}
	design_efield_monitor['name'] = 'design_efield_monitor'
	design_efield_monitor['type'] = 'DFTMonitor'
	design_efield_monitor['power_monitor'] = False
	design_efield_monitor['monitor type'] = '3D'
	design_efield_monitor['dev_id'] = 0
	design_efield_monitor['x span'] = cfg.pv.device_size_lateral_bordered_um * 1e-6
	design_efield_monitor['y span'] = cfg.pv.device_size_lateral_bordered_um * 1e-6
	design_efield_monitor['z max'] = cfg.pv.device_vertical_maximum_um * 1e-6
	design_efield_monitor['z min'] = cfg.cv.device_vertical_minimum_um * 1e-6
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



if __name__ == "__main__":
	sim_objects = construct_sim_objects()
	design_regions = construct_device_regions()