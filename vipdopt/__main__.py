"""Run the Vipdopt software package."""
import logging
import numpy as np
import os
import sys
from argparse import SUPPRESS, ArgumentParser

sys.path.append(os.getcwd())    # 20240219 Ian: Only added this so I could debug some things from my local VSCode
import vipdopt
from vipdopt.optimization import FoM
from vipdopt.utils import setup_logger, import_lumapi

f = sys.modules[__name__].__file__
if not f:
	raise ModuleNotFoundError('SHOULD NEVER REACH HERE')

path = os.path.dirname(f)
path = os.path.join(path, '..')
sys.path.insert(0, path)


from pathlib import Path

from vipdopt.project import Project

if __name__ == '__main__':
	parser = ArgumentParser(
		prog='vipdopt',
		description='Volumetric Inverse Photonic Design Optimizer',
	)
	parser.add_argument('-v', '--verbose', action='store_const', const=True,
							default=False, help='Enable verbose output.')
	parser.add_argument(
		 '--log',
		type=Path,
		default=None,
		help='Path to the log file.'
	)

	subparsers = parser.add_subparsers()
	opt_parser = subparsers.add_parser('optimize')

	opt_parser.add_argument('-v', '--verbose', action='store_const', const=True,
							default=SUPPRESS, help='Enable verbose output.')
	opt_parser.add_argument(
		'directory',
		type=Path,
		help='Project directory to use',
	)
	opt_parser.add_argument(
		 '--log',
		type=Path,
		default=SUPPRESS,
		help='Path to the log file.'
	)
	
	opt_parser.add_argument(
		'config',
		type=str,
		help='Configuration file to use in the optimization',
	)

	args = parser.parse_args()
	if args.log is None:
		args.log = args.directory / 'dev.log'

	# Set verbosity
	level = logging.DEBUG if args.verbose else logging.INFO
	vipdopt.logger = setup_logger('global_logger', level, log_file=args.log)
	vipdopt.logger.info(f'Starting up optimization file: {os.path.basename(__file__)}')
	vipdopt.logger.info("All modules loaded.")
	vipdopt.logger.debug(f'Current working directory: {os.getcwd()}')

	#
	#* Step 0: Set up simulation conditions and environment.
	# i.e. current sources, boundary conditions, supporting structures, surrounding regions.
	# Also any other necessary editing of the Lumerical environment and objects.
	vipdopt.logger.info('Beginning Step 0: Project Setup...')
	
	project = Project()
	# What does the Project class contain?
	# 'dir': directory where it's stored; 'config': SonyBayerConfig object; 'optimization': Optimization object;
	# 'optimizer': Adam/GDOptimizer object; 'device': Device object;
	# 'base_sim': Simulation object; 'src_to_sim_map': dict with source names as keys, Simulation objects as values
	# 'foms': list of FoM objects, 'weights': array of shape (#FoMs, nλ)

	# TODO: Partitioning the base_sim into simulations: i.e. a list of Simulation objects
	# TODO: And the same with devices
	# Multiple simulations may be created here due to the need for large-area simulation segmentation, or genetic optimizations

	# project.base_sim = LumericalSimulation(sim.json)
	# project.optimizer = AdamOptimizer()
	# project.foms = some list of FoMs
	# project.save_as('test_project')
	project.load_project(args.directory, file_name='processed_config.yml')
	project.save()

	# Now that config is loaded, set up lumapi
	vipdopt.lumapi = import_lumapi(project.config.data['lumapi_filepath_local'])   #todo: If Windows

	# # Debug that base_sim is correctly created...
	# project.base_sim.connect()		
	# project.base_sim.save( project.folders['DATA'] / project.base_sim.info['name'] )
 
	# # For each subdevice (design) region, create a field_shape variable to store the E- and H-fields for adjoint calculation.
	# field_shapes = []
	# # Open simulation files and create subdevice (design) regions as well as attached meshes and monitors.
	# field_shape = project.base_sim.get_field_shape()
	# # correct_field_shape = ( project.config.data['device_voxels_simulation_mesh_lateral_bordered'],
	# #                 		project.config.data['device_voxels_simulation_mesh_vertical']
	# #                 	)
	# # assert field_shape == correct_field_shape, f"The sizes of field_shape {field_shape} should instead be {correct_field_shape}."
	# field_shapes.append(field_shape)
	# project.design.field_shape = field_shape

	vipdopt.logger.info('Completed Step 0: Project and Device Setup')

	# #
	# # Step 1
	# #
	
	# # Create all different versions of simulation that are run    

	forward_sources = []	# Meant to store Source objects
	adjoint_sources = []	# Meant to store Source objects
 
	# for simulation in simulations:
	# 	# We create fwd_src, adj_src as Source objects, then link them together using an "FoM" class, 
	# 	# which can then store E_fwd and E_adj gradients. We can also then just return a gradient by doing FoM.E_fwd * FoM.E_adj
	# 	# todo: monitor might also need to be their own class.

	# 	# Add the forward sources
	# 	fwd_srcs, fwd_mons = env_constr.construct_fwd_srcs_and_monitors(simulation)
	# 	for idx, fwd_src in enumerate(fwd_srcs):
	# 		forward_sources.append(Source.Fwd_Src(fwd_src, monitor_object=fwd_mons[fwd_src['attached_monitor']], sim_id=simulation['SIM_INFO']['sim_id']))

	# 	# Add dipole or adjoint sources, and their monitors.
	# 	# Adjoint monitors and what data to obtain should be included in the FoM class / indiv_fom.
	# 	adj_srcs, adj_mons = env_constr.construct_adj_srcs_and_monitors(simulation)
	# 	for idx, adj_src in enumerate(adj_srcs):
	# 		adjoint_sources.append(Source.Adj_Src(adj_src, monitor_object=adj_mons[adj_src['attached_monitor']], sim_id=simulation['SIM_INFO']['sim_id']))


	# fwd_srcs = [f'forward_src_{d}' for d in 'xy']
	# adj_srcs = [f'adj_src_{n}{d}' for n in range(project.config['num_adjoint_sources']) for d in 'xy']
	#! TODO: Change the filenames to temp_fwd_i.fsp, temp_adj_i.fsp with the appropriate maps
	fwd_srcs = [src.name for src in project.base_sim.sources() if any(x in src.name for x in ['forward','fwd'])]
	adj_srcs = [src.name for src in project.base_sim.sources() if any(x in src.name for x in ['adjoint','adj'])]
	# We can also obtain these lists in a similar way from project.src_to_sim_map.keys()

	# vipdopt.logger.info('Creating forward simulations...')
	# fwd_sims = [project.base_sim.with_enabled([src], src) for src in fwd_srcs]
	# vipdopt.logger.info('Creating adjoint simulations...')
	# adj_sims = [project.base_sim.with_enabled([src], src) for src in adj_srcs]
	# all_sims = fwd_sims + adj_sims
	
	# license_checked = False
	# for sim_name, sim in project.src_to_sim_map.items():
	# 	vipdopt.logger.info(f'Creating sim with enabled: {sim_name}')
	# 	sim.connect(license_checked=license_checked)
	# 	if not license_checked:
	# 		license_checked = True
	# 	sim.setup_env_resources()
	# 	sim.save( os.path.abspath( project.folders['DATA'] / sim.info['name'] ) )
	# 	# # sim.close()

	vipdopt.logger.info('Completed Step 1: All Simulations Setup')


	#
	#* Step 2: Set up figure of merit. Determine monitors from which E-fields, Poynting fields, and transmission data will be drawn.
	# Create sources and figures of merit (FoMs)
	# Handle all processing of weighting and functions that go into the figure of merit
	#

	vipdopt.logger.info("Beginning Step 2: Figure of Merit setup.")

	# Setup FoMs
	f_bin_all = np.array(range(len(project.config['lambda_values_um'])))
	f_bin_1 = np.array_split(f_bin_all, 2)[0]
	f_bin_2 = np.array_split(f_bin_all, 2)[1]
	# fom_dict_old = [{'fwd': [0], 'adj': [0], 'freq_idx_opt': f_bin_all, 'freq_idx_restricted_opt': []},
	# 			{'fwd': [0], 'adj': [2], 'freq_idx_opt': f_bin_all, 'freq_idx_restricted_opt': []}
	# 			]

	foms: list[FoM] = []
	weights = []
	fom_dict: dict
	for name, fom_dict in project.config.pop('figures_of_merit').items():
		foms.append(FoM.from_dict(name, fom_dict, project.src_to_sim_map))
		weights.append(fom_dict['weight'])
	project.foms = foms
	project.weights = weights
	# todo: do we need to reassign this to a FUNCTION?
	#! also sum isn't working
	full_fom = sum(np.multiply(weights, foms), FoM.zero(foms[0]))
	# This is another FoM object
	# that basically calls all of the weights and FoMs that are assigned to it
	# so you can just call full_fom._bayer_fom()
	# Call it after all the data has been stored!

	vipdopt.logger.info("Completed Step 2: Figure of Merit setup complete.")
	# utility.backup_all_vars(globals(), cfg.cv.shelf_fn)

	#
	# Step 3
	#
	vipdopt.logger.info('Beginning Step 3: Run Optimization')
	
	# Update optimization with fom_function
	project.optimization.fom = full_fom

	# Create histories and plots
	# todo: SAVE TO OPTIMIZATION OR TO PROJECT?
	# opt.create_history(cfg.cv.fom_types, cfg.cv.total_num_iterations, cfg.pv.num_design_frequency_points)
 
	project.optimization.run()

	vipdopt.logger.info('Completed Step 3: Run Optimization')

	#
	# Cleanup
	#
	# for i in range(len(all_sims)):
