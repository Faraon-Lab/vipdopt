"""Code for managing a project."""

from __future__ import annotations

import sys
import os
from copy import copy
import shutil
from pathlib import Path
from typing import Any

import numpy as np

sys.path.append(os.getcwd())
import vipdopt
from vipdopt.configuration import Config, SonyBayerConfig
from vipdopt.optimization import Device, FoM, GradientOptimizer, Optimization
from vipdopt.optimization.filter import Sigmoid
from vipdopt.simulation import LumericalEncoder, LumericalSimulation
from vipdopt.utils import PathLike, ensure_path


def create_internal_folder_structure(main_dir, work_dir, debug_mode=False):
	# global DATA_FOLDER, SAVED_SCRIPTS_FOLDER, OPTIMIZATION_INFO_FOLDER, OPTIMIZATION_PLOTS_FOLDER
	# global DEBUG_COMPLETED_JOBS_FOLDER, PULL_COMPLETED_JOBS_FOLDER, EVALUATION_FOLDER, EVALUATION_CONFIG_FOLDER, EVALUATION_UTILS_FOLDER
	
	directories = {'MAIN': main_dir}

	#* Output / Save Paths 
	DATA_FOLDER = work_dir
	SAVED_SCRIPTS_FOLDER = os.path.join(DATA_FOLDER, 'saved_scripts')
	OPTIMIZATION_INFO_FOLDER = os.path.join(DATA_FOLDER, 'opt_info')
	OPTIMIZATION_PLOTS_FOLDER = os.path.join(OPTIMIZATION_INFO_FOLDER, 'plots')
	DEBUG_COMPLETED_JOBS_FOLDER = 'ares_test_dev'
	PULL_COMPLETED_JOBS_FOLDER = DATA_FOLDER
	if debug_mode:
		PULL_COMPLETED_JOBS_FOLDER = DEBUG_COMPLETED_JOBS_FOLDER

	EVALUATION_FOLDER = os.path.join(main_dir, 'eval')
	EVALUATION_CONFIG_FOLDER = os.path.join(EVALUATION_FOLDER, 'configs')
	EVALUATION_UTILS_FOLDER = os.path.join(EVALUATION_FOLDER, 'utils')

	# parameters['MODEL_PATH'] = os.path.join(DATA_FOLDER, 'model.pth')
	# parameters['OPTIMIZER_PATH'] = os.path.join(DATA_FOLDER, 'optimizer.pth')


	# #* Save out the various files that exist right before the optimization runs for debugging purposes.
	# # If these files have changed significantly, the optimization should be re-run to compare to anything new.

	if not os.path.isdir( SAVED_SCRIPTS_FOLDER ):
		os.mkdir( SAVED_SCRIPTS_FOLDER )
	if not os.path.isdir( OPTIMIZATION_INFO_FOLDER ):
		os.mkdir( OPTIMIZATION_INFO_FOLDER )
	if not os.path.isdir( OPTIMIZATION_PLOTS_FOLDER ):
		os.mkdir( OPTIMIZATION_PLOTS_FOLDER )

	try:
		shutil.copy2( main_dir + "/slurm_vis10lyr.sh", SAVED_SCRIPTS_FOLDER + "/slurm_vis10lyr.sh" )
	except Exception as ex:
		pass
	# shutil.copy2( cfg.python_src_directory + "/SonyBayerFilterOptimization.py", SAVED_SCRIPTS_FOLDER + "/SonyBayerFilterOptimization.py" )
	# shutil.copy2( os.path.join(python_src_directory, yaml_filename), 
	# 			 os.path.join(SAVED_SCRIPTS_FOLDER, os.path.basename(yaml_filename)) )
	# shutil.copy2( python_src_directory + "/configs/SonyBayerFilterParameters.py", SAVED_SCRIPTS_FOLDER + "/SonyBayerFilterParameters.py" )
	# # TODO: et cetera... might have to save out various scripts from each folder

	#  Create convenient folder for evaluation code
	if not os.path.isdir( EVALUATION_FOLDER ):
		os.mkdir( EVALUATION_FOLDER )
	
	
	# TODO: finalize this when the Project directory internal structure is finalized    
	# if os.path.exists(EVALUATION_CONFIG_FOLDER):
	#     shutil.rmtree(EVALUATION_CONFIG_FOLDER)
	# shutil.copytree(os.path.join(main_dir, "configs"), EVALUATION_CONFIG_FOLDER)
	# if os.path.exists(EVALUATION_UTILS_FOLDER):
	#     shutil.rmtree(EVALUATION_UTILS_FOLDER)
	# shutil.copytree(os.path.join(main_dir, "utils"), EVALUATION_UTILS_FOLDER)

	#shutil.copy2( os.path.abspath(python_src_directory + "/evaluation/plotter.py"), EVALUATION_UTILS_FOLDER + "/plotter.py" )
	
	
	directories.update( {'DATA': DATA_FOLDER,
						'SAVED_SCRIPTS': SAVED_SCRIPTS_FOLDER,
						'OPT_INFO': OPTIMIZATION_INFO_FOLDER,
						'OPT_PLOTS': OPTIMIZATION_PLOTS_FOLDER,
						'PULL_COMPLETED_JOBS': PULL_COMPLETED_JOBS_FOLDER,
						'DEBUG_COMPLETED_JOBS': DEBUG_COMPLETED_JOBS_FOLDER,
						'EVALUATION': EVALUATION_FOLDER,
						'EVAL_CONFIG': EVALUATION_CONFIG_FOLDER,
						'EVAL_UTILS': EVALUATION_UTILS_FOLDER
					} )
	return directories


class Project:
	"""Class for managing the loading and saving of projects."""

	def __init__(self, config_type: type[Config]=SonyBayerConfig) -> None:
		"""Initialize a Project."""
		self.dir = Path('.')  # Project directory; defaults to root
		self.config = config_type()
		self.config_type = config_type

		self.optimization: Optimization = None
		self.optimizer: GradientOptimizer = None
		self.device: Device = None
		self.base_sim: LumericalSimulation = None
		self.src_to_sim_map = {}

	@classmethod
	def from_dir(
		cls: type[Project],
		project_dir: PathLike,
		config_type: type[Config]=SonyBayerConfig,
	) -> Project:
		"""Create a new Project from an existing project directory."""
		proj = Project(config_type=config_type)
		proj.load_project(project_dir)
		return proj

	@ensure_path
	def load_project(self, project_dir: Path, file_name:str ='config.json'):
		"""Load settings from a project directory - or creates them if initializing for the first time. 
		MUST have a config file in the project directory."""
		self.dir = project_dir
		cfg = Config.from_file(project_dir / file_name)
		cfg_sim = Config.from_file(project_dir / 'sim.json')
		cfg.data['base_simulation'] = cfg_sim.data              #! 20240221: Hacked together to combine the two outputs of template.py
		self._load_config(cfg)

	def _load_config(self, config: Config | dict):
		"""Load and setup optimization from an appropriate JSON config file."""
		
		# Load config file
		if isinstance(config, dict):
			config = self.config_type(config)
		cfg = copy(config)

		try:
			# Setup optimizer
			optimizer: str = cfg.pop('optimizer')
			optimizer_settings: dict = cfg.pop('optimizer_settings')
			try:
				optimizer_type = getattr(sys.modules['vipdopt.optimization'], optimizer)
			except AttributeError:
				raise NotImplementedError(f'Optimizer {optimizer} not currently supported')\
				from None
			self.optimizer = optimizer_type(**optimizer_settings)
		except Exception as e:
			self.optimizer = None

		try:
			# Setup base simulation -
			# Are we running using Lumerical or ceviche or fdtd-z or SPINS or?
			vipdopt.logger.info(f'Loading base simulation from sim.json...')
			base_sim = LumericalSimulation(cfg.pop('base_simulation'))
			#! TODO: IT'S NOT LOADING cfg.data['base_simulation']['objects']['FDTD']['dimension'] properly!
			vipdopt.logger.info('...successfully loaded base simulation!')
		except Exception as e:
			base_sim = LumericalSimulation()
			
		try:
			# TODO: Are we running it locally or on SLURM or on AWS or?
			base_sim.promise_env_setup(
				mpi_exe=cfg['mpi_exe'],
				nprocs=cfg['nprocs'],
				solver_exe=cfg['solver_exe'],
			)
		except Exception as e:
			pass
			
		self.base_sim = base_sim
		self.src_to_sim_map = {
			src: base_sim.with_enabled([src], name=src) for src in base_sim.source_names()
		}
		sims = self.src_to_sim_map.values()

		# General Settings
		iteration = cfg.get('current_iteration', 0)
		epoch = cfg.get('current_epoch', 0)

		# Setup FoMs
		self.foms = []
		#! 20240224 Ian - Took this part out to handle it in main.py. Could move it back later
		#! But some restructuring should be discussed

		# Load device
		try:
			device_source = cfg.pop('device')
			self.device = Device.from_source(device_source)
		except Exception as e:
      
			#* Design Region(s) + Constraints
			# e.g. feature sizes, bridging/voids, permittivity constraints, corresponding E-field monitor regions
			self.device = Device(
				(cfg['device_voxels_simulation_mesh_vertical'], cfg['device_voxels_simulation_mesh_lateral']),
				(cfg['min_device_permittivity'], cfg['max_device_permittivity']),
				(0, 0, 0),
				randomize=True,
				init_seed=0,
				filters=[Sigmoid(0.5, 1.0), Sigmoid(0.5, 1.0)],
			)

		# Setup Folder Structure
		work_dir = self.dir / '.tmp'
		work_dir.mkdir(exist_ok=True, mode=0o777)
		vipdopt_dir = os.path.dirname(__file__)         # Go up one folder
		
		running_on_local_machine = False
		slurm_job_env_variable = os.getenv('SLURM_JOB_NODELIST')
		if slurm_job_env_variable is None:
			running_on_local_machine = True
		
		self.folders = create_internal_folder_structure(self.dir, work_dir, running_on_local_machine)

		# Setup Optimization
		self.optimization = Optimization(
			sims,
			self.device,
			self.optimizer,
			None,				# full_fom
			#will be filled in once full_fom is calculated
			# might need to load the array of foms[] as well...
			# and the weights AS WELL AS the full_fom()
			# and the gradient function
			start_epoch=epoch,
			start_iter=iteration,
			max_epochs=cfg.get('max_epochs', 1),
			iter_per_epoch=cfg.get('iter_per_epoch', 100),
			work_dir=work_dir,
		)

		# TODO Register any callbacks for Optimization here
		#! TODO: Is the optimization accessing plots and histories when being called?

		self.config = cfg

	def get(self, prop: str, default: Any=None) -> Any | None:
		"""Get parameter and return None if it doesn't exist."""
		return self.config.get(prop, default)

	def pop(self, name: str) -> Any:
		"""Pop a parameter."""
		return self.config.pop(name)

	def __getitem__(self, name: str) -> Any:
		"""Get value of a parameter."""
		return self.config[name]

	def __setitem__(self, name: str, value: Any) -> None:
		"""Set the value of an attribute, creating it if it doesn't already exist."""
		self.config[name] = value

	def current_device_path(self) -> Path:
		"""Get the current device path for saving."""
		epoch = self.optimization.epoch
		iteration = self.optimization.iteration
		return self.dir / 'device' / f'e_{epoch}_i_{iteration}.npy'

	def save(self):
		"""Save this project to it's pre-assigned directory."""
		self.save_as(self.dir)

	@ensure_path
	def save_as(self, project_dir: Path):
		"""Save this project to a specified directory, creating it if necessary."""
		(project_dir / 'device').mkdir(parents=True, exist_ok=True)
		self.dir = project_dir
		cfg = self._generate_config()

		self.device.save(self.current_device_path())  # Save device to file for current epoch/iter
		cfg.save(project_dir / 'config.json', cls=LumericalEncoder)


		#! TODO: Save histories and plots

	def _generate_config(self) -> Config:
		"""Create a JSON config for this project's settings."""
		cfg = copy(self.config)

		# Miscellaneous Settings
		cfg['current_epoch'] = self.optimization.epoch
		cfg['current_iteration'] = self.optimization.iteration

		# Optimizer
		cfg['optimizer'] = type(self.optimizer).__name__
		cfg['optimizer_settings'] = vars(self.optimizer)

		# FoMs
		foms = []
		for i, fom in enumerate(self.foms):
			data = fom.as_dict()
			data['weight'] = self.weights[i]
			foms.append(data)
		cfg['figures_of_merit'] = {fom.name: foms[i] for i, fom in enumerate(self.foms)}

		# Device
		cfg['device'] = self.current_device_path()    #! 20240221 Ian: Edited this to match config.json in test_project
		# cfg['device'] = vars(self.device)

		# Simulation
		cfg['base_simulation'] = self.base_sim.as_dict()

		return cfg


if __name__ == '__main__':
	project_dir = Path('./test_project/')
	output_dir = Path('./test_output_optimization/')

	opt = Project.from_dir(project_dir)
	opt.save_as(output_dir)

	# Test that the saved format is loadable

	# # Also this way
