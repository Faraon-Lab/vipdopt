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
from vipdopt.optimization.filter import Sigmoid, Scale
from vipdopt.simulation import LumericalEncoder, LumericalSimulation
from vipdopt.utils import PathLike, ensure_path, glob_first


def create_internal_folder_structure(root_dir: Path,debug_mode=False):
    # global DATA_FOLDER, SAVED_SCRIPTS_FOLDER, OPTIMIZATION_INFO_FOLDER, OPTIMIZATION_PLOTS_FOLDER
    # global DEBUG_COMPLETED_JOBS_FOLDER, PULL_COMPLETED_JOBS_FOLDER, EVALUATION_FOLDER, EVALUATION_CONFIG_FOLDER, EVALUATION_UTILS_FOLDER
    
    directories: dict[str, Path] = {'main': root_dir}

    #* Output / Save Paths 
    data_folder = root_dir / 'data'
    temp_folder = root_dir / '.tmp'
    checkpoint_folder = data_folder / 'checkpoints'
    saved_scripts_folder = data_folder / 'saved_scripts'
    optimization_info_folder = data_folder / 'opt_info'
    optimization_plots_folder = optimization_info_folder / 'plots'
    debug_completed_jobs_folder = temp_folder / 'ares_test_dev'
    pull_completed_jobs_folder = data_folder
    if debug_mode:
        pull_completed_jobs_folder = debug_completed_jobs_folder

    evaluation_folder = root_dir / 'eval'
    evaluation_config_folder = evaluation_folder / 'configs'
    evaluation_utils_folder = evaluation_folder / 'utils'

    # parameters['MODEL_PATH'] = DATA_FOLDER / 'model.pth'
    # parameters['OPTIMIZER_PATH'] = DATA_FOLDER / 'optimizer.pth'


    # #* Save out the various files that exist right before the optimization runs for debugging purposes.
    # # If these files have changed significantly, the optimization should be re-run to compare to anything new.

    try:
        shutil.copy2( root_dir + "/slurm_vis10lyr.sh", saved_scripts_folder + "/slurm_vis10lyr.sh" )
    except Exception as ex:
        pass
    # shutil.copy2( cfg.python_src_directory + "/SonyBayerFilterOptimization.py", SAVED_SCRIPTS_FOLDER + "/SonyBayerFilterOptimization.py" )
    # shutil.copy2( os.path.join(python_src_directory, yaml_filename), 
    # shutil.copy2( python_src_directory + "/configs/SonyBayerFilterParameters.py", SAVED_SCRIPTS_FOLDER + "/SonyBayerFilterParameters.py" )
    # # TODO: et cetera... might have to save out various scripts from each folder

    #  Create convenient folder for evaluation code
    # if not os.path.isdir( evaluation_folder ):
    #     evaluation_folder.mkdir(exist_ok=True)
    
    
    # TODO: finalize this when the Project directory internal structure is finalized    
    # if os.path.exists(EVALUATION_CONFIG_FOLDER):
    #     shutil.rmtree(EVALUATION_CONFIG_FOLDER)
    # shutil.copytree(os.path.join(main_dir, "configs"), EVALUATION_CONFIG_FOLDER)
    # if os.path.exists(EVALUATION_UTILS_FOLDER):
    #     shutil.rmtree(EVALUATION_UTILS_FOLDER)
    # shutil.copytree(os.path.join(main_dir, "utils"), EVALUATION_UTILS_FOLDER)

    #shutil.copy2( os.path.abspath(python_src_directory + "/evaluation/plotter.py"), EVALUATION_UTILS_FOLDER + "/plotter.py" )
    
    

    directories = {
        'root': root_dir,
        'data': data_folder,
        'saved_scripts': saved_scripts_folder,
        'opt_info': optimization_info_folder,
        'opt_plots': optimization_plots_folder,
        'pull_completed_jobs': pull_completed_jobs_folder,
        'debug_completed_jobs': debug_completed_jobs_folder,
        'evaluation': evaluation_folder,
        'eval_config': evaluation_config_folder,
        'eval_utils': evaluation_utils_folder,
        'checkpoints': checkpoint_folder,
        'temp': temp_folder,
    }
    for d in directories.values():
        d.mkdir(exist_ok=True, mode=0o777, parents=True)  # Create missing directories with full perms
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
        self.foms = []
        self.weights = []
        self.subdirectories: dict[str, Path] = {}

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
    def load_project(self, project_dir: Path, file_name:str ='processed_config.yaml'):
        """Load settings from a project directory - or creates them if initializing for the first time. 
        MUST have a config file in the project directory."""
        self.dir = project_dir
        cfg_file = project_dir / file_name
        if not cfg_file.exists():
            # Search the directory for a configuration file
            cfg_file = glob_first(project_dir, '**/*config*.{json,yaml,yml}')
        cfg = Config.from_file(cfg_file)
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
        
        # TODO: Are we running it locally or on SLURM or on AWS or?
        base_sim.promise_env_setup(
            mpi_exe=cfg['mpi_exe'],
            nprocs=cfg['nprocs'],
            solver_exe=cfg['solver_exe'],
            nsims=len(base_sim.source_names())
        )
            
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
        for name, fom_dict in cfg.pop('figures_of_merit').items():
            self.weights.append(fom_dict['weight'])
            self.foms.append(FoM.from_dict(name, fom_dict, self.src_to_sim_map))
        full_fom = sum(np.multiply(self.weights, self.foms))
        
        # Overall Weights for each FoM
        self.weights = np.array(self.weights)
        
        # Set up SPECTRAL weights - Wavelength-dependent behaviour of each FoM (e.g. spectral sorting)
        # spectral_weights_by_fom = np.zeros((len(fom_dict), cfg.pv.num_design_frequency_points))
        spectral_weights_by_fom = np.zeros((cfg['num_bands'], cfg['num_design_frequency_points']))
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

            # Plotting code to check weighting shapes
            import matplotlib.pyplot as plt
            plt.vlines(wl_band_bound_idxs['left'], 0,1, 'b','--')
            plt.vlines(wl_band_bound_idxs['right'], 0,1, 'r','--')
            plt.vlines(wl_band_bound_idxs['peak'], 0,1, 'k','-')
            for fom in spectral_weights_by_fom:
            	plt.plot(fom)
            plt.show()
            print(3)
        
            return spectral_weights_by_fom

        wl_band_bounds = assign_bands(wl_band_bounds, cfg['lambda_values_um'], cfg['num_bands'])
        vipdopt.logger.info(f'Desired peak locations per band are: {wl_band_bounds["peak"]}')

        # Convert to the nearest matching indices of lambda_values_um
        wl_band_bound_idxs = wl_band_bounds.copy()
        for key, val in wl_band_bounds.items():
            wl_band_bound_idxs[key] = np.searchsorted(cfg['lambda_values_um'], val)

        spectral_weights_by_fom = determine_spectral_weights(spectral_weights_by_fom, wl_band_bound_idxs, mode='gaussian') # args, kwargs
        # Repeat green weighting for other FoM such that green weighting applies to FoMs 1,3
        if self.config['simulator_dimension'] == '3D':      #! Bayer Filter Functionality
            spectral_weights_by_fom = np.insert(spectral_weights_by_fom, 3, spectral_weights_by_fom[1,:], axis=0)

        self.weights = self.weights[..., np.newaxis] * spectral_weights_by_fom


        # Load device
        try:
            device_source = cfg.pop('device')
            self.device = Device.from_source(device_source)
        except Exception as e:
      
            #* Design Region(s) + Constraints
            # e.g. feature sizes, bridging/voids, permittivity constraints, corresponding E-field monitor regions
            region_coordinates = {'x': np.linspace(-0.5 * cfg['device_size_lateral_bordered_um'], 0.5 * cfg['device_size_lateral_bordered_um'], cfg['device_voxels_lateral_bordered'])}
            if cfg['simulator_dimension'] == '2D':
                voxel_array_size = ( cfg['device_voxels_lateral_bordered'], cfg['device_voxels_vertical'], 3 )
                region_coordinates.update({
                    'y': np.linspace(cfg['device_vertical_minimum_um'], cfg['device_vertical_maximum_um'], cfg['device_voxels_vertical']),
                    'z': np.linspace(-0.5 * cfg['mesh_spacing_um'] * 1e-6, 0.5 * cfg['mesh_spacing_um'] * 1e-6, 3)
                })
            elif cfg['simulator_dimension'] == '3D':
                voxel_array_size = ( cfg['device_voxels_lateral_bordered'], cfg['device_voxels_lateral_bordered'], cfg['device_voxels_vertical'] )
                region_coordinates.update({
                    'y': np.linspace(-0.5 * cfg['device_size_lateral_bordered_um'], 0.5 * cfg['device_size_lateral_bordered_um'], cfg['device_voxels_lateral_bordered']),
                    'z': np.linspace(cfg['device_vertical_minimum_um'], cfg['device_vertical_maximum_um'], cfg['device_voxels_vertical'])
                })
                
            self.device = Device(
                voxel_array_size,       # (cfg['device_voxels_simulation_mesh_vertical'], cfg['device_voxels_simulation_mesh_lateral']),
                (cfg['min_device_permittivity'], cfg['max_device_permittivity']),
                region_coordinates,
                randomize=False,            #! 20240228 Ian - Changed this
                init_seed=0,
                filters=[Sigmoid(0.5, 1.0), Scale((cfg['min_device_permittivity'], cfg['max_device_permittivity']))],
            )

        # Setup Folder Structure
        work_dir = self.dir / '.tmp'
        work_dir.mkdir(exist_ok=True, mode=0o777)
        vipdopt_dir = os.path.dirname(__file__)         # Go up one folder
        
        running_on_local_machine = False
        slurm_job_env_variable = os.getenv('SLURM_JOB_NODELIST')
        if slurm_job_env_variable is None:
            running_on_local_machine = True
        
        self.subdirectories = create_internal_folder_structure(self.dir, running_on_local_machine)

        # Setup Optimization
        self.optimization = Optimization(
            sims,
            self.device,
            self.optimizer,
            full_fom,                # full_fom
            #will be filled in once full_fom is calculated
            # might need to load the array of foms[] as well...
            # and the weights AS WELL AS the full_fom()
            # and the gradient function
            cfg,
            start_epoch=epoch,
            start_iter=iteration,
            max_epochs=cfg.get('max_epochs', 1),
            iter_per_epoch=cfg.get('iter_per_epoch', 100),
            env_vars=self.base_sim._env_vars.copy(),
            dirs=self.subdirectories,
        )

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
