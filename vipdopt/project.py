
from __future__ import annotations

import os
import sys
import contextlib
import shutil
import copy
import glob
from pathlib import Path
from typing import Any

sys.path.append(Path.cwd())
import vipdopt
from vipdopt.configuration import Config, ProjectConfig
from vipdopt.eval import Evaluation
from vipdopt.optimization import (
    Device,
    # FoM,
    GradientOptimizer,
    Optimization,
    # LumericalOptimization,
    # SuperFoM,
)
from vipdopt.simulation import SimEncoder, Simulation
from vipdopt.simulation.lumfdtd import LumericalFDTD
from vipdopt.utils import PathLike, ensure_path, glob_first, read_config_file

def create_internal_folder_structure(root_dir: Path, pull_files_debug_mode=False):
    """Create the subdirectories of the project folder.
    pull_files_debug_mode: if True, sets the folder for pulling completed jobs to the debug folder which should contain already-run sims.
    This is for avoiding simulation runtime during debug testing.
    """
    directories: dict[str, Path] = {'main': root_dir}

    # * Output / Save Paths
    data_folder = root_dir / 'data'
    summary_folder = root_dir / 'summary'
    temp_folder = root_dir / '.tmp'
    device_folder = root_dir / 'device'
    checkpoint_folder = data_folder / 'checkpoints'
    saved_scripts_folder = data_folder / 'saved_scripts'
    optimization_info_folder = data_folder / 'opt_info'
    optimization_plots_folder = optimization_info_folder / 'plots'
    debug_completed_jobs_folder = root_dir / 'ares_test_dev'
    pull_completed_jobs_folder = temp_folder
    if pull_files_debug_mode:
        pull_completed_jobs_folder = debug_completed_jobs_folder

    evaluation_folder = root_dir / 'eval'
    evaluation_config_folder = evaluation_folder / 'configs'
    evaluation_info_folder = evaluation_folder / 'opt_info'
    evaluation_utils_folder = evaluation_folder / 'utils'
    evaluation_temp_folder = evaluation_folder / '.tmp'

    # parameters['MODEL_PATH'] = DATA_FOLDER / 'model.pth'
    # parameters['OPTIMIZER_PATH'] = DATA_FOLDER / 'optimizer.pth'

    # Save out the various files that exist right before the optimization runs for
    # debugging purposes. If these files have changed significantly, the optimization
    # should be re-run to compare to anything new.

    with contextlib.suppress(Exception):
        for file in list(glob.glob('*.sh')):
            shutil.copy2(root_dir/file, saved_scripts_folder/file)

    # shutil.copy2(
    # cfg.python_src_directory + "/SonyBayerFilterOptimization.py",
    # SAVED_SCRIPTS_FOLDER + "/SonyBayerFilterOptimization.py" )
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

    # shutil.copy2(
    #     os.path.abspath(python_src_directory + "/evaluation/plotter.py"),
    #     EVALUATION_UTILS_FOLDER + "/plotter.py" )

    directories = {
        'root': root_dir,
        'data': data_folder,
        'summary': summary_folder,
        'saved_scripts': saved_scripts_folder,
        'opt_info': optimization_info_folder,
        'opt_plots': optimization_plots_folder,
        'pull_completed_jobs': pull_completed_jobs_folder,
        'debug_completed_jobs': debug_completed_jobs_folder,
        'device': device_folder,
        'evaluation': evaluation_folder,
        'eval_config': evaluation_config_folder,
        'eval_info': evaluation_info_folder,
        'eval_utils': evaluation_utils_folder,
        'eval_temp': evaluation_temp_folder,
        'checkpoints': checkpoint_folder,
        'temp': temp_folder,
    }
    # Create missing directories with full permissions
    for d in directories.values():
        d.mkdir(exist_ok=True, mode=0o777, parents=True)
    return directories

class Project:
    """Class for managing the loading and saving of projects."""

    def __init__(self, config_type:type[Config] = Config) -> None:
        """Initialize a Project."""

        """Initialize a Project."""
        self.dir = Path('.')  # Project directory; defaults to root
        self.config = config_type()
        self.config_type = config_type

        self.device: Device | None = None
        self.base_sim: Simulation | None = None
        self.src_to_sim_map: dict[str, Simulation] = {}
        # self.foms: list[FoM] = []
        # self.weights: npt.NDArray | list = []
        self.subdirectories: dict[str, Path] = {}

        self.optimizations: list[Optimization] = []
        self.evaluations: list[Evaluation] = []

    @classmethod
    def from_dir(
        cls: type[Project],
        project_dir: PathLike,
        config_type: type[Config] = Config,
    ) -> Project:
        """Create a new Project from an existing project directory."""
        proj = Project(config_type=config_type)
        proj.load_project(project_dir)
        return proj

    @ensure_path
    def load_project(self, project_dir: Path,
                     project_name: str = 'project.json', config_name: str = 'config.json',
                     override_dir: bool=True):
        """Load settings from a project directory - or create them if initializing.

        MUST have a config file in the project directory.
        """

        self.dir = project_dir

        project_save_file = project_dir / project_name
        cfg_file = project_dir / config_name

        if project_save_file.exists():          # Restarting from save
            vipdopt.logger.info("Found project savefile. Loading settings.")
            project_save = read_config_file(project_save_file)
            if override_dir:
                self.dir = Path(project_save['dir'])
            cfg_file = cfg_file.with_suffix('.json')    # Reload the JSON (save) instead of the YAML (initialise) file.

        if not cfg_file.exists():
            # Search the directory for a configuration file
            cfg_file = glob_first(project_dir, '**/*config*.{yaml,yml,json}')
        cfg = Config.from_file(cfg_file)

        # Append simulation data from sim.json to the config from config.json
        if not project_save_file.exists():      # Initializing
            cfg_sim = Config.from_file(self.dir / 'sim.json')
            cfg.data['base_simulation'] = cfg_sim.data

        self._load_config(cfg)

    def _load_config(self, config: Config | dict):
        """Load and setup optimization from an appropriate config file."""

        # Load config file
        cfg = copy.copy(config)
        if not isinstance(config, Config):
            cfg = self.config_type(cfg)
        assert isinstance(cfg, Config)

        # Setup Folder Structure
        self.manager = 'LOCAL'
        slurm_job_env_variable = os.getenv('SLURM_JOB_NODELIST')
        if slurm_job_env_variable is not None:
            self.manager = 'SLURM'

        self.subdirectories = create_internal_folder_structure(
            self.dir,
            pull_files_debug_mode=cfg.get('pull_sim_files_from_debug_folder')
        )
        vipdopt.logger.info('Internal folder substructure created.')

        # Load Base Simulation
        self.base_sim, self.src_to_sim_map = Simulation._load_from_config(cfg, self.dir,
                                                                         solver=cfg.get('solver_name', 'LumericalFDTD'))
        sims = list(self.src_to_sim_map.values())

        ### Some stuff

        # Load Device
        self.device = Device.load_config(cfg)

        # General (Other) Settings
        iteration = cfg.get('current_iteration', 0)

        self.config = cfg

    @classmethod
    def load_optimizer(cls, cfg: Config):
        """Load the optimizer from a config."""
        optimizer: str = cfg.pop('optimizer', None)
        if optimizer is None:
            vipdopt.logging.warning('No optimizer declared in config.')
            optimizer = 'NLOptOptimizer'
        optimizer_settings: dict = cfg.pop('optimizer_settings', {})
        try:
            optimizer_type = getattr(sys.modules['vipdopt.optimization'], optimizer)
        except AttributeError:
            raise NotImplementedError(
                f'Optimizer {optimizer} not currently supported'
            ) from None

        return optimizer_type(**optimizer_settings)

    def save(self):
        """Save this project to it's pre-assigned directory."""
        self.save_as(self.dir)

    #! TODO:
    @ensure_path
    def save_as(self, project_dir: Path):
        """Save this project to a specified directory, creating it if necessary."""
        # NOTE: Created this according to vars(project) after creating a fresh project.

        # This dictionary will be saved as a YAML/JSON and stores all the variables that aren't class objects.
        # Coding each key-value pair manually so as to be careful.
        proj_cfg = ProjectConfig()

        # Dir
        proj_cfg.update({'dir': self.dir})

        # Optimization:

        # Optimizer: Handled below in generate_config()

        # Device: Handled below in _generate_config()
        # assert self.device is not None
        # self.device.save(self.subdirectories['device'])

        # Base Sim: Handled below in _generate_config()
        # src_to_sim_map: Handled in _load_config()
        # FoMs: Handled below in _generate_config()
        # Weights: Handled in _load_config()
        # Subdirectories: Handled in _load_config()
        # Spectral weights: Handled in _load_config()

        # Config
        cfg = self._generate_config()
        cfg.save(project_dir / 'config.json', cls=SimEncoder)
        # Config Type:
        proj_cfg.update({'config_type': self.config_type.__name__})

        proj_cfg.save(project_dir / 'project.json', cls=SimEncoder)

    def _generate_config(self) -> Config:
        """Create a JSON config for this project's settings."""
        cfg = copy.copy(self.config)

        for i, opt in enumerate(self.optimizations):
            opt_dict = {}

            assert opt is not None
            assert opt.optimizer is not None
            assert opt.base_sim is not None

            # Miscellaneous Settings
            opt_dict['current_iteration'] = opt.iteration

            # Optimizer
            opt_dict['optimizer'] = type(opt.optimizer).__name__
            opt_dict['optimizer_settings'] = vars(opt.optimizer)

            # FoMs
            opt_dict['figures_of_merit'] = {}

            foms = []
            for j, fom in enumerate(opt.fom.foms):
                data = fom[0].as_dict()
                data['weight'] = opt.fom.weights[j]
                foms.append(data)
            opt_dict['figures_of_merit'].update({f'fom_{k}': foms[k] for k, _ in enumerate(opt.fom.foms)})

            # Device
            opt_dict['device'] = opt.current_device_path()

            # Base Simulation
            opt_dict['base_simulation'] = opt.base_sim.as_dict()

            cfg[f'opt_{i}'] = opt_dict

        return cfg

    def start_all_optimizations(self):
        for opt in self.optimizations:
            self.start_optimization(opt)

    def start_optimization(self, opt):
        """Start this project's optimization."""
        opt.loop = True
        opt.run()

    def stop_optimization(self, opt):
        """Stop this project's optimization."""
        idx = self.optimizations.index(opt)
        vipdopt.logger.info('stopping optimization early')
        self.optimization.loop = False

    def stop_all_optimizations(self):
        for opt in self.optimizations:
            self.stop_optimization(opt)