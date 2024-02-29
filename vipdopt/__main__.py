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

    # Debug that base_sim is correctly created...
    project.base_sim.connect(license_checked=False)
    project.base_sim.save( project.folders['DATA'] / project.base_sim.info['name'] )
 
    # For each subdevice (design) region, create a field_shape variable to store the E- and H-fields for adjoint calculation.
    field_shapes = []
    # Open simulation files and create subdevice (design) regions as well as attached meshes and monitors.
    field_shape = project.base_sim.get_field_shape()
    # correct_field_shape = ( project.config.data['device_voxels_simulation_mesh_lateral_bordered'],
    #                 		project.config.data['device_voxels_simulation_mesh_vertical']
    #                 	)
    # assert field_shape == correct_field_shape, f"The sizes of field_shape {field_shape} should instead be {correct_field_shape}."
    field_shapes.append(field_shape)
    project.device.field_shape = field_shape

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
    
    license_checked = False
    for sim_name, sim in project.src_to_sim_map.items():
        vipdopt.logger.info(f'Creating sim with enabled: {sim_name}')
        sim.connect(license_checked=True)
        if not license_checked:
            license_checked = True
        # sim.setup_env_resources()
        sim.save( os.path.abspath( project.folders['DATA'] / sim.info['name'] ) )
        # # sim.close()

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
    
    #! 20240228 Ian - We need to store these functions somewhere and they take over the _math_helper() of FoM because that's not working right now
    #! They could go into project.py or optimization.py ?
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

        final_weights = np.array(weights)

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
        
        if final_weights.shape != performance_weighting.shape:
            performance_weighting = performance_weighting[..., np.newaxis]
            
        final_weights = final_weights * performance_weighting #[..., np.newaxis]

        device_gradient = overall_combine_function( indiv_foms, final_weights, 'gradient' )
        return device_gradient

    vipdopt.logger.info("Completed Step 2: Figure of Merit setup complete.")
    # utility.backup_all_vars(globals(), cfg.cv.shelf_fn)

    #
    # Step 3
    #
    vipdopt.logger.info('Beginning Step 3: Run Optimization')
    
    # Update optimization with fom_function
    project.optimization.foms = project.foms            # List of FoM objects
    project.optimization.weights = project.weights       # List of weights corresponding to above
    project.optimization.fom = full_fom
    project.optimization.fom_func = overall_figure_of_merit
    project.optimization.grad_func = overall_adjoint_gradient
    
    project.optimization.run()

    vipdopt.logger.info('Completed Step 3: Run Optimization')

    #
    # Cleanup
    #
    # for i in range(len(all_sims)):
