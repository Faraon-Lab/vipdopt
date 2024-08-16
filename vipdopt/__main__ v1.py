"""Run the Vipdopt software package."""

import logging
import os
import sys
from argparse import SUPPRESS, ArgumentParser

import numpy as np

# Import main package and add program folder to PATH.
sys.path.append( os.getcwd() )  # 20240219 Ian: Only added this so I could debug some things from my local VSCode
import vipdopt
from vipdopt.utils import import_lumapi, setup_logger

f = sys.modules[__name__].__file__
if not f:
    raise ModuleNotFoundError('SHOULD NEVER REACH HERE')

path = os.path.dirname(f)
path = os.path.join(path, '..')
sys.path.insert(0, path)


from pathlib import Path

from vipdopt.gui import start_gui
from vipdopt.project import Project

if __name__ == '__main__':
    
    # Set up argument parser
    parser = ArgumentParser(
        prog='vipdopt',
        description='Volumetric Inverse Photonic Design Optimizer',
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_const',
        const=True,
        default=False,
        help='Enable verbose output.',
    )
    parser.add_argument(
        '--log', type=Path, default='dev.log', help='Path to the log file.'
    )
    
    # Set up argument subparsers
    subparsers = parser.add_subparsers(help='commands', dest='command')
    opt_parser = subparsers.add_parser('optimize')
    gui_parser = subparsers.add_parser('gui')
    
    # Configure optimizer subparser
    opt_parser.add_argument(
        '-v',
        '--verbose',
        action='store_const',
        const=True,
        default=SUPPRESS,
        help='Enable verbose output.',
    )
    opt_parser.add_argument(
        'directory',
        type=Path,
        help='Project directory to use',
    )
    opt_parser.add_argument(
        '--log', type=Path, default=SUPPRESS, help='Path to the log file.'
    )
    opt_parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Configuration file to use in the optimization; defaults to config.yaml',
    )
    
    args = parser.parse_args()

    # Set up logging
    log_file = args.directory / args.log if args.command == 'optimize' else args.log
    # Set verbosity
    level = logging.DEBUG if args.verbose else logging.INFO
    vipdopt.logger = setup_logger('global_logger', level, log_file=log_file)
    
    # GUI? 
    if args.command == 'gui':
        sys.exit(start_gui([]))
    elif args.command is None:
        print(parser.format_usage())
        sys.exit(1)
    
    
    #
    # * Step 0: Set up simulation conditions and environment. ======================================================================================
    # i.e. current sources, boundary conditions, supporting structures, surrounding regions.
    # Also any other necessary editing of the Lumerical environment and objects.
    vipdopt.logger.info('Beginning Step 0: Project Setup...')

    project = Project()
    project.load_project(args.directory, config_name=args.config)
    # What does the Project class contain?
    # 'dir': directory where it's stored; 'config': SonyBayerConfig object; 'optimization': Optimization object;
    # 'optimizer': Adam/GDOptimizer object; 'device': Device object;
    # 'base_sim': Simulation object; 'src_to_sim_map': dict with source names as keys, Simulation objects as values
    # 'foms': list of FoM objects, 'weights': array of shape (#FoMs, nλ)

    # Now that config is loaded, set up lumapi
    if os.getenv('SLURM_JOB_NODELIST') is None:
        vipdopt.lumapi = import_lumapi(
            project.config.data['lumapi_filepath_local']
        )  # Windows (local machine)
    else:
        vipdopt.lumapi = import_lumapi(
            project.config.data['lumapi_filepath_hpc']
        )  # HPC (Linux)
    
    
    # * Image Import and Processing for Topologically Optimized Metasurface Color Router
    #  https://saturncloud.io/blog/how-to-convert-an-image-to-grayscale-using-numpy-arrays-a-comprehensive-guide/#step-1-import-necessary-libraries
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    img = mpimg.imread('sciadv.adn9000-f2_cropped.jpg')
    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    img = rgb2gray(img)
    
    x_start = 12; x_end = 704; y_start = 13; y_end = 703
    img = 255 - img[x_start:x_end, y_start:y_end]
    def normalize(A):
        return (A-np.min(A))/(np.max(A) - np.min(A))
    img = normalize(img)
    # plt.imshow(img, cmap=plt.get_cmap('gray'))
    
    # Interpolate to a grid of 120x120 - device size 1.2um, resolution 10nm
    from scipy import interpolate
    design_region_x = np.linspace(0,img.shape[0]-1,img.shape[0])
    design_region_y = np.linspace(0,img.shape[1]-1,img.shape[1])
    design_region_import_x = np.linspace(0, img.shape[0]-1, project.device.field_shape[0])
    design_region_import_y = np.linspace(0, img.shape[1]-1, project.device.field_shape[1])
    design_region_import = np.array(
                                np.meshgrid(
                                    design_region_import_x,
                                    design_region_import_y,
                                    indexing='ij'
                                )
                            ).transpose((1, 2, 0))

    img = interpolate.interpn(
        (design_region_x, design_region_y),
        img,
        design_region_import,
        method='linear',
    )
    img = project.device.binarize(img)
    img = np.repeat(img[:, :, np.newaxis], project.device.field_shape[2], axis=2)
    
    project.device.set_design_variable(img)
    # * ==========================================================================================================
    
    
    # base sim is project.base_sim
    # We can create new simulations from the foMs by using fom.create_forward_sim()
    # and passing base sim as a template
    project.base_sim.set_path(project.dir / 'base_sim.fsp')
    # project.base_sim.set_path('test_data\\base_sim.fsp')
    fwd_sim = project.foms[0].create_forward_sim(project.base_sim)[0]
    fwd_sim.set_path(project.dir / 'fwd_sim.fsp')
    
    fdtd = project.optimization.fdtd
    vipdopt.fdtd = fdtd      # Makes it available to global
    fdtd.connect(hide=False) # True)
    
    debug_post_run=True
    if not debug_post_run:
        
        plt.imshow(np.abs(project.device.get_design_variable()[...,0]), cmap=plt.get_cmap('gray'))
        cur_density, cur_permittivity = project.device.import_cur_index(
                    fwd_sim.objects['design_import'],
                    reinterpolation_factors = 1, # project.config.get('reinterpolate_permittivity_factor'),
                    binarize=False
                )
        
        # fdtd.save(project.base_sim.get_path(), project.base_sim)
        # fdtd.save(project.dir / 'base_sim.fsp', project.base_sim)
        fdtd.save(fwd_sim.get_path(), fwd_sim)
        # fdtd.save('test_data\\adj_sim.fsp', adj_sim)
        
        fdtd.addjob(fwd_sim.get_path())
        # fdtd.addjob('test_data\\adj_sim.fsp')
        fdtd.runjobs(1)
        fdtd.reformat_monitor_data([fwd_sim])
    else:
        fdtd.load(fwd_sim.get_path())
        fdtd.reformat_monitor_data([fwd_sim])
    
    # fom_val = project.optimization.fom.foms[1][0].compute_fom()
    # vipdopt.logger.debug(f'FoM: {fom_val.shape}')
    # grad_val = project.optimization.fom.foms[1][0].compute_grad()
    # vipdopt.logger.debug(f'Gradient: {grad_val.shape}')
    
    quad_trans = []
    quad_colors = ['blue','xkcd:grass green','red','xkcd:olive green','black','green']
    quad_labels = ['B','G1','R','G2','Total','G1+G2']
    for idx, num in enumerate([5,4,7,6,8]):
        quad_trans.append(fwd_sim.monitors()[num].trans_mag)
    quad_trans.append(quad_trans[1]+quad_trans[3])

    fig, ax = plt.subplots()
    for idx, num in enumerate([0,2,-1]):
        plt.plot(project.config['lambda_values_um'], quad_trans[num], color=quad_colors[num], label=quad_labels[num])
    for idx, num in enumerate([1,3]):
        plt.plot(project.config['lambda_values_um'], quad_trans[num], linestyle='--', color=quad_colors[num], label=quad_labels[num])

    plt.plot(project.config['lambda_values_um'], quad_trans[-2], color='black', label=f'Overall')
    plt.hlines(0.225, project.config['lambda_values_um'][0], project.config['lambda_values_um'][-1], 
                color='black', linestyle='--',label="22.5%")
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim([0,1])
    plt.xlabel('Wavelength (um)')
    plt.ylabel('Transmission')
    plt.title('PML BCs')
    plt.tight_layout()
    plt.savefig(project.dir / 'quad_trans_aperiodic.png', bbox_inches='tight')
    
    print('Reached end of code.')