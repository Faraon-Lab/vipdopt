#
# Parameter file for the Lumerical Processing Sweep
#
# This script processes two types of config files.
# 1) The exact config.yaml and cfg_to_params.py that was used for the optimization. Its variables can be replaced or redeclared without 
# issue, and most of them will be replaced by values drawn directly from the .fsp file. But calling this will cover any bases that we have
# potentially missed.
# 2) The sweep_settings.json determines the types of monitors, sweep parameters, and plots that will be generated from the optimization.
#

import os
import sys
import numpy as np
import json

# Add filepath to default path
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())

#! Gets all parameters from config file - explicitly calls it so that the variables pass into this module's globals()
from configs.global_params import *
# These parameters can and will be selectively overwritten e.g. environment variables like folder locations

#* Logging
# Logger is already initialized through the call to utility.py
logging.info(f'Starting up Lumerical processing params file: {os.path.basename(__file__)}')
logging.debug(f'Current working directory: {os.getcwd()}')

#* Establish folder structure & location
python_src_directory = os.path.dirname(os.path.dirname(__file__))   # Go up one folder

running_on_local_machine = False
slurm_job_env_variable = os.getenv('SLURM_JOB_NODELIST')
if slurm_job_env_variable is None:
	running_on_local_machine = True

accept_params_from_opt_file = True

#* Debug Options
start_from_step = 0                 # 0 if running entire file in one shot  # NOTE: Overwrites optimization config.

#* Environment Variables
# Repeat the call to lumapi filepaths in case something has changed
if running_on_local_machine:	
	lumapi_filepath =  lumapi_filepath_local
else:	
	#! do not include spaces in filepaths passed to linux
	lumapi_filepath = lumapi_filepath_hpc


#* Input Arguments
parser = argparse.ArgumentParser(description=\
					"Takes the config files from the previous optimization, as well as the sweep settings config, and processes them for \
                    Lumerical processing.")
for argument in utility.argparse_universal_args:
	parser.add_argument(argument[0], argument[1], **(argument[2]))
parser.add_argument("-s", "--sweepconfig", help="Sweep Settings (JSON) path", default='configs/sweep_settings.json')
args = parser.parse_args()

#* Load sweep JSON
sweep_filename = args.sweepconfig
sweep_settings = json.load(open(sweep_filename))
logging.debug(f'Read parameters from file: {sweep_filename}')

print(3)