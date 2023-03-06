#
# Parameter file for the Lumerical Processing Sweep
#

import os
import sys
import numpy as np
import json

# This script processes two types of config files.
# 1) The exact config.yaml and cfg_to_params.py that was used for the optimization. Its variables can be replaced or redeclared without 
# issue, and most of them will be replaced by values drawn directly from the .fsp file. But calling this will cover any bases that we have
# potentially missed.
# 2) The sweep_settings.json determines the types of monitors, sweep parameters, and plots that will be generated from the optimization.

# Add filepath to default path
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#! Gets all parameters from config file - explicitly calls it so that the variables pass into this module's globals()
from cfg_to_params_sony import *

print(3)