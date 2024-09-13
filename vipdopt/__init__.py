"""Package for optimization and inverse design of volumetric photonic devices."""

__version__ = '2.0.1'

# Import lumerical
import os
import logging
from configparser import ConfigParser
from pathlib import Path
from types import ModuleType
from typing import Any

from vipdopt.utils import import_lumapi

LUMERICAL_LOCATION_FILE = Path(__file__).parent / 'lumerical.cfg'
fdtd: Any = None
logger: logging.Logger = logging.getLogger()

cfg = ConfigParser()
cfg.read(LUMERICAL_LOCATION_FILE)

# Now that config is loaded, set up lumapi
if os.getenv('SLURM_JOB_NODELIST') is None:
    lumapi_path = cfg['Lumerical'].get('lumapi_path_local', None)   # Windows (local machine)
else:
    lumapi_path = cfg['Lumerical'].get('lumapi_path_hpc', None)     # SLURM HPC (Linux)
if lumapi_path is None:
    try:
        lumapi_path = cfg['Lumerical']['lumapi_path']
    except Exception as err:
        logger.info(f'Lumapi path not found. Check {LUMERICAL_LOCATION_FILE}')

lumapi: ModuleType = import_lumapi(lumapi_path)
