"""Package for optimization and inverse design of volumetric photonic devices."""

__version__ = '2.0.1'

# Import lumerical
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
lumapi_path = cfg['Lumerical']['lumapi_path']

lumapi: ModuleType = import_lumapi(lumapi_path)
