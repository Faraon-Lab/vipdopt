"""Package for optimization and inverse design of volumetric photonic devices."""

# Import lumerical
import logging
from getpass import getuser
from types import ModuleType
from typing import Any

from vipdopt.utils import import_lumapi

fdtd: Any = None
logger: logging.Logger = logging.getLogger()

try:
    lumapi: ModuleType | None = import_lumapi(
        f'/central/home/{getuser()}/lumerical/v232/api/python/lumapi.py'
    )
except FileNotFoundError:
    # logger.exception('lumapi not found. Using dummy values\n')
    lumapi = None
