"""Package for optimization and inverse design of volumetric photonic devices."""

# Import lumerical
import logging
from getpass import getuser
from types import ModuleType
from typing import Any

from vipdopt.utils import import_lumapi

lumapi: ModuleType = import_lumapi(
    f'/central/home/{getuser()}/lumerical/v232/api/python/lumapi.py'
)
fdtd: Any
logger: logging.Logger = logging.getLogger()
