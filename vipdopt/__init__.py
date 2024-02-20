"""Package for optimization and inverse design of volumetric photonic devices."""

# Import lumerical
import logging
from getpass import getuser
from types import ModuleType

from vipdopt.utils import import_lumapi

lumapi: ModuleType = import_lumapi(
    f'/central/home/{getuser()}/lumerical/v232/api/python/lumapi.py'
)
logger: logging.Logger = logging.getLogger()
