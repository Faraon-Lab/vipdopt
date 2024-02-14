"""Package for optimization and inverse design of volumetric photonic devices."""

# Import lumerical
from getpass import getuser
import logging

from vipdopt.utils import import_lumapi, setup_logger

lumapi = import_lumapi(f'/central/home/{getuser()}/lumerical/v232/api/python/lumapi.py')
logger: logging.Logger = logging.getLogger()
