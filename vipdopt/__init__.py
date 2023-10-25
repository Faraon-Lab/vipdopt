"""Package for optimization and inverse design of volumetric photonic devices."""


# Import lumerical
from getpass import getuser

from vipdopt.utils import import_lumapi

lumapi = import_lumapi(f'/central/home/{getuser()}/lumerical/v232/api/python/lumapi.py')
