"""Package for abstracting interactions with the simulations."""

from vipdopt.simulation.lumfdtd import ISolver, LumericalFDTD
from vipdopt.simulation.monitor import Monitor, Power, Profile
# from vipdopt.simulation.lumfdtdsimobject import (
#     Import,
#     LumericalEncoder,
#     LumericalSimObject,
#     LumericalSimObjectType,
# )
from vipdopt.simulation.simobject import (
    Import,
    SimEncoder,
    SimObject,
    SimObjectType,
)
from vipdopt.simulation.simulation import (
    ISimulation,
    Simulation,
    LumericalSimulation,
)
from vipdopt.simulation.source import DipoleSource, GaussianSource, Source, TFSFSource, PlaneSource

__all__ = [
    'ISimulation',
    'LumericalSimObject',
    'LumericalSimObjectType',
    'LumericalSimulation',
    'LumericalEncoder',
    'Monitor',
    'Power',
    'Profile',
    'Source',
    'DipoleSource',
    'GaussianSource',
    'TFSFSource',
    'PlaneSource',
    'Import',
    'ISolver',
    'LumericalFDTD',
]
