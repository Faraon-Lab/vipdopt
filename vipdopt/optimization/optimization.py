"""Class for handling optimization setup, running, and saving."""

from vipdopt.device import Device
from vipdopt.simulation import LumericalSimulation


class Optimization:
    """Class for orchestrating all the pieces of an optimization."""

    def __init__(
            self,
            sim: LumericalSimulation,
            fom,
            device: Device,
    ):
        """Initialize Optimzation object."""
