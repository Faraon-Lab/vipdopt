"""Class for general sources in a simulation."""


import numpy.typing as npt

from vipdopt.simulation import LumericalSimulation


# TODO: Rename this class to Monitor probably
class Source:
    """Class representing the different sources in a simulation.

    A source is defined by the monitor that observes it in the simulation.
    """

    def __init__(self, sim: LumericalSimulation, monitor: str) -> None:
        """Initialize a source."""
        self.sim = sim
        self.monitor = monitor
        self.shape = self.sim.get_field_shape()

    @property
    def e(self) -> npt.ArrayLike:
        """Return the e field produced by this source."""
        return self.sim.get_efield(self.monitor)

    @property
    def h(self) -> npt.ArrayLike:
        """Return the h field produced by this source."""
        return self.sim.get_hfield(self.monitor)

    def transmission_magnitude(self) -> npt.ArrayLike:
        """Return the transmission magnitude of this source."""
        return self.sim.get_transimission_magnitude(self.monitor)
