"""Class for general sources in a simulation."""


import numpy.typing as npt

from vipdopt.simulation import LumericalSimulation


class Monitor:
    """Class representing the different source monitors in a simulation."""

    def __init__(self, sim: LumericalSimulation, source_name: str, monitor_name: str) -> None:
        """Initialize a Monitor."""
        self.sim = sim
        self.source_name = source_name
        self.monitor_name = monitor_name

        # Initialize field values
        self._shape = None
        self._e = None
        self._h = None
        self._trans_mag = None
    
    @property
    def shape(self) -> npt.ArrayLike:
        """Return the shape of the numpy array for this monitor's fields."""
        if self._shape is None:
            if self.sim.fdtd is not None:
                self._shape = self.sim.get_field_shape()
        return self._shape

    @property
    def e(self) -> npt.ArrayLike:
        """Return the e field produced by this source."""
        if self._e is None:
            self._e = self.sim.get_efield(self.monitor_name)
        return self._e

    @property
    def h(self) -> npt.ArrayLike:
        """Return the h field produced by this source."""
        if self._h is None:
            self._h = self.sim.get_hfield(self.monitor_name)
        return self._h

    @property
    def trans_mag(self) -> npt.ArrayLike:
        """Return the transmission magnitude of this source."""
        if self._trans_mag is None:
            self._trans_mag = self.sim.get_transimission_magnitude(self.monitor_name)
        return self._trans_mag
