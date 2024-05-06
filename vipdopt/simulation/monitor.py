"""Class for general sources in a simulation."""

import numpy.typing as npt

from vipdopt.simulation.simobject import LumericalSimObject, LumericalSimObjectType
from vipdopt.simulation.simulation import LumericalSimulation


class Monitor(LumericalSimObject):
    """Class representing the different source monitors in a simulation."""

    def __init__(
        self,
        name: str,
        type: LumericalSimObjectType,
        sim: LumericalSimulation | None=None,
    ) -> None:
        """Initialize a Monitor."""
        super().__init__(name, type)
        if sim is not None:
            self.set_sim(sim)

    def set_sim(self, sim: LumericalSimulation):
        """Set which sim this monitor is connected to."""
        self.sim = sim
        self.reset()

    def __eq__(self, __value: object) -> bool:
        """Test equality."""
        if isinstance(__value, Monitor):
            return self.sim == __value.sim and self.name == __value.name
        return super().__eq__(__value)

    def reset(self):
        """Reset all cached values from the simulation."""
        self._tshape = None  # transmission shape
        self._fshape = None  # field shape
        self._e = None
        self._h = None
        self._trans_mag = None

    @property
    def tshape(self) -> tuple[int, ...]:
        """Return the shape of the numpy array for this monitor's fields."""
        if self._tshape is None and self.sim.fdtd is not None:
            self._tshape = self.sim.get_transmission_shape(self.monitor_name)
        return self._tshape

    @property
    def fshape(self) -> tuple[int, ...]:
        """Return the shape of the numpy array for this monitor's fields."""
        if self._fshape is None and self.sim.fdtd is not None:
            self._fshape = self.e.shape
        return self._fshape

    @property
    def e(self) -> npt.NDArray:
        """Return the e field measured by this monitor."""
        if self._e is None:
            self._e = self.sim.get_efield(self.monitor_name)
        return self._e

    @property
    def h(self) -> npt.NDArray:
        """Return the h field measured by this monitor."""
        if self._h is None:
            self._h = self.sim.get_hfield(self.monitor_name)
        return self._h

    @property
    def trans_mag(self) -> npt.NDArray:
        """Return the transmission magnitude measured by this monitor."""
        if self._trans_mag is None:
            self._trans_mag = self.sim.get_transmission_magnitude(self.monitor_name)
        return self._trans_mag


class Proflie(Monitor):
    def __init__(
            self,
            name: str,
            sim: LumericalSimulation | None = None,
    ) -> None:
        super().__init__(name, LumericalSimObjectType.PROFILE, sim)


class Power(Monitor):
    def __init__(
            self,
            name: str,
            sim: LumericalSimulation | None = None,
    ) -> None:
        super().__init__(name, LumericalSimObjectType.POWER, sim)


class Index(Monitor):
    def __init__(
            self,
            name: str,
            sim: LumericalSimulation | None = None,
    ) -> None:
        super().__init__(name, LumericalSimObjectType.INDEX, sim)
