"""Class for general sources in a simulation."""

from pathlib import Path

import numpy as np
import numpy.typing as npt

import vipdopt
from vipdopt.simulation.simobject import (
    MONITOR_TYPES,
    LumericalSimObject,
    LumericalSimObjectType,
)
from vipdopt.utils import ensure_path


class Monitor(LumericalSimObject):
    """Class representing the different source monitors in a simulation."""

    def __init__(
        self,
        name: str,
        obj_type: LumericalSimObjectType,
        src: Path | None = None,
    ) -> None:
        """Initialize a Monitor."""
        if obj_type not in MONITOR_TYPES:
            raise ValueError(
                f'Cannot create a Monitor of type {obj_type}; permitted choices are '
                f'"{MONITOR_TYPES}"'
            )
        super().__init__(name, obj_type)
        self.src = src
        self.reset()

    @ensure_path
    def set_src(self, src: Path):
        """Set the source file this monitor is connected to."""
        self.src = src
        self.reset()

    def __eq__(self, __value: object) -> bool:
        """Test equality."""
        if isinstance(__value, Monitor):
            return self.src == __value.src and self.name == __value.name
        return super().__eq__(__value)

    def reset(self):
        """Reset all cached values from the simulation."""
        self._tshape = None  # transmission shape
        self._fshape = None  # field shape
        self._e = None  # E field
        self._h = None  # H field
        self._p = None  # Poynting vector
        self._t = None  # Transmission
        self._sp = None  # Source Power
        self._power = None  # Power

        self._sync = self.src is not None  # Only set to sync if the source file exists

    def load_source(self):
        """Load the monitor's data from its source file."""
        if self.src is None:
            raise RuntimeError(f'Monitor {self} has no source to load data from.')
        vipdopt.logger.debug(f'Loading monitor data from {self.src} into memory...')
        data: npt.NDArray = np.load(self.src, allow_pickle=True)
        self._e = data['e']
        self._h = data['h']
        self._p = data['p']
        self._t = data['t']
        self._sp = data['sp']
        self._power = data['power']
        self._tshape = self._t.shape
        self._fshape = self._e.shape

        self._sync = False  # Don't need to sync anymore

    @property
    def tshape(self) -> tuple[int, ...]:
        """Return the shape of the numpy array for this monitor's transmission."""
        if self._sync:
            self.load_source()
        return self._tshape

    @property
    def fshape(self) -> tuple[int, ...]:
        """Return the shape of the numpy array for this monitor's fields."""
        if self._sync:
            self.load_source()
        return self._fshape

    @property
    def e(self) -> npt.NDArray:
        """Return the E field measured by this monitor."""
        if self._sync:
            self.load_source()
        return self._e

    @property
    def h(self) -> npt.NDArray:
        """Return the H field measured by this monitor."""
        if self._sync:
            self.load_source()
        return self._h

    @property
    def p(self) -> npt.NDArray:
        """Return the Poynting vector measured by this monitor."""
        if self._sync:
            self.load_source()
        return self._p

    @property
    def sp(self) -> npt.NDArray:
        """Return the source power measured by this monitor."""
        if self._sync:
            self.load_source()
        return self._sp

    @property
    def t(self) -> npt.NDArray:
        """Return the transmission measured by this monitor."""
        if self._sync:
            self.load_source()
        return self._t

    @property
    def power(self) -> npt.NDArray:
        """Return the transmission measured by this monitor."""
        if self._sync:
            self.load_source()
        return self._power

    @property
    def trans_mag(self) -> npt.NDArray:
        """Return the transmission magnitude measured by this monitor."""
        if self._sync:
            self.load_source()
        return np.abs(self._t)


class Profile(Monitor):
    def __init__(
        self,
        name: str,
        src: Path | None = None,
    ) -> None:
        super().__init__(name, LumericalSimObjectType.PROFILE, src)


class Power(Monitor):
    def __init__(
        self,
        name: str,
        src: Path | None = None,
    ) -> None:
        super().__init__(name, LumericalSimObjectType.POWER, src)


# class Index(Monitor):
#     def __init__(
#         self,
#         name: str,
#         src: Path | None = None,
#     ) -> None:
#         super().__init__(name, LumericalSimObjectType.INDEX, src)
