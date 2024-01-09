"""Class for general sources in a simulation."""

from typing import Any

import numpy as np
import numpy.typing as npt

from vipdopt.simulation import LumericalSimulation

class Source:

    def __init__(self, sim: LumericalSimulation, monitor: str) -> None:
        self.sim = sim
        self.monitor = monitor
        self.shape = self.sim.get_field_shape()

    def E(self) -> npt.ArrayLike:
        return self.sim.get_efield(self.monitor)

    def H(self) -> npt.ArrayLike:
        return self.sim.get_hfield(self.monitor)
    
    def transmission_magnitude(self) -> npt.ArrayLike:
        return self.sim.get_transimission_magnitude(self.monitor)