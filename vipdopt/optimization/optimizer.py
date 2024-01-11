from vipdopt.optimization.device import Device


import numpy.typing as npt
import abc


# TODO: Add support for otehr types of optimizers
class GradientOptimizer(abc.ABC):
    """Abstraction class for all gradient-based optimizers."""

    @abc.abstractmethod
    def step(self, device: Device, gradient: npt.ArrayLike):
        """Step forward one iteration in the optimization process."""