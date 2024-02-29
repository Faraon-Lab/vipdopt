"""Code for representing optimizers."""
import abc

import numpy.typing as npt

from vipdopt.optimization.device import Device


# TODO: Add support for otehr types of optimizers
class GradientOptimizer(abc.ABC):
    """Abstraction class for all gradient-based optimizers."""

    def __init__(self, **kwargs):
        """Initialize a GradientOptimizer."""
        vars(self).update(kwargs)

    @abc.abstractmethod
    def step(self, device: Device, gradient: npt.ArrayLike, iteration: int):
        """Step forward one iteration in the optimization process."""

class GradientDescentOptimizer(GradientOptimizer):
    """Optimizer for doing basic gradient descent."""

    def __init__(self, step_size=0.01, **kwargs):
        super().__init__(step_size=step_size, **kwargs)
    
    def step(self, device: Device, gradient: npt.ArrayLike, iteration: int):
        """Step with the gradient."""
        grad = device.backpropagate(gradient)
        w_hat = device.get_design_variable() - self.step_size * grad

        device.set_design_variable(w_hat)