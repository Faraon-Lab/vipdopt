"""Adam optimizer."""

from typing import Callable, Sequence

import numpy as np
import numpy.typing as npt

from vipdopt.optimization.device import Device
from vipdopt.utils import Number
from vipdopt.optimization.optimizer import GradientOptimizer

class AdamOptimizer(GradientOptimizer):

    def __init__(
            self,
            step_size: float,
            betas: tuple[float, float]=(0.9, 0.999),
            eps: float=1e-8,
            moments: tuple[float, float] | npt.ArrayLike=(0.0, 0.0),
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.moments = moments
        self.step_size = step_size
        self.betas = betas
        self.eps = eps

    def step(self, device: Device, gradient: npt.ArrayLike, iteration: int):
        """Take gradient step using Adam algorithm."""
        grad = device.backpropagate(gradient)

        b1, b2 = self.betas
        m = self.moments[0, ...]
        v = self.moments[1, ...]

        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        self.moments = np.array([m, v])

        m_hat = m / (1 - b1 ** (iteration + 1))
        v_hat = v / (1 - b2 ** (iteration + 1))
        w_hat = device.w - self.step_size * m_hat / np.sqrt(v_hat + self.eps)

        # Apply changes
        device.set_design_variable(np.maximum(np.minimum(w_hat, 1), 0))
