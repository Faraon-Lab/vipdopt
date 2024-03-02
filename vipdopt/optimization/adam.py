"""Adam optimizer."""


import numpy as np
import numpy.typing as npt

from vipdopt.optimization.device import Device
from vipdopt.optimization.optimizer import GradientOptimizer


class AdamOptimizer(GradientOptimizer):
    """Optimizer implementing the Adaptive Moment Estimation (Adam) algorithm."""

    def __init__(
            self,
            step_size: float=0.01,
            betas: tuple[float, float]=(0.9, 0.999),
            eps: float=1e-8,
            moments: npt.ArrayLike=(0.0, 0.0),
            **kwargs,
    ) -> None:
        """Initialize an AdamOptimizer instance."""
        super().__init__(
            step_size=step_size,
            betas=tuple(betas),
            eps=float(eps),
            moments=np.array(moments),
            **kwargs,
        )

    def step(self, device: Device, gradient: npt.ArrayLike, iteration: int):
        """Take gradient step using Adam algorithm."""
        device.backpropagate(gradient)       # Already performed outside of this function

        # Changed to gradient descent for now...
        b1, b2 = self.betas
        m = self.moments[0, ...]
        v = self.moments[1, ...]

        m = b1 * m + (1 - b1) * gradient
        v = b2 * v + (1 - b2) * gradient ** 2
        self.moments = np.array([m, v])

        m_hat = m / (1 - b1 ** (iteration + 1))
        v_hat = v / (1 - b2 ** (iteration + 1))
        w_hat = device.get_design_variable() + self.step_size * m_hat / np.sqrt(v_hat + self.eps)


        # Apply changes
        device.set_design_variable(np.maximum(np.minimum(w_hat, 1), 0))
