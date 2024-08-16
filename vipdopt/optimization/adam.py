"""Adam optimizer."""

import numpy as np
import numpy.typing as npt

import vipdopt
from vipdopt.optimization.device import Device
from vipdopt.optimization.optimizer import GradientOptimizer


class AdamOptimizer(GradientOptimizer):
    """Optimizer implementing the Adaptive Moment Estimation (Adam) algorithm."""

    betas: tuple[float, float]
    moments: npt.NDArray
    step_size: float
    eps: float

    def __init__(
        self,
        step_size: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        moments: npt.ArrayLike = (0.0, 0.0),
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
        gradient = device.backpropagate(gradient)
        b1, b2 = self.betas
        save_history = False
        if save_history:    # This is if you want to save the entire moment history.
            m = self.moments[0, ...]      
            v = self.moments[1, ...]
        else:               # Drastically reduces the checkpoint savefile size
            m = self.moments[0]
            v = self.moments[1]

        m = b1 * m + (1 - b1) * gradient
        v = b2 * v + (1 - b2) * gradient**2
        self.moments = np.array([m, v])

        m_hat = m / (1 - b1 ** (iteration + 1))
        v_hat = v / (1 - b2 ** (iteration + 1))
        w_hat = device.get_design_variable() + self.step_size * m_hat / np.sqrt(
            v_hat + self.eps
        )
        
        device_diff = np.abs(np.maximum(np.minimum(w_hat, 1), 0) - device.get_design_variable())
        device_diff_without_clipping = np.abs(w_hat - device.get_design_variable())
        vipdopt.logger.info(f'Max change is {np.max(device_diff)}')
        vipdopt.logger.info(f'Max change without clipping is {np.max(device_diff_without_clipping)}')
        vipdopt.logger.info(f'Min change is {np.min(device_diff)}')
        vipdopt.logger.info(f'Min change without clipping is {np.min(device_diff_without_clipping)}')

        # Apply changes
        device.set_design_variable(np.maximum(np.minimum(w_hat, 1), 0))
