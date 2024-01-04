"""Adam optimizer."""

from typing import Callable, Sequence

import numpy as np

from vipdopt.device import Device
from vipdopt.utils import Number
from vipdopt.optimization.optimization import Optimizer

class AdamOptimizer(Optimizer):

    def __init__(
            self,
            moments: tuple[float, float],
            step_size: float,
            betas: tuple[float, float]=(0.9, 0.999),
            eps: float=1e-8,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.moments = moments
        self.step_size = step_size
        self.betas = betas
        self.eps = eps

    def set_callback(self, func: Callable):
        return super().set_callback(func)
    
    def step(
            self,
            device: Device,
            gradient,
            *args,
            **kwargs
    ):
        super().step(device, *args, **kwargs)

        # Take gradient step using Adam algorithm
        grad = device.backpropagate(gradient)

        b1, b2 = self.betas
        m, v = self.moments

        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        self.moments = (m, v)

        m_hat = m / (1 - b1 ** (self.iteration + 1))
        v_hat = v / (1 - b2 ** (self.iteration + 1))
        w_hat = device.w - self.step_size * m_hat / np.sqrt(v_hat + self.eps)

        # Apply changes
        device.set_design_variable(np.maximum(np.minimum(w_hat, 1), 0))
    
    def run(self, device: Device):
        while self.iteration < self.max_iter:
            for callback in self._callbacks:
                callback()

            # TODO:
            # For sim in sims:
            #     call sim.run()
            
            # device should have access to it's necessary monitors in the simulations
            fom = self.fom_func(device)

            # Likewise for gradient
            gradient = self.grad_func(device)

            # Step with the gradient
            self.step(device, gradient)
