"""Adam optimizer."""

from typing import Callable, Sequence

import numpy as np
import numpy.typing as npt

from vipdopt.device import Device
from vipdopt.utils import Number
from vipdopt.optimization.optimization import Optimizer
from vipdopt.optimization.fom import BayerFilterFoM, FoM
from vipdopt.simulation import ISimulation

class AdamOptimizer(Optimizer):

    def __init__(
            self,
            step_size: float,
            betas: tuple[float, float]=(0.9, 0.999),
            eps: float=1e-8,
            moments: tuple[float, float]| npt.ArrayLike=(0.0, 0.0),
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
            gradient: npt.ArrayLike,
            *args,
            **kwargs
    ):
        super().step(device, *args, **kwargs)

        # Take gradient step using Adam algorithm
        grad = device.backpropagate(gradient)

        b1, b2 = self.betas
        m = self.moments[0, ...]
        v = self.moments[1, ...]

        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        self.moments = np.array([m, v])

        m_hat = m / (1 - b1 ** (self.iteration + 1))
        v_hat = v / (1 - b2 ** (self.iteration + 1))
        w_hat = device.w - self.step_size * m_hat / np.sqrt(v_hat + self.eps)

        # Apply changes
        device.set_design_variable(np.maximum(np.minimum(w_hat, 1), 0))
    
    def run(self, device: Device, sim: ISimulation, foms: list[FoM]):
        while self.iteration < self.max_iter:
            for callback in self._callbacks:
                callback()

            # TODO:
            # For sim in sims:
            #     call sim.run()
            sim.save(f'test_sim_step_{self.iteration}')
            sim.run()
            
            # device should have access to it's necessary monitors in the simulations
            fom = self.fom_func(foms)
            self.fom_hist.append(fom)

            # Likewise for gradient
            gradient = self.grad_func(foms)

            # Step with the gradient
            self.param_hist.append(device.get_design_variable())
            self.step(device, gradient)
            self.iteration += 1

            break
