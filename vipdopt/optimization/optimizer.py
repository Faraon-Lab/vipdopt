"""Code for representing optimizers."""
import sys

import abc
import numpy.typing as npt
import numpy as np

import vipdopt
from vipdopt.optimization.device import Device

def _load_optimizer(cfg):
        """Load the optimizer from a config."""
        optimizer: str = cfg.pop('optimizer', None)
        if optimizer is None:
            vipdopt.logging.warning('No optimizer declared in config.')
            return
        optimizer_settings: dict = cfg.pop('optimizer_settings', {})
        try:
            optimizer_type = getattr(sys.modules['vipdopt.optimization'], optimizer)
        except AttributeError:
            raise NotImplementedError(
                f'Optimizer {optimizer} not currently supported'
            ) from None
        
        return optimizer_type(**optimizer_settings)



# TODO: Add support for other types of optimizers
class GradientOptimizer(abc.ABC):
    """Abstraction class for all gradient-based optimizers."""

    def __init__(self, **kwargs):
        """Initialize a GradientOptimizer."""
        vars(self).update(kwargs)

    @abc.abstractmethod
    def step(self, device: Device, gradient: npt.ArrayLike, iteration: int):
        """Step forward one iteration in the optimization process."""


class GradientAscentOptimizer(GradientOptimizer):
    """Optimizer for doing basic gradient ascent."""

    step_size: float

    def __init__(self, step_size=0.01, **kwargs):
        """Initialize a GradientDescentOptimizer."""
        super().__init__(step_size=step_size, **kwargs)

    def step(
        self,
        device: Device,
        gradient: npt.ArrayLike,
        iteration: int,  # noqa: ARG002
    ):
        """Step with the gradient."""
        grad = device.backpropagate(gradient)
        w_hat = device.get_design_variable() + self.step_size * grad

        device.set_design_variable(device.clip(w_hat))

    # TODO: For Gradient Descent/Ascent only: Need to check whether it's respecting the epochs and everything.
    # todo: At the moment this is implemented in a way where it doesn't respect the epoch maximums and minimums.
    def scale_step_size(
        epoch_start_design_change_min,
        epoch_start_design_change_max,
        epoch_end_design_change_min,
        epoch_end_design_change_max,
    ):
        """Begin scaling of step size so that the design change stays within epoch_design_change limits in config."""
        # # 20240726 Ian - one day i'll understand this code
        # if use_fixed_step_size:
        #     step_size = fixed_step_size
        # else:
        #     step_size = step_size_start
        #     check_last = False
        #     last = 0

        #     while True:
        #         # Gets proposed design variable according to Eq. S2, OPTICA Paper Supplement: https://doi.org/10.1364/OPTICA.384228
        #         # Divides step size by 2 until the difference in the design variable is within the ranges set.

        #         proposed_design_variable = cur_design_variable + step_size * design_gradient
        #         proposed_design_variable = np.maximum(                       # Makes sure that it's between 0 and 1
        #                                                 np.minimum(proposed_design_variable, 1.0),
        #                                                 0.0)

        #         difference = np.abs(proposed_design_variable - cur_design_variable)
        #         max_difference = np.max(difference)

        #         if (max_difference <= max_change_design) and (max_difference >= min_change_design):
        #             break										# max_difference in [min_change_design, max_change_design]
        #         elif (max_difference <= max_change_design):		# max_difference < min_change_design, by definition
        #             step_size *= 2
        #             if (last ^ 1) and check_last:	# For a Boolean, last ^ 1 = !last
        #                 break						# skips the next two lines only if last=0 and check_last=True
        #             check_last = True
        #             last = 1
        #         else:											# max_difference > max_change_design
        #             step_size /= 2
        #             if (last ^ 0) and check_last:	# For a Boolean, last ^ 0 = last
        #                 break						# skips the next two lines only if last=1 and check_last=True
        #             check_last = True
        #             last = 0

        return 3


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

        clipped = device.clip(w_hat)
        device_diff = np.abs(clipped - device.get_design_variable())
        device_diff_without_clipping = np.abs(w_hat - device.get_design_variable())
        vipdopt.logger.info(f'Max change is {np.max(device_diff)}')
        vipdopt.logger.info(
            f'Max change without clipping is {np.max(device_diff_without_clipping)}'
        )
        vipdopt.logger.info(f'Min change is {np.min(device_diff)}')
        vipdopt.logger.info(
            f'Min change without clipping is {np.min(device_diff_without_clipping)}'
        )

        # Apply changes
        device.set_design_variable(clipped)


class NonGradientOptimizer(abc.ABC):
    """Abstraction class for all non-gradient-based optimizers."""

    def __init__(self, **kwargs):
        """Initialize a GradientOptimizer."""
        vars(self).update(kwargs)

    @abc.abstractmethod
    def step(self, device: Device, gradient: npt.ArrayLike, iteration: int):
        """Step forward one iteration in the optimization process."""

class NLOptOptimizer():
    def __init__(self, **kwargs):
        """Initialize an Optimizer class interfacing with NLOpt package."""
        vars(self).update(kwargs)

    def step(self, device: Device, gradient: npt.ArrayLike, iteration: int):
        """Step forward one iteration in the optimization process."""
