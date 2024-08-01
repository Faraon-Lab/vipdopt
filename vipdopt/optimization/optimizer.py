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
