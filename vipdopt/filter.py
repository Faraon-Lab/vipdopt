"""Module for the abstract Filter class and all its implementations."""

import abc
from typing import get_args

import numpy as np
import numpy.typing as npt
from overrides import override

from vipdopt.utils import Number, sech

SIGMOID_BOUNDS = (0.0, 1.0)


# TODO: design a way for different filters to take different arguments in methods
# TODO: Make code more robust to inputs being arrays iinstead of scalars
class IFilter(abc.ABC):
    """An abstract interface for Filters."""

    @abc.abstractproperty
    def _bounds(self):
        pass

    def verify_bounds(self, variable: npt.ArrayLike | Number) -> bool:
        """Checks if variable is within bounds of this filter.

        Variable can either be a single number or an array of numbers.
        """
        return bool(
            (np.min(variable) >= self._bounds[0])
            and (np.max(variable) <= self._bounds[1])
        )

    @abc.abstractmethod
    def forward(self, x: npt.ArrayLike | Number) -> npt.ArrayLike | np.number:
        """Propogate x through the filter and return the result."""

    @abc.abstractmethod
    def chain_rule(
        self,
        deriv_out: npt.ArrayLike | Number,
        var_out: npt.ArrayLike | Number,
        var_in: npt.ArrayLike | Number,
    ) -> npt.ArrayLike | Number:
        """Apply the chain rule and propogate the derivative back one step."""


class Sigmoid(IFilter):
    """Applies a sigmoidal projection filter to binarize an input.

    Takes an input auxiliary density p(x) ranging from 0 to 1 and applies a sigmoidal
    projection filter to binarize / push it to either extreme. This depends on the
    strength of the filter. See OPTICA paper supplement Section IIA,
    https://doi.org/10.1364/OPTICA.384228,  for details. See also Eq. (9) of
    https://doi.org/10.1007/s00158-010-0602-y.

    Attributes:
        eta (Number): The center point of the sigmoid. Must be in range [0, 1].
        beta (Number): The strength of the sigmoid
        _bounds (tuple[Number]): The bounds of the filter. Always equal to (0, 1)
        _denominator (Number | npt.ArrayLike): The denominator used in various methods;
            for reducing re-computation.
    """
    @property
    def _bounds(self):
        return SIGMOID_BOUNDS

    def __init__(self, eta: Number, beta: Number) -> None:
        """Initialize a sigmoid filter based on eta and beta values."""
        if not self.verify_bounds(eta):
            raise ValueError('Eta must be in the range [0, 1]')

        self.eta = eta
        self.beta = beta

        # Calculate denominator for use in methods
        self._denominator = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))

    def __repr__(self) -> str:
        """Return a string representation of the filter."""
        return f'Sigmoid filter with eta={self.eta:0.3f} and beta={self.beta:0.3f}'

    @override
    def forward(self, x: npt.ArrayLike | Number) -> npt.ArrayLike | np.number:
        """Propogate x through the filter and return the result.
        All input values of x above the threshold eta, are projected to 1, and the
        values below, projected to 0. This is Eq. (9) of https://doi.org/10.1007/s00158-010-0602-y.
        """
        numerator = np.tanh(self.beta * self.eta) + np.tanh(self.beta * (x - self.eta))
        return numerator / self._denominator

    @override
    def chain_rule(
        self,
        deriv_out: npt.ArrayLike | Number,
        var_out: npt.ArrayLike | Number,
        var_in: npt.ArrayLike | Number
    ) -> npt.ArrayLike | Number:
        """Apply the chain rule and propogate the derivative back one step.

        Returns the first argument, multiplied by the direct derivative of forward()
        i.e. Eq. (9) of https://doi.org/10.1007/s00158-010-0602-y, with respect to
        \\tilde{p_i}.
        """
        del deriv_out, var_out  # not needed for sigmoid filter

        numerator = self.beta * np.power(sech(self.beta * (var_in - self.eta)), 2)
        return numerator / self._denominator

    def fabricate(self, x: npt.ArrayLike | Number) -> npt.ArrayLike | Number:
        """Apply filter to input as a hard step-function instead of sigmoid.

        Returns:
            _bounds[0] where x <= eta, and _bounds[1] otherwise
        """
        fab = np.array(x)
        fab[fab <= self.eta] = self._bounds[0]
        fab[fab > self.eta] = self._bounds[1]
        if isinstance(x, get_args(Number)):
            return fab.item()
        return fab
