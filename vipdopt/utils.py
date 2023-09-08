"""Module containing common functions used throughout the package."""

import numpy as np
import numpy.typing as npt


def sech(z: npt.ArrayLike | np.number) -> npt.ArrayLike | np.number:
    """Hyperbolic Secant."""
    return 1.0 / np.cosh(np.asanyarray(z))
