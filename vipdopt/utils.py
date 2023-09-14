"""Module containing common functions used throughout the package."""


import numpy as np
import numpy.typing as npt

Number = int | float | complex

def sech(z: npt.ArrayLike | Number) -> npt.ArrayLike | Number:
    """Hyperbolic Secant."""
    return 1.0 / np.cosh(np.asanyarray(z))
