"""Module containing common functions used throughout the package."""

from typing import TypeVar

import numpy as np
import numpy.typing as npt

Number = TypeVar('Number', int, float, complex)

# Generic types for type hints
T = TypeVar('T')
R = TypeVar('R')

def sech(z: npt.ArrayLike | Number) -> npt.ArrayLike | Number:
    """Hyperbolic Secant."""
    return 1.0 / np.cosh(np.asanyarray(z))
