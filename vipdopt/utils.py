"""Module containing common functions used throughout the package."""

import importlib.util as imp
from typing import TypeVar

import numpy as np
import numpy.typing as npt

Number = int | float | complex

# Generic types for type hints
T = TypeVar('T')
R = TypeVar('R')

def sech(z: npt.ArrayLike | Number) -> npt.ArrayLike | Number:
    """Hyperbolic Secant."""
    return 1.0 / np.cosh(np.asanyarray(z))

def import_lumapi(loc: str):
    """Import the Lumerical python api from a specified location."""
    spec_lin = imp.spec_from_file_location('lumapi', loc)
    lumapi = imp.module_from_spec(spec_lin)
    spec_lin.loader.exec_module(lumapi)
    return lumapi
