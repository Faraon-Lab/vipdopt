"""Subpackage providing optimization functionality for inverse design."""

# from vipdopt.optimization.adam import AdamOptimizer
from vipdopt.optimization.device import Device
# from vipdopt.optimization.filter import Filter, Scale, Sigmoid
from vipdopt.optimization.fom import (
#     BayerFilterFoM,
    FoM,
#     GaussianFoM,
#     SuperFoM,
#     UniformMAEFoM,
    UniformMSEFoM,
    MSEFoM,
)
from vipdopt.optimization.optimization import Optimization #, LumericalOptimization
from vipdopt.optimization.optimizer import NLOptOptimizer, GradientAscentOptimizer, GradientOptimizer, AdamOptimizer

__all__ = [
    'AdamOptimizer',
    'GradientAscentOptimizer',
    'BayerFilterFoM',
    'Device',
    'Filter',
    'FoM',
    'FoM',
    'UniformMAEFoM',
    'UniformMSEFoM',
    'SuperFoM',
    'GaussianFoM',
    'GradientOptimizer',
    'LumericalOptimization',
    'Sigmoid',
    'Scale',
]
