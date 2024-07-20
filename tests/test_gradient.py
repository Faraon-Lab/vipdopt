import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest
from jax import grad, jit, vmap

from testing import assert_close, assert_greater_than, assert_less_than
from vipdopt.optimization import (
    AdamOptimizer,
    Device,
    FoM,
    GaussianFoM,
    GradientAscentOptimizer,
    UniformMAEFoM,
    UniformMSEFoM,
)

DEVICE_SIZE = 25 * 25 * 5


def avg_abs_dist(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.ArrayLike:
    """Compute the avergae absolute distance."""
    return np.abs(x - y).mean()


@pytest.mark.smoke()
@pytest.mark.parametrize(
    'opt, device, fom',
    [
        (
            GradientAscentOptimizer(step_size=1e-4),
            {'randomize': True, 'init_seed': 0},
            UniformMAEFoM(range(5), [], 0.5),
        ),
        (
            AdamOptimizer(step_size=1e-4),
            {'randomize': True, 'init_seed': 0},
            UniformMAEFoM(range(5), [], 0.5),
        ),
        (
            GradientAscentOptimizer(step_size=1e-4),
            {'randomize': True, 'init_seed': 0},
            UniformMSEFoM(range(5), [], 0.5),
        ),
        (
            AdamOptimizer(step_size=1e-4),
            {'randomize': True, 'init_seed': 0},
            UniformMSEFoM(range(5), [], 0.5),
        ),
    ],
    indirect=['device'],
)
def test_step(opt, device: Device, fom: FoM):
    """Test a single step with the gradient."""
    initial_w = device.get_design_variable()
    initial_dist = avg_abs_dist(initial_w, 0.5)

    opt.step(device, fom.compute_grad(device.get_design_variable()), 0)

    new_w = device.get_design_variable()
    new_dist = avg_abs_dist(new_w, 0.5)

    assert_less_than(new_dist, initial_dist)  # Should have moved closer to target
    assert_close(new_w, initial_w, err=0.0001)  # One step should not perturb too much


@pytest.mark.parametrize(
    'opt, device',
    [
        (GradientAscentOptimizer(step_size=1e-4), {'randomize': True, 'init_seed': 0}),
        (GradientAscentOptimizer(step_size=1e-4), {'init_density': 1.0}),
        (AdamOptimizer(step_size=1e-4), {'randomize': True, 'init_seed': 0}),
    ],
    indirect=['device'],
)
def test_uniform(opt, device: Device):
    """Test that a device conforms to uniformity in right circumstances."""
    n_freq = 5
    n_iter = 10000

    fom = UniformMAEFoM(
        range(n_freq),
        [],
        0.5,  # Tests  absolute value from 0.5
    )

    for i in range(n_iter):
        g = fom.compute_grad(device.get_design_variable())
        opt.step(device, g, i)

    f = fom.compute_fom(device.get_design_variable())
    assert_close(f, 1.0, 0.01)

    assert_close(device.get_design_variable(), 0.5)


@pytest.mark.parametrize(
    'device',
    [
        {'randomize': True, 'init_seed': 0},
    ],
    indirect=True,
)
def test_dual_fom_uniform(device: Device):
    """Using two opposing FoMs should balance out."""
    n_iter = 10000

    # Tests squared error with 0.0
    fom1 = UniformMSEFoM(range(5), [], 0.0)
    # Tests squared error with 1.0
    fom2 = UniformMSEFoM(range(5), [], 1.0)

    # Since both are equally weighted, should balance out to 0.5 in theory
    fom = fom1 + fom2

    opt = GradientAscentOptimizer(step_size=1e-4)

    n_iter = 50000
    for i in range(n_iter):
        g = fom.compute_grad(device.get_design_variable())
        opt.step(device, g, i)

    # Check that FoM is maximized at x = 0.5
    f = fom.compute_fom(device.get_design_variable())
    assert_close(f, 1.5)
    assert_close(device.get_design_variable(), 0.5)


@pytest.mark.parametrize(
    'opt, device',
    [
        (GradientAscentOptimizer(step_size=1e-4), {'randomize': True, 'init_seed': 0}),
        (GradientAscentOptimizer(step_size=1e-4), {'init_density': 1.0}),
        (AdamOptimizer(step_size=1e-4), {'randomize': True, 'init_seed': 0}),
    ],
    indirect=['device'],
)
def test_gaussianfom(opt, device: Device):
    fom = GaussianFoM(range(5), [], 25, 5)

    for i in range(50000):
        g = fom.compute_grad(device.get_design_variable())
        opt.step(device, g, i)

    f = fom.compute_fom(device.get_design_variable())
    assert_close(f, 1.0 * DEVICE_SIZE)

    w = device.get_design_variable()
    k = fom.kernel[..., np.newaxis]

    assert_close(np.square(w - k), np.zeros(w.shape))


@pytest.mark.parametrize(
    'opt, device',
    [
        (GradientAscentOptimizer(step_size=1e-4), {'randomize': True, 'init_seed': 0}),
    ],
    indirect=['device'],
)
def test_autograd(opt, device: Device):
    # Use logistic function for FoM
    def fom(x: npt.NDArray):
        return 1 / (1 + jnp.exp(-x))

    # Find gradient using autograd
    grad_func = jit(vmap(vmap(vmap(grad(fom, holomorphic=True)))))

    n_iter = 100000
    for i in range(n_iter):
        g = grad_func(device.get_design_variable())
        opt.step(device, g, i)

    w = device.get_design_variable()
    f = fom(w).sum()

    assert_greater_than(f, 0.8 * DEVICE_SIZE)
    assert_greater_than(w, 1.5)
