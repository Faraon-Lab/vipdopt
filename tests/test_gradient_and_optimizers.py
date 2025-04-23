import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest
from copy import copy
from jax import grad, jit, vmap

from testing import assert_close, assert_greater_than, assert_less_than, assert_equal
from vipdopt.optimization import (
    AdamOptimizer,
    Device,
    FoM,
    GaussianFoM,
    GradientAscentOptimizer,
    UniformMAEFoM,
    UniformMSEFoM,
    MSEFoM,
)

DEVICE_SIZE = (25, 25, 5)


def avg_abs_dist(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.ArrayLike:
    """Compute the average absolute distance."""
    return np.abs(x - y).mean()


@pytest.mark.smoke()
@pytest.mark.parametrize(
    'opt, example_indirect_device, fom',
    [
        (
            GradientAscentOptimizer(step_size=1e-4),
            {'randomize': True, 'init_seed': 0},
            UniformMAEFoM(None, None, constant=0.5, pos_max_freqs=range(5), all_freqs=range(5)),
        ),
        (
            AdamOptimizer(step_size=1e-4),
            {'randomize': True, 'init_seed': 0},
            UniformMAEFoM(None, None, constant=0.5, pos_max_freqs=range(5), all_freqs=range(5)),
        ),
        (
            GradientAscentOptimizer(step_size=1e-4),
            {'randomize': True, 'init_seed': 0},
            UniformMSEFoM(None, None, constant=0.5, pos_max_freqs=range(5), all_freqs=range(5)),
        ),
        (
            AdamOptimizer(step_size=1e-4),
            {'randomize': True, 'init_seed': 0},
            UniformMSEFoM(None, None, constant=0.5, pos_max_freqs=range(5), all_freqs=range(5)),
        ),
    ],
    indirect=['example_indirect_device'],
)
def test_step(opt, example_indirect_device: Device, fom: FoM):
    """Test a single step with the gradient."""
    initial_w = copy(example_indirect_device.get_design_variable())
    # initial_dist = avg_abs_dist(initial_w, 0.5)

    opt.step(example_indirect_device, 
             -1*fom.compute_grad(x=example_indirect_device.get_design_variable()), 
             0)

    new_w = example_indirect_device.get_design_variable()
    # new_dist = avg_abs_dist(new_w, 0.5)
    
    # assert_less_than(new_dist, initial_dist)            # This may fail depending on random seed.
    assert_less_than(fom.compute_fom(x=new_w), fom.compute_fom(x=initial_w))  # Should have moved closer to target
    assert_close(new_w, initial_w, err=5*1e-4)  # One step should not perturb too much


@pytest.mark.parametrize(
    'opt, example_indirect_device',
    [
        (GradientAscentOptimizer(step_size=1e-3), {'randomize': True, 'init_seed': 0}),
        (GradientAscentOptimizer(step_size=1e-4), {'init_density': 1.0}),
        (AdamOptimizer(step_size=1e-3), {'randomize': True, 'init_seed': 0}),
    ],
    indirect=['example_indirect_device'],
)
def test_uniform(opt, example_indirect_device: Device):
    """Test that a device conforms to uniformity in right circumstances."""
    n_freq = 5
    n_iter = 10000

    fom = UniformMAEFoM(None, None, 
                        constant=0.5,       # Tests  absolute value from 0.5
                        pos_max_freqs=range(5), all_freqs=range(5))

    for i in range(n_iter):
        g = fom.compute_grad(x=example_indirect_device.get_design_variable())
        opt.step(example_indirect_device, -1*g, i)

    f = fom.compute_fom(x=example_indirect_device.get_design_variable())
    
    assert_close(f, 0.0, 0.01)
    assert_close(example_indirect_device.get_design_variable(), 0.5, err=5*opt.step_size)


@pytest.mark.parametrize(
    'example_indirect_device',
    [
        {'randomize': True, 'init_seed': 0},
    ],
    indirect=True,
)
def test_dual_fom_uniform(example_indirect_device: Device):
    """Using two opposing FoMs should balance out."""
    n_iter = 10000

    # Tests squared error with 0.0
    fom1 = UniformMSEFoM(None, None, constant=0.0, pos_max_freqs=range(5), all_freqs=range(5))
    # Tests squared error with 1.0
    fom2 = UniformMSEFoM(None, None, constant=1.0, pos_max_freqs=range(5), all_freqs=range(5))

    # Since both are equally weighted, should balance out to 0.5 in theory
    fom = FoM(None, None, [(fom1,),(fom2,)], (1,1))

    # opt = GradientAscentOptimizer(step_size=1e-4)       # <-- can't hack it.
    opt = AdamOptimizer(step_size=1e-3)

    n_iter = 10000
    for i in range(n_iter):
        g = fom.compute_grad(x=example_indirect_device.get_design_variable())
        opt.step(example_indirect_device, -1*g, i)

    # Check that FoM is maximized at x = 0.5
    f = fom.compute_fom(x=example_indirect_device.get_design_variable())
    assert_close(f, 0.5)
    assert_close(example_indirect_device.get_design_variable(), 0.5)


@pytest.mark.parametrize(
    'opt, example_indirect_mock_device',
    [
        # (GradientAscentOptimizer(step_size=1e-3), {'size':DEVICE_SIZE, 'randomize': True, 'init_seed': 0}),
        # (GradientAscentOptimizer(step_size=1e-4), {'size':DEVICE_SIZE, 'init_density': 1.0}),
        # # The basic gradient ascent is hopeless here for i < 50000
        (AdamOptimizer(step_size=1e-4), {'size':DEVICE_SIZE, 'randomize': True, 'init_seed': 0}),
    ],
    indirect=['example_indirect_mock_device'],
)
def test_gaussianfom(opt, example_indirect_mock_device: Device):
    
    # import matplotlib.pyplot as plt
    
    fom = MSEFoM(target=0.5*np.ones(DEVICE_SIZE))
    fom.set_target_2d_gaussian(arr_shape=DEVICE_SIZE, peak=1, N=9, std=2, center_point=(6,6))
    # plt.imshow(fom.target[...,0])

    for i in range(10000):      # if this changes, obviously the errors below should change as well.
        g = fom.compute_grad(x=example_indirect_mock_device.get_design_variable())
        opt.step(example_indirect_mock_device, -1*g, i)

    # plt.imshow(np.real(example_indirect_mock_device.get_design_variable()[...,0]))

    w = np.real(example_indirect_mock_device.get_design_variable())
    k = fom.target

    assert_close(np.square(w - k), np.zeros(w.shape), err=0.03)
    
    f = fom.compute_fom(x=example_indirect_mock_device.get_design_variable())
    assert_close(f, 0, err=0.01)



@pytest.mark.parametrize(
    'opt, example_indirect_mock_device',
    [
        (GradientAscentOptimizer(step_size=1e-3), {'size':DEVICE_SIZE, 'randomize': True, 'init_seed': 0}),
        (AdamOptimizer(step_size=1e-3), {'size':DEVICE_SIZE, 'randomize': True, 'init_seed': 0}),
    ],
    indirect=['example_indirect_mock_device'],
)
def test_autograd(opt, example_indirect_mock_device: Device):
    """Test that the gradient works when computed using autograd."""

    # Use logistic function for FoM
    def logistic_func(x: npt.NDArray):
        return 1 / (1 + jnp.exp(-x))

    # Find gradient using autograd
    grad_func = jit(vmap(vmap(vmap(grad(logistic_func)))))

    n_iter = 10000
    for i in range(n_iter):
        g = grad_func(jnp.real(example_indirect_mock_device.get_design_variable()))
        opt.step(example_indirect_mock_device, g, i)

    w = example_indirect_mock_device.get_design_variable()
    f = logistic_func(w).sum()

    assert_close(w.mean(), 1.0)
    assert_greater_than(f, 0.70 * np.prod(DEVICE_SIZE))  # FoM is getting close to 0.73 everywhere


TEST_OPTIMIZER = AdamOptimizer()
@pytest.mark.smoke()
def test_save_load(tmpdir):
    opt1 = TEST_OPTIMIZER
    opt1_vars = copy(vars(TEST_OPTIMIZER))
    opt2 = AdamOptimizer(**opt1_vars)
    p = tmpdir / 'optimizer.yml'
    opt_dict_1 = opt1.as_dict()
    
    opt3 = AdamOptimizer.from_dict(opt_dict_1)
    
    assert_equal(opt1.as_dict(), opt3.as_dict())
    # assert_equal(opt_dict_1, opt3.as_dict())      # <-- can't use this one because the load function pops everything out.
    
    for attr in vars(opt1):
        assert_close(opt2.__getattribute__(attr), opt1.__getattribute__(attr))
        assert_close(opt3.__getattribute__(attr), opt1.__getattribute__(attr))