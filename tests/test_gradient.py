
import numpy as np
import matplotlib.pyplot as plt

from testing import assert_close

from vipdopt.optimization import UniformFoM, GradientDescentOptimizer, Device, LumericalOptimization, AdamOptimizer
from vipdopt.simulation import Power


def test_uniform():
    """Test that a device conforms to uniformity in right circumstances."""
    device = Device(
        (30, 30, 10),
        (0.0, 1.0),
        {'x': np.array(0), 'y': np.array(0), 'z': np.array(0)},
        # init_density=0.25,
        randomize=True,
        init_seed=0
    )
    mon = Power('mon1')

    fom = UniformFoM(
        'TE+TM',
        [],
        [],
        [mon],
        [],
        range(10),
        [],
        0.5  # Tests  absolute value from 0.5
    )

    n_iter = 50
    opt = GradientDescentOptimizer()  # This should work regardless of optimizer, but doesn't ://
    f_hist = np.zeros(n_iter)

    for i in range(n_iter):
        mon._e = device.get_design_variable()

        f = fom.compute_fom()
        f_hist[i] = f.mean()

        mon._e = device.get_design_variable()
        g = fom.compute_grad()
        opt.step(device, g, i)
    
    # print(device.get_design_variable())

    # plt.plot(np.arange(n_iter), f_hist)
    # plt.show()

    assert_close(f_hist[-1], 1.0, err=0.01)
    assert_close(device.get_design_variable(), 0.5, err=0.05)
    