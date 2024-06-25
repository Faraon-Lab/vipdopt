
import numpy as np
import matplotlib.pyplot as plt

from testing import assert_close

from vipdopt.optimization import UniformFoM, GradientAscentOptimizer, Device, LumericalOptimization, AdamOptimizer
from vipdopt.simulation import Power


def test_uniform():
    """Test that a device conforms to uniformity in right circumstances."""
    n_freq = 2
    n_iter = 10000


    mon = Power('mon1')
    fom = UniformFoM(
        'TE+TM',
        [],
        [],
        [mon],
        [],
        range(n_freq),
        [],
        0.5  # Tests  absolute value from 0.5
    )

    for opt in [GradientAscentOptimizer(step_size=1e-4), AdamOptimizer(step_size=1e-4)]:
        device = Device(
            (5, 5, n_freq),
            (0.0, 1.0),
            {'x': np.array(0), 'y': np.array(0), 'z': np.array(0)},
            # init_density=0.25,
            randomize=True,
            init_seed=0
        )

        f_hist = np.zeros(n_iter)

        for i in range(n_iter):
            mon._e = device.get_design_variable()
            g = fom.compute_grad()
            opt.step(device, g, i)
        

        mon._e = device.get_design_variable()

        f = fom.compute_fom()
        assert_close(f, 1.0)

        assert_close(device.get_design_variable(), 0.5)
        