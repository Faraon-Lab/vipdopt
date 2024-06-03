from copy import copy

import numpy as np
import numpy.typing as npt
import pytest

from tests.conftest import MONITOR_DATA_DIR
from testing.utils import assert_close, assert_equal
# from vipdopt.optimization import FoM, SuperFoM
from vipdopt.simulation import Monitor, Power, Profile


def test_load(focal_monitor_efield, transmission_monitor_t):
    m = Power('focal_monitor_0')
    m.set_src(MONITOR_DATA_DIR / 'sim_focal_monitor_0.npz')

    assert_close(m.e, focal_monitor_efield)

    m = Power('transmission_monitor_0')
    m.set_src(MONITOR_DATA_DIR / 'sim_transmission_monitor_0.npz')

    assert_close(m.t, transmission_monitor_t)


def test_reset(mocker, focal_monitor_efield):
    m = Power('monitor')
    with pytest.raises(RuntimeError, match=r'has no source to load data from'):
        m._sync = True
        e = m.load_source()
    m.set_src(MONITOR_DATA_DIR / 'sim_focal_monitor_0.npz')

    spy = mocker.spy(m, 'load_source')

    assert m._e is None
    e = m.e
    assert_close(m._e, focal_monitor_efield)
    assert_close(m.e, focal_monitor_efield)
    h = m.h  # This shouldn't load again; still in memory.

    spy.assert_called_once()

    m.reset()
    assert m._e is None  # Data should no longer be in memory
