import pytest

from testing.utils import assert_close
from tests.conftest import MONITOR_DATA_DIR

# from vipdopt.optimization import FoM, SuperFoM
from vipdopt.simulation import Power


def test_load(focal_monitor_efield, transmission_monitor_t):
    m = Power('focal_monitor_0')
    m.set_source(MONITOR_DATA_DIR / 'sim_focal_monitor_0.npz')

    assert_close(m.e, focal_monitor_efield)

    m = Power('transmission_monitor_0')
    m.set_source(MONITOR_DATA_DIR / 'sim_transmission_monitor_0.npz')

    assert_close(m.t, transmission_monitor_t)


def test_reset(mocker, focal_monitor_efield):
    m = Power('monitor')
    m._sync = True  # noqa: SLF001
    with pytest.raises(RuntimeError, match=r'has no source to load data from'):
        e = m.load_source()
    m.set_source(MONITOR_DATA_DIR / 'sim_focal_monitor_0.npz')

    spy = mocker.spy(m, 'load_source')

    assert m._e is None  # noqa: SLF001
    e = m.e
    assert_close(m._e, focal_monitor_efield)  # noqa: SLF001
    assert_close(e, focal_monitor_efield)
    _ = m.h  # This shouldn't load again; still in memory.

    spy.assert_called_once()

    m.reset()
    assert m._e is None  # Data should no longer be in memory  # noqa: SLF001
