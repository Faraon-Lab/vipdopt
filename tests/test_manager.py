import numpy as np
import pytest

from testing import assert_equal
from vipdopt.manager import MPIPool


def square(n):
    return n ** 2


@pytest.mark.mpi(min_size=2)
def test_submit():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    with MPIPool(comm) as pool:
        res = pool.submit(square, 4)
        if pool.is_manager():
            assert_equal(res, 16)
        else:
            assert_equal(res, None)


@pytest.mark.mpi(min_size=2)
def test_map():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    with MPIPool(comm) as pool:
        res = pool.map(square, range(10))
        if pool.is_manager():
            assert_equal(res, np.power(range(10), 2))
        else:
            assert_equal(res, [None] * 10)


@pytest.mark.mpi(min_size=2)
def test_no_comm():
    with MPIPool() as pool:
        res = pool.submit(square, 4)
        if pool.is_manager():
            assert_equal(res, 16)
        else:
            assert_equal(res, None)
