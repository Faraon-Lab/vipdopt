import time
from concurrent.futures import wait

import numpy as np
import pytest

from testing import assert_equal, catch_exits
from vipdopt.mpi import FileExecutor, FunctionExecutor

#
# Tests for FunctionExecutor
#

def square(n):
    return n ** 2

@pytest.mark.mpi(min_size=2)
def test_submit_func():
    with FunctionExecutor() as ex:
        res = ex.submit(square, 4).result()
        assert_equal(res, 16)


@pytest.mark.mpi(min_size=2)
def test_map():
    with FunctionExecutor() as ex:
        res = list(ex.map(square, range(10)))
        assert_equal(res, np.power(range(10), 2))


@pytest.mark.mpi(min_size=1)
def test_no_workers():
    # Create a communicator with only one rank
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    root = comm.Get_group().Incl([0])
    root_comm = comm.Create(root)

    # if not the root, we're done
    if root_comm == MPI.COMM_NULL:
        return

    # Using only one rank forces the code to use `manager_thread` as target for Pool
    with FunctionExecutor(comm=root_comm) as ex:
        assert_equal(ex.submit(square, 4).result(), 16)
        assert_equal(list(ex.map(square, range(10))), np.power(range(10), 2))


def long_wait(x: int):
    time.sleep(x)
    return x


@pytest.mark.mpi(min_size=2)
def test_timeout():
    with FunctionExecutor() as ex, pytest.raises(TimeoutError):
        list(ex.map(long_wait, range(3), timeout=1))

#
# Tests for FileExecutor
#

@catch_exits
@pytest.mark.mpi(min_size=1)
def test_submit_file():
    with FileExecutor() as ex:
        res = ex.submit('python', ['testing/mpitest.py'], num_workers=2).result()
        assert_equal(res, 1)


@catch_exits
@pytest.mark.mpi(min_size=1)
def test_submit_one_worker():
    """Should still work when "mpirun-ing" a file with only one process"""
    with FileExecutor() as ex:
        res = ex.submit('python', ['testing/mpitest.py'], num_workers=1).result()
        assert_equal(res, 0)


@catch_exits
@pytest.mark.mpi(min_size=1)
def test_multiple_files():
    with FileExecutor() as ex:
        fut1 = ex.submit('python', ['testing/mpitest.py'], num_workers=1)
        fut2  = ex.submit('python', ['testing/mpitest.py'], num_workers=2)

        wait([fut1, fut2])

        res1 = fut1.result()
        res2 = fut2.result()

        assert_equal(res1, 0)
        assert_equal(res2, 1)
