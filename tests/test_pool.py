import time
from concurrent.futures import wait

import numpy as np
import pytest
from mpi4py import MPI

from testing import assert_equal, catch_exits, sleep_and_raise
from vipdopt.mpi import BrokenExecutorError, FileExecutor, FunctionExecutor
from vipdopt.mpi.pool import Executor, comm_split


#
# General Tests for any executor
#
@catch_exits
@pytest.mark.parametrize(
        'executor, kwargs',
        [
            (FunctionExecutor, {'arg1': 1, 'arg2': 2}),
            (FileExecutor, {'arg1': 1, 'arg2': 2}),
        ],
)
@pytest.mark.mpi()
def test_kwargs(executor: Executor, kwargs: dict):
    with executor(**kwargs) as ex:
        assert ex is not None


@catch_exits
@pytest.mark.parametrize(
        'executor',
        [
            (FunctionExecutor),
            (FileExecutor),
        ],
)
@pytest.mark.mpi(min_size=2)
def test_arg_root(executor: Executor):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    for root in range(comm.Get_size()):
        with executor(comm=comm, root=root) as ex:
            if rank == root:
                assert ex is not None
            else:
                assert_equal(ex, None)
        with executor(root=root) as ex:
            if rank == root:
                assert ex is not None
            else:
                assert_equal(ex, None)

@catch_exits
@pytest.mark.parametrize(
        'executor',
        [
            (FunctionExecutor),
            (FileExecutor),
        ],
)
@pytest.mark.mpi()
def test_bad_root(executor: Executor):
    size = MPI.COMM_WORLD.Get_size()
    with pytest.raises(
        ValueError,
        match=f'Expected a potisitve root rank. Received {-size}',
    ):
        executor(root=-size)
    with pytest.raises(
        ValueError,
        match='Expected a potisitve root rank. Received -1',
    ):
        executor(root=-1)
    with pytest.raises(
        ValueError,
        match='Root rank cannot be larger than largest rank',
    ):
        executor(root=size)


@catch_exits
@pytest.mark.parametrize(
        'executor',
        [
            (FunctionExecutor),
            (FileExecutor),
        ],
)
@pytest.mark.mpi(min_size=2)
def test_bad_comm(executor: Executor):
    comm_world = MPI.COMM_WORLD
    intercomm, intracomm = comm_split(comm_world, 0)
    with pytest.raises(
        ValueError,
        match=r'Expected\ an intracommunicator,\ received .*',
    ):
        executor(intercomm)

    intercomm.Free()
    if intracomm:
        intracomm.Free()


@catch_exits
@pytest.mark.parametrize(
        'executor, target, args',
        [
            (FunctionExecutor, time.sleep, [0]),
            (FileExecutor, 'python', [['testing/mpitest.py']]),
        ],
)
@pytest.mark.mpi()
def test_initializer(executor: Executor, target, args):
    with executor(initializer=time.sleep, initargs=(0,)) as ex:
        ex.submit(target, *args).result()
#
# Tests for FunctionExecutor
#

@pytest.mark.mpi()
def test_initializer_error():
    executor = FunctionExecutor(initializer=sleep_and_raise, initargs=(0,))
    with executor as ex:
        future = ex.submit(time.sleep, 0)
        with pytest.raises(BrokenExecutorError):
            future.result()


@pytest.mark.mpi()
def test_default():
    with FunctionExecutor() as ex:
        if ex is not None:
            ex.bootup()
            f1 = ex.submit(time.sleep, 0)
            f2 = ex.submit(time.sleep, 0)
            ex.shutdown()

            assert_equal(f1.result(), None)
            assert_equal(f2.result(), None)


@pytest.mark.mpi()
def test_self():
    """Use COMM_SELF"""
    with FunctionExecutor(comm=MPI.COMM_SELF) as ex:
        future = ex.submit(time.sleep, 0)
        assert_equal(future.result(), None)
        assert_equal(future.exception(), None)

        future = ex.submit(sleep_and_raise, 0)
        with pytest.raises(RuntimeError, match='expected raise'):
            future.result()
        assert_equal(type(future.exception()), RuntimeError)

        list(ex.map(time.sleep, [0, 0]))
        list(ex.map(time.sleep, [0, 0], timeout=1))
        with pytest.raises(TimeoutError):
            list(ex.map(time.sleep, [0, 0.2], timeout=0))


def square(n):
    return n ** 2


@pytest.mark.mpi()
def test_submit_func():
    with FunctionExecutor() as ex:
        res = ex.submit(square, 4).result()
        assert_equal(res, 16)


@pytest.mark.mpi(min_size=1)
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


@pytest.mark.mpi(min_size=1)
def test_multiple_workers():
    with FunctionExecutor(max_workers=2) as ex:
        res = list(ex.map(square, range(10)))
        assert_equal(res, np.power(range(10), 2))

#
# Tests for FileExecutor
#

@catch_exits
@pytest.mark.mpi(min_size=1)
def test_submit_file(tmpdir):
    p = tmpdir / 'work'
    with FileExecutor(root_dir=p) as ex:
        res = ex.submit('python', ['testing/mpitest.py'], num_workers=2).result()
        assert_equal(res, 1)


@catch_exits
@pytest.mark.mpi(min_size=1)
def test_submit_one_worker(tmpdir):
    """Should still work when "mpirun-ing" a file with only one process"""
    p = tmpdir / 'work'
    with FileExecutor(root_dir=p) as ex:
        res = ex.submit('python', ['testing/mpitest.py'], num_workers=1).result()
        assert_equal(res, 0)


@catch_exits
@pytest.mark.mpi(min_size=1)
def test_multiple_files(tmpdir):
    p = tmpdir / 'work'
    with FileExecutor(root_dir=p) as ex:
        fut1 = ex.submit('python', ['testing/mpitest.py'], num_workers=1)
        fut2  = ex.submit('python', ['testing/mpitest.py'], num_workers=2)

        wait([fut1, fut2])

        res1 = fut1.result()
        res2 = fut2.result()

        assert_equal(res1, 0)
        assert_equal(res2, 1)


@catch_exits
@pytest.mark.mpi(min_size=1)
def test_multiple_executors(tmpdir):
    """Ensure everything works when creating an executor in multiple contexts."""
    p = tmpdir / 'work'
    with FileExecutor(root_dir=p) as ex:
        res = ex.submit('python', ['testing/mpitest.py'], num_workers=2).result()
        assert_equal(res, 1)

    with FileExecutor(root_dir=p) as ex:
        res = ex.submit('python', ['testing/mpitest.py'], num_workers=2).result()
        assert_equal(res, 1)
