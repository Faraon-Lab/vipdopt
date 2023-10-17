# import pytest

# from testing import assert_equal
# from vipdopt.manager import MPIPool


# @pytest.mark.mpi()
# def test_MPI(mocker):
#     def square(n):
#         return n ** 2

#     spy = mocker.spy(square)

#     from mpi4py import MPI
#     comm = MPI.COMM_WORLD
#     with MPIPool(comm) as pool:
#         assert_equal(pool.submit(square, 4), 16)

#     assert_equal(spy.call_count, 1)
#     assert_equal(spy.spy_return, 16)
