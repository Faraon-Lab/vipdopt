mpiexec -n 1 -H localhost:4 python -m pytest tests/test_pool.py --with-mpi -v
mpiexec -n 3 -H localhost:7 python -m pytest tests/test_pool.py --with-mpi -v