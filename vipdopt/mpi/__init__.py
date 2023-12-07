"""Sub package containing all MPI related code."""

from vipdopt.mpi.pool import BrokenExecutorError, FileExecutor, FunctionExecutor

__all__ = ['FileExecutor', 'FunctionExecutor', 'BrokenExecutorError']
