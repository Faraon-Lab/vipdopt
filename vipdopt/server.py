"""Entry point for MPI workers."""

if __name__ == '__main__':
   from vipdopt.pool import ServerWorker
   s = ServerWorker()
