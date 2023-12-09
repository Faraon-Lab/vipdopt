"""Job Pool / Manager Implemented with MPI."""
from __future__ import annotations

import abc
import logging
import os
import shutil
import stat
import sys
import threading
import time
import typing
import uuid
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Future, as_completed
from enum import Enum
from itertools import starmap
from pathlib import Path
from typing import Any

from mpi4py import MPI
from overrides import override

from vipdopt.utils import R

SERIALIZED = None
_thread_level = None
MAIN_RUN_NAME = '__worker__'
_setup_threads_lock = threading.Lock()
_tls = threading.local()
SLEEP_TIME = 0.001

class BrokenExecutorError(BaseException):
    """Exception class for signalling that an executor is broken."""

def do_nothing(x: Any):
    """A dummy function that does literally nothing."""
    return

class TaskType(Enum):
    """Enum specifying the type of `Task` created."""
    FUNCTION = 0
    EXECUTABLE = 1

class Task:
    """Wrapper for pool tasks and associated data to be stored."""

    def __init__(self, task_type: TaskType, **options):
        """Initialize a Task object.

        Arguments:
            task_type (TaskType): Type of the task

            **options:
                If task_type is TaskType.FUNCTION:
                    function (Callable[..., R]): A function to call, REQUIRED.
                    args (tuple[Any]): Arguments to pass into `function`.
                    kwargs (dict): Keyword arguments to pass into `function`.
                    callback (Callable[R, None]): A function to call on the result
                        of `function`.
                If task_type is TaskType.EXECUTABLE:
                    exe (str): The executable to run; REQUIRED
                    num_workers (int): How many child processes to spawn; defaults to 1.
                    args (list[str]): Command-line arguments to call `exe` with.
                    mpi_info (dict): MPI.Info data to pass to spawned child processes.

        Raises:
            ValueError: `num_workers` provided was not a positive integer.

        """
        self.type = task_type

        if task_type == TaskType.FUNCTION:
            self._callback = None
            logging.debug('Creating function task')
            self._func = options['function']
            self._args = options.pop('args', ())
            self._kwargs = options.pop('kwargs', {})
            self._callback = options.pop('callback', do_nothing)
        else:
            logging.debug('Creating executable task')
            num_workers = options.pop('num_workers', 1)
            if num_workers < 1:
                raise ValueError(
                    f'num_workers must be a positive integer; got {num_workers}'
                )
            self.num_workers = num_workers
            exe = options.pop('exe')
            args = options.pop('args', None)
            args = [] if args is None else list(args)
            args = [exe, *args]
            self.args = args
            self.mpi_info: dict = options.pop('mpi_info', {})

        logging.debug('...successfully created new task')

    def __call__(self) -> tuple[Any | None, None | BaseException]:
        """Call `self._function` with the provided arguments."""
        if self.type != TaskType.FUNCTION:
            raise TypeError('__call__ is only defined for Tasks with type "FUNCTION"')

        assert self._callback is not None  # so MyPy is happy

        logging.debug(f'Calling {self}')
        try:
            res = self._func(*self._args, **self._kwargs)
            self._callback(res)
        except BaseException as e:
            return (None, e)
        return (res, None)

    def __repr__(self) -> str:
        """Return string representation of a Task."""
        match self.type:
            case TaskType.FUNCTION:
                return f'\"Function Task: {self._func}({self._args}, {self._kwargs})\"'
            case TaskType.EXECUTABLE:
                return \
                    f'\"File Task: mpirun -n {self.num_workers} {" ".join(self.args)}\"'

# Defining a type alias so I don't have to type this everytime lol
WorkQueue = deque[tuple[Future, Task] | None]

def get_max_workers():
    """Get the maximum number of workers available to use."""
    max_workers = os.environ.get('MAX_WORKERS')
    if max_workers is not None:
        logging.debug('non None max_workers found in environment')
        return int(max_workers)
    if MPI.UNIVERSE_SIZE != MPI.KEYVAL_INVALID:
        usize = MPI.COMM_WORLD.Get_attr(MPI.UNIVERSE_SIZE)
        if usize is not None:
            wsize = MPI.COMM_WORLD.Get_size()
            logging.debug(f'usize: {usize}; wsize: {wsize}')
            return max(usize - wsize, 1)
    return 1

def set_comm_server(intracomm: MPI.Intracomm):
    """Set the current intracomm."""
    global _tls
    _tls.comm_server = intracomm

def initialize(options: dict):
    """Call an initializer before creating a job manager."""
    initializer = options.pop('initializer', None)
    initargs = options.pop('initargs', ())
    initkwargs = options.pop('initkwargs', {})
    if initializer is not None:
        try:
            initializer(*initargs, **initkwargs)
        except BaseException:
            return False
    return True

@typing.no_type_check
def import_main(mod_name: str, mod_path: str, init_globals: dict, run_name: str):
    """Import main module into worker processes."""
    import runpy
    import types

    module = types.ModuleType(run_name)
    if init_globals is not None:
        module.__dict__.update(init_globals)
        module.__name__ = run_name

    class TempModulePatch(runpy._TempModule):
        def __init__(self, mod_name):
            super().__init__(mod_name)
            self.module = module

    temp_module = runpy._TempModule
    runpy._TempModule = TempModulePatch
    import_main.sentinel = (mod_name, mod_path)
    main_module = sys.modules['__main__']
    try:
        sys.modules['__main__'] = sys.modules[run_name] = module
        if mod_name:  # pragma: no cover
            runpy.run_module(mod_name, run_name=run_name, alter_sys=True)
        elif mod_path:  # pragma: no branch
            safe_path = getattr(sys.flags, 'safe_path', sys.flags.isolated)
            if not safe_path:  # pragma: no branch
                sys.path[0] = os.path.realpath(os.path.dirname(mod_path))
            runpy.run_path(mod_path, run_name=run_name)
        sys.modules['__main__'] = sys.modules[run_name] = module
    except BaseException:  # pragma: no cover
        sys.modules['__main__'] = main_module
        raise
    finally:
        del import_main.sentinel
        runpy._TempModule = temp_module

def serialized(function):
    """Return a wrapper around a function so that it requires the `SERIALZED` lock."""
    def wrapper(*args, **kwargs):
        """Serialized wrapper around function."""
        assert SERIALIZED is not None
        with SERIALIZED:
            return function(*args, **kwargs)
    if SERIALIZED is None:
        return function
    return wrapper

def comm_split(comm: MPI.Intracomm, root: int) -> tuple[MPI.Intercomm, MPI.Intracomm]:
    """Create an intercommunicator for the manager to communicate with workers."""
    if root >= comm.Get_size():
        raise ValueError(f'Expected a root rank in range'
                        f'[0, ..., {comm.Get_size() - 1}]. Received {root}')

    if comm.Get_size() == 1:
        return MPI.Intercomm(MPI.COMM_NULL), MPI.Intracomm(MPI.COMM_NULL)


    # Split into two groups: root and everything else
    rank = comm.Get_rank()
    full_group = comm.Get_group()
    group = full_group.Incl([root]) if rank == root else full_group.Excl([root])
    full_group.Free()
    intracomm = MPI.Intracomm(comm.Create(group))
    group.Free()

    local_leader = 0
    remote_leader = (0 if root else 1) if rank == root else root

    intercomm = intracomm.Create_intercomm(
        local_leader,
        comm,
        remote_leader,
        tag=0,
    )
    if rank == root:
        intracomm.Free()
    return intercomm, intracomm

class Executor(abc.ABC):
    """Abstract Executor Class."""
    def __init__(
            self,
            comm: MPI.Intracomm | None=None,
            root: int=0,
            max_workers: int | None=None,
            initializer: Callable | None=None,
            initargs: Sequence[Any]=(),
            **kwargs,
        ) -> None:
        """Initialize an Executor object.

        Arguments:
            comm (MPI.Intracomm): Intracommunicator to use.
            root (int): Which rank to use as root; defaults to 1.
            max_workers (int | None): The maximum workers to use; will attempt to
                compute this manually if None is provided.
            initializer (Callable): An initializer to call before creating executor.
            initargs (Sequence[Any]): Arguments to pass to `initializer`
            **kwargs: Various options to create the executor with.

        Raises:
            ValueError: `comm` provided was an intercommunicator.
            ValueError: `max_workers` provided was not a positive integer.
            ValueError: `rank` out of bounds.
        """
        if root < 0:
            raise ValueError(f'Expected a potisitve root rank. Received {root}')
        if root >= MPI.COMM_WORLD.Get_size():
            raise ValueError('Root rank cannot be larger than largest rank')
        if comm is not None:
            if comm.Is_inter():
                raise ValueError(
                    f'Expected an intracommunicator, received {type(comm)}'
                )
            if root >= comm.Get_size():
                raise ValueError(f'Expected a root rank in range'
                                f'[0, ..., {comm.Get_size() - 1}]. Received {root}')
        elif MPI.COMM_WORLD.Get_size() > 1:
            comm = MPI.COMM_WORLD

        self._root = root
        self._comm = comm

        if max_workers is not None:
            if max_workers <= 0:
                raise ValueError(f'Expected positive max_workers, got {max_workers}')
            kwargs['max_workers'] = max_workers
        if initializer is not None:
            kwargs['initializer'] = initializer
            kwargs['initargs'] = initargs

        self._options = kwargs
        self._shutdown = False
        self._broken = ''
        self._lock = threading.Lock()
        self._pool: Pool | None = None

    def is_manager(self):
        """Return whether this process is the root rank."""
        comm = MPI.COMM_WORLD if self._comm is None else self._comm
        return comm.Get_rank() == self._root

    @abc.abstractmethod
    def setup(self):
        """Setup executor."""

    def __enter__(self):
        """Return self for entering a context manager."""
        ex = self if self.is_manager() else None
        self.setup()
        self._executor = ex

        return ex

    def __exit__(self, *args):
        """Cleanup after exiting a context manager."""
        ex = self._executor
        self._executor = None

        if ex is not None:
            self.shutdown(wait=True)
            return False
        return True

    def shutdown(self, wait: bool=True, cancel_futures: bool=False):
        """Cleanup the executor and any loose threads."""
        with self._lock:
            if not self._shutdown:
                self._shutdown = True
                if self._pool is not None:
                    self._pool.done()
            if cancel_futures and self._pool is not None:
                self._pool.cancel()
            pool = None
            if wait:
                pool = self._pool
                self._pool = None
        if pool is not None:
            pool.join()


class FunctionExecutor(Executor):
    """Job executor for function jobs."""
    def __init__(
            self,
            comm: MPI.Intracomm | None=None,
            root: int=0,
            max_workers: int | None=None,
            initializer: Callable | None=None,
            initargs: Sequence[Any]=(),
            **kwargs,
        ) -> None:
        """Initialize a FunctionExecutor object. Same arguments as Executor."""
        super().__init__(comm, root, max_workers, initializer, initargs, **kwargs)

    def _bootstrap(self):
        """Create a pool if it doesn't yet exist."""
        if self._pool is None:
            self._pool = Pool(self, manager_function)

    def bootup(self, wait=True):
        """Bootup the FunctionExecutor."""
        with self._lock:
            if self._shutdown:
                raise RuntimeError('Cannot boot up after shutdown')
            self._bootstrap()
            assert self._pool is not None
            if wait:
                self._pool.wait()
            return self

    @override
    def setup(self):
        """Setup the FunctionExecutor.

        The root process with create a Pool object, while all other processes will
        enter the FunctionWorker loop.
        """
        if self.is_manager():
            self._pool = Pool(self, manager_function, self._comm, self._root)
        elif self._comm is not None:
            comm, intracomm = comm_split(self._comm, self._root)
            logging.debug(f'Seting up worker thread {comm.Get_rank()}')
            set_comm_server(intracomm)
            FunctionWorker(comm, sync=False)
            intracomm.Free()

    @property
    def num_workers(self):
        """The number of workers assigned to this FunctionExecutor."""
        with self._lock:
            if self._shutdown:
                return 0
            self._bootstrap()
            assert self._pool is not None  # Here so mypy is happy

            self._pool.wait()
            return self._pool.size

    def submit(
            self,
            fn: Callable[..., R],
            *args,
            callback: Callable[[list], Any]=do_nothing,
            **kwargs,
        ) -> Future:
        """Submit a task to be computed by the Executor.

        Arguments:
            fn (Callable[..., R]): The function to be executed.
            *args: Arguments to pass into `function`.
            callback (Callable[R, None]): A function to call on the result
                    of `function`.
            **kwargs: Keyword arguments to pass into `function`.

        Raises:
            RuntimeError: Calling this function after the executor has shutdown.

        Returns:
            (Future): Future representing the result of the function call.
        """
        with self._lock:
            if self._shutdown:
                raise RuntimeError('Cannot submit jobs after shutdown')

            self._bootstrap()  # Ensure pool exists before submitting
            assert self._pool is not None  # Here so mypy is happy

            future: Future = Future()
            task = Task(
                TaskType.FUNCTION,
                function=fn,
                args=args,
                callback=callback,
                kwargs=kwargs,
            )
            logging.debug(f'Pushed new task {task} to pool')
            self._pool.push((future, task))
            return future

    def map(self,  # noqa: A003
        fn: Callable[..., R],
        *iterables: Iterable,
        ordered: bool=True,
        timeout: float | None=None,
    ) -> Iterator[R]:
        """Return an iterator containing all results of executing fn.

        Arguments:
            fn (Callable[..., R]): The function to be evaluated
            *iterables: Iterables containing positional arguments to pass to `fn`
            ordered (bool): Whether the output order should be the same
            timeout (float): Maximum number of seconds to wait before aborting
                execution. If None, then there is no time limit.

        Returns:
            (Iterator[R]): An iterator containig all calls to `fn`. Equivalent to the
                output of `map(fn, *iterables)`.

        Raises:
            TimeoutError: If the execution didn't finish before the time limit.
        """
        return self.starmap(fn, zip(*iterables, strict=False), ordered, timeout)

    def starmap(self,
                fn: Callable[..., R],
                iterable: Iterable[tuple],
                ordered: bool=True,
                timeout: float | None=None,
    ) -> Iterator[R]:
        """Return an iterator containing the results of fn for every set of arguments.

        Arguments:
            fn (Callable[..., R]): The function to be evaluated
            iterable: Iterable containing tuples of positional arguments to pass to `fn`
            ordered (bool): Whether the output order should be the same
            timeout (float): Maximum number of seconds to wait before aborting
                execution. If None, then there is no time limit.


        Returns:
            (Iterator[R]): An iterator containig all calls to `fn`. Equivalent to the
                output of `itertools.starmap(fn, iterable)`.

        Raises:
            TimeoutError: If the execution didn't finish before the time limit.
        """
        if timeout is not None:
            timer = time.monotonic
            end_time = timeout + timer()

        futures: list[Future] | set[Future] = []
        if ordered:
            futures = [self.submit(fn, *args) for args in iterable]
        else:
            futures = {self.submit(fn, *args) for args in iterable}

        def result(future: Future, timeout: float | None=None):
            try:
                try:
                    return future.result(timeout)
                finally:
                    future.cancel()
            finally:
                del future

        try:
            if ordered:
                assert isinstance(futures, list)  # So MyPy is happy

                futures.reverse()
                while futures:
                    res = result(futures.pop()) if timeout is None else \
                        result(futures.pop(), end_time - timer())
                    yield res
            else:
                assert isinstance(futures, set)  # So MyPy is happy

                iterator = as_completed(futures) if timeout is None else \
                    as_completed(futures, end_time - timer())
                for f in iterator:
                    futures.remove(f)
                    yield result(f)
        finally:
            while futures:
                futures.pop().cancel()


class FileExecutor(Executor):
    """Job executor for executable jobs."""
    def __init__(
            self,
            comm: MPI.Intracomm | None=None,
            root: int=0,
            max_workers: int | None=None,
            initializer: Callable | None=None,
            initargs: Sequence[Any]=(),
            **kwargs,
        ) -> None:
        """Initialize a FileExecutor object. Same arguments as Executor."""
        super().__init__(comm, root, max_workers, initializer, initargs, **kwargs)

    @override
    def setup(self):
        """Setup the FileExecutor.

        The root process with create a Pool object, while all other processes will
        exit so that the manager may spawn new processes as needed.
        """
        if self.is_manager():
            time.sleep(1)
            self._pool = Pool(self, manager_file)
        else:
            # pass
            sys.exit(0)

    def _bootstrap(self):
        """Create a pool if it doesn't yet exist."""
        if self._pool is None:
            self._pool = Pool(self, manager_file)

    @property
    def num_workers(self):
        """The number of workers assigned to this FunctionExecutor."""
        with self._lock:
            if self._shutdown:
                return 0
            self._bootstrap()
            assert self._pool is not None  # Here so mypy is happy

            self._pool.wait()
            return self._pool.size

    def submit(
            self,
            exe: str,
            exe_args: list[str] | None=None,
            num_workers=1,
            mpi_info: dict | None=None,
        ) -> Future[int]:
        """Submit a task to be computed by the Executor.

        Arguments:
            exe (str): The executable to run.
            exe_args (list[str]): Command-line arguments to be passed to `exe`.
            num_workers (int): The number of worker processes to use for `exe`.
            mpi_info (dict): Info to pass to MPI.Comm.Spawn

        Raises:
            ValueError: If num_workers is greater than the maximum amount of workers.
            RuntimeError: Calling this function after the executor has shutdown.

        Returns:
            (Future[int]): Future representing the exit code of the executable.
        """
        if mpi_info is None:
            mpi_info = {}
        if exe_args is None:
            exe_args = []
        if num_workers > self.num_workers:
            raise ValueError(
                'Cannot request more workers for a task than are available'
            )
        with self._lock:
            if self._shutdown:
                raise RuntimeError('Cannot submit jobs after shutdown')
            self._bootstrap()
            assert self._pool is not None  # Here so mypy is happy

            future: Future[int] = Future()
            task = Task(
                TaskType.EXECUTABLE,
                exe=exe,
                args=exe_args,
                num_workers=num_workers,
                mpi_info=mpi_info,
            )
            logging.debug(f'Pushed new task {task} to pool')
            self._pool.push((future, task))
            return future


def barrier(comm: MPI.Comm):
    """Wrapper for MPI.Comm.Barrier."""
    request = comm.Ibarrier()
    while not request.Test():
        time.sleep(SLEEP_TIME)


class Pool:
    """Job pool for handling processing of `Task`'s and work allocation."""

    def __init__(
            self,
            executor: Executor,
            target: Callable,
            *args,
        ):
        """Initialize a Pool object.

        Arguments:
            executor (Executor): The executor that created this pool.
            target (Callable): Function to call to setup the manager process.
            *args: Arguments to be passed to `target`.
        """
        self.executor = executor
        self.size = None
        self.queue: WorkQueue = deque()

        self.event = threading.Event()

        self.thread = threading.Thread(
            target=target,
            args=(self, executor._options, *args),
        )
        self.setup_threads()
        self.thread.daemon = not hasattr(threading, '_register_atexit')
        self.thread.start()

    def setup_queue(self, n) -> WorkQueue:
        """Setup attributes for this job pool."""
        self.size = n
        self.event.set()
        return self.queue

    def setup_threads(self):
        """Create a global lock to use when executing serial code."""
        global SERIALIZED
        global _thread_level

        with _setup_threads_lock:
            if _thread_level is None:
                _thread_level = MPI.Query_thread()
                if _thread_level < MPI.THREAD_MULTIPLE:
                    SERIALIZED = threading.Lock()
        if _thread_level < MPI.THREAD_SERIALIZED:
            logging.warning('Thread level should be at least MPI_THREAD_SERIALIZED')

    def wait(self):
        """Wait for threads to finish."""
        self.event.wait()

    def push(self, item):
        """Add work to the queue."""
        self.queue.appendleft(item)

    def done(self):
        """Signal that the pool is finished by enqueueing None."""
        self.push(None)

    def join(self):
        """Join the manager thread."""
        self.thread.join()

    def broken(self, message: str):
        """Handle errors happening within the pool execution."""
        lock = None
        ex = self.executor
        if ex is not None:
            ex._broken = message
            if not ex._shutdown:
                lock = ex._lock

        def handler(future: Future):
            if future.set_running_or_notify_cancel():
                excep = BrokenExecutorError(message)
                future.set_exception(excep)

        self.event.set()
        if lock:
            with lock:
                self.cancel(handler)


    def cancel(self, handler: Callable[[Future], None] | None=None):
        """Cancel all incomplete jobs in the pool, calling a handler if provided."""
        while True:
            try:
                item = self.queue.pop()
            except LookupError:
                break
            if item is None:
                self.push(None)
                break
            future, task = item
            if handler:
                handler(future)
            else:
                future.cancel()
                future.set_running_or_notify_cancel()


def manager_function(
        pool: Pool,
        options: dict,
        intracomm: MPI.Intracomm | None=None,
        root: int=0,
):
    """Manager target for FunctionExecutor."""
    if intracomm is None:
        # There are no workers so need to spawn some
        pyexe = options.pop('python_exe', None)
        args = options.pop('python_args', None)
        nprocs = options.pop('max_workers', 1)
        mpi_info = options.pop('mpi_info', None)
        logging.debug(f'Spawning {nprocs} FunctionWorkers...')
        comm = FunctionManager.spawn(pyexe, args, nprocs, mpi_info)
    elif intracomm.Get_size() == 1:
        logging.debug(f'comm provided; size={intracomm.Get_size()}')
        # Only one total process in communicator
        options['num_workers'] = 1
        set_comm_server(MPI.COMM_SELF)
        manager_thread(pool, options)
        return
    else:
        # There are multiple processes in the comm so split
        logging.debug(f'comm provided; size={intracomm.Get_size()}')
        comm, _ = serialized(comm_split)(intracomm, root)

    assert comm != MPI.COMM_NULL
    assert comm.Get_size() == 1
    assert comm.Is_inter()

    manager = FunctionManager()

    # Synchronize comm
    sync = options.pop('sync', True)
    manager.sync(comm, options, sync)
    if not manager.intialize(comm, options):
        logging.debug('Error encountered when calling initializer. Aborting...')
        manager.stop(comm)
        pool.broken('Error encountered in intializer')
        return

    size = comm.Get_remote_size()
    queue = pool.setup_queue(size)
    workers = set(range(size))
    logging.debug(f'Created pool of size {size} with workers: {workers}')
    manager.execute(comm, 0, workers, queue)
    manager.stop(comm)

def manager_thread(pool: Pool, options: dict):
    """Manager target when there are no worker processes.."""
    logging.debug(f'Creating manager_thread on rank {MPI.COMM_WORLD.Get_rank()}')
    size = options.pop('num_workers', 1)
    queue = pool.setup_queue(size)
    threads: deque[threading.Thread] = deque()
    max_threads = size - 1

    def adjust():
        if len(threads) < max_threads:
            thread = threading.Thread(target=worker)
            thread.start()
            threads.append(thread)

    def execute(future: Future, task: Task):
        res, e = task()
        if e is not None:
            future.set_exception(e)
        else:
            future.set_result(res)

    def worker():
        if not initialize(options):
            queue.appendleft(None)
            return
        while True:
            try:
                item = queue.pop()
            except LookupError:
                time.sleep(SLEEP_TIME)
                continue
            if item is None:
                queue.appendleft(None)
                break
            future, task = item
            if future.set_running_or_notify_cancel():
                if queue:
                    adjust()
                execute(future, task)
            del future, task, item

    worker()
    for thread in threads:
        thread.join()
    queue.pop()


def manager_file(pool: Pool, options: dict):
    """Manager target for FileExecutor."""
    manager = FileManager()

    size = options.pop('max_workers', get_max_workers())
    queue = pool.setup_queue(size)
    workers = WorkerSet(range(size))
    logging.debug(f'Created pool of size {size} with workers: {workers}')
    manager.execute(options, workers, queue)
    manager.stop()


class WorkerSet:
    """A set for containing worker ids."""

    def __init__(self, ids: Iterable[int] | None=None) -> None:
        """Initialize a WorkerSet."""
        self.ids = set(ids) if ids is not None else set()

    def __len__(self) -> int:
        """Return the number of ids."""
        return len(self.ids)

    def pop(self, n: int=1) -> Iterator[int]:
        """Get and remove ids from the set.

        Arguments:
            n (int): The number of ids to pop.


        Returns:
            if n == 1,
                (int): The worker id removed from the set
            else,
                (Iterator[int]): An iterator containing n ids that were removed


        Raises:
            ValueError: If n is less than 1 or greater than the size of the set.
        """
        if n < 1 or n > len(self):
            raise ValueError(f'Expected n in [1, ..., {len(self)}], got {n}')
        for _ in range(n):
            yield self.ids.pop()

    def add(self, n: int):
        """Add a number to the worker set."""
        self.ids.add(n)

    def union(self, s: Iterable[int]) -> WorkerSet:
        """Return the union of two sets of integers."""
        return WorkerSet(self.ids.union(s))

    def __iter__(self) -> Iterator[int]:
        """Return an iterator over the ids."""
        return self.ids.__iter__()

    def __repr__(self) -> str:
        """Return a string representation of the WorkerSet."""
        return f'WorkerSet({self.ids})'

    def __str__(self) -> str:
        """Return a readable string version of the WorkerSet."""
        return '{' + ', '.join(str(id) for id in self) + '}'

    def __hash__(self) -> int:
        """Return a hashcode for the WorkerSet."""
        return frozenset(self.ids).__hash__()

    def __bool__(self) -> bool:
        """Return whether the set is empty."""
        return len(self) > 0


class FileManager:
    """Class for spawning and running executable jobs."""

    def __init__(self):
        """Initialize a FileManager."""
        logging.debug(f'Creating FileManager on rank {MPI.COMM_WORLD.Get_rank()}')
        self.workers = WorkerSet()
        self.pending: dict[WorkerSet, tuple[Future, Path, Task]] = {}  # type: ignore

    def get_next_job_id(self) -> uuid.UUID:
        """Get a unique id number for a job."""
        return uuid.uuid4()

    def execute(
            self,
            options: dict,
            workers: WorkerSet,
            tasks: WorkQueue,
    ):
        """Execute all tasks in the provided queue.

        Arguments:
            options (dict): Options to use when executing tasks.
            workers (WorkerSet): The list of workers to use with this manager.
            tasks (WorkQueue): Queue of tasks to be executed.
        """
        self.workers = workers
        self.root_dir: Path = Path(options.pop('root_dir', './tmp'))

        while True:
            # If there are tasks left to be done and workers to assign, do so
            if len(tasks) > 0 and self.workers:
                stop = self.spawn(tasks)
                logging.debug(f'Stop the loop? {stop}')
                if stop:
                    break
            # Test to see if any pending jobs are complete yet
            idx, flag = self.testany()
            if self.pending and flag:
                logging.debug('Job finished!')
                self.read(idx)
            time.sleep(SLEEP_TIME)
        logging.debug(f'Done sending tasks. Waiting on {len(self.pending)} jobs...')
        self.readall()  # Wait for all remaining pending jobs to complete.

    @serialized
    def _get_all_out_files(self) -> list[tuple[WorkerSet, Path, Task]]:
        """Return all pending tasks, their assigned workers, and work directories."""
        return [
            (workers, outname, task)
            for workers, (_, outname, task) in self.pending.items()
        ]

    def _any_index(self, bools: list[bool]) -> tuple[int, bool]:
        """Return the index of a True boolean in a list.

        Arguments:
            bools (list[bool]): A list of boolean values.


        Returns:
            (tuple[int, bool]): The index of the first True boolean and a flag showing
                if any of the values were True. The value of flag is equivalent to
                calling the built-in `any(l)`. If no values are True, index is -1.
        """
        for i, val in enumerate(bools):
            if val:
                return (i, True)
        return (-1, False)

    @serialized
    def test(self, workers: WorkerSet, job_dir: Path, task: Task) -> bool:
        """Test to see if a task is completed.

        Completion is indicated by the scratch directory containing a file for each
        worker process containing their respective exit code.

        Arguments:
            workers (WorkerSet): The set of workers assigned to this task.
            job_dir (str): The scratch work directory used for this task.
            task (Task): The task being executed (NOT USED)


        Returns:
            (bool): Whether the task is complete.
        """
        out_dir = job_dir / 'out'

        # If output directory doesn't exist then the job is certainly not finished yet.
        if not out_dir.exists():
            return False

        # Check if the number of exit code files is equal to the number of workers
        num_files = len(
            [
                name for name in os.listdir(out_dir)
                if (out_dir / name).is_file()
        ])
        return num_files == len(workers)

    @serialized
    def testany(self) -> tuple[int, bool]:
        """Return the index of a finished job, if any exist, otherwise (-1, False)."""
        job_dir_list = self._get_all_out_files()
        exists = list(starmap(self.test, job_dir_list))
        return self._any_index(exists)

    def create_bash_script(self, task: Task) -> tuple[Path, Path]:
        """Setup a scratch work directory and create a wrapper executable.

        This wrapper executable simply calls the original executable and outputs the
        exit code to a temporary file. This is done as there is no way in MPI to block
        for child processes to finish. The scratch work directory is created to keep
        all jobs organized.

        This method will create an empty scratch work directory for the task, deleting
        any existing files there.

        Arguments:
            task (Task): The task to create the directory for.


        Returns:
            (tuple[str, str]): The wrapper executable name and scratch directory path.
                The resulting `job_dir` will be 'tmp/ID' where ID is a unique id
                generated by `get_next_job_id`.
        """
        # Make job directory for the task
        job_id = self.get_next_job_id()
        job_dir = self.root_dir / f'{job_id}'
        shutil.rmtree(job_dir, ignore_errors=True)
        job_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectory for the exitcode outputs
        out_subdir = job_dir / 'out'
        out_subdir.mkdir(exist_ok=True)
        exename = job_dir / 'exe.sh'

        # If in debug mode, add echo lines to the wrapper executable
        verbose = logging.getLogger().level <= logging.DEBUG

        arg_str = ' '.join(task.args)

        with open(exename, 'w') as f:
            if verbose:
                f.writelines([
                    '#!/bin/bash\n',
                    f'echo running "{arg_str}"\n',
                    f'{arg_str}\n',
                    f'echo "$?" >> "{out_subdir}/$OMPI_COMM_WORLD_RANK.err"\n',
                    'echo errcode from $OMPI_COMM_WORLD_RANK saved\n',
                ])
            else:
                f.writelines([
                    '#!/bin/bash\n',
                    f'{arg_str}\n',
                    f'echo "$?" >> "{out_subdir}/$OMPI_COMM_WORLD_RANK.err"\n',
                ])

        exename.chmod(exename.stat().st_mode | stat.S_IEXEC)
        return exename, job_dir

    @serialized
    def readall(self):
        """Read all remaining pending jobs."""
        while self.pending:
            i, flag = self.testany()
            if flag:
                self.read(i)
            time.sleep(SLEEP_TIME)

    @serialized
    def read(self, idx: int):
        """Read exit codes from a completed task and update the corresponding Future."""
        # Get all job information for specified index
        workers = list(self.pending.keys())[idx]
        future, job_dir, task = self.pending.pop(workers)

        assert job_dir.exists()

        logging.info(
            f"...Done executing 'mpirun -n {task.num_workers}"
            f' {" ".join(task.args)}\'\n'
        )

        # Read out all exit codes in the job scratch directory
        out_dir = job_dir / 'out'

        err_files = out_dir.iterdir()
        errcodes = [1] * len(workers)
        for i, fname in enumerate(err_files):
            with open(fname) as f:
                errcodes[i] = int(f.readline().rstrip())

        logging.debug(f'errcodes={errcodes}')

        # Make the workers available again
        self.workers = self.workers.union(workers)

        # If any exit code was non-zero, the result is 1
        res = int(any(code != 0 for code in errcodes))
        future.set_result(res)

        # Delete temporary work directory unless in debug mode
        if logging.getLogger().level > logging.DEBUG:
            shutil.rmtree(job_dir)

    @serialized
    def spawn(self, tasks: WorkQueue) -> bool:
        """Spawn worker processes to run a task.

        Arguments:
            tasks (WorkQueue): Queue of tasks being executed.

        Returns:
            (bool): Whether the main execution loop should finish, i.e. whether all
                tasks have been started.
        """
        logging.debug(f'tasks before spawn: {tasks}, {len(tasks)}')
        logging.debug(f'pending before spawn: {self.pending}, {len(self.pending)}')

        # If no tasks left keep looping
        try:
            item = tasks.pop()
        except LookupError:
            logging.debug('No tasks queued, continuing loop')
            return False

        # None signals that the Pool is finished
        if item is None:
            logging.debug('`None` popped from tasks')
            return True

        future, task = item

        nprocs = task.num_workers

        # If there aren't enough free workers for this task continue loop.
        if nprocs > len(self.workers):
            logging.debug('Not enough workers yet, adding back to queue')
            tasks.appendleft(item)  # Add item back to queue
            return False

        # If the future was canceled we're done
        if not future.set_running_or_notify_cancel():
            logging.debug('Future canceled, continuing loop')
            return False


        ids = WorkerSet(self.workers.pop(n=nprocs))
        exename, job_dir = self.create_bash_script(task)

        # Spawn the tasks and update pending list
        try:
            logging.info(f'Executing \'mpirun -n {nprocs} {" ".join(task.args)}\'...\n')
            MPI.COMM_SELF.Spawn(str(exename), maxprocs=nprocs)
            self.pending[ids] = (future, job_dir, task)
        except BaseException as e:
            logging.exception('ERROR OCCURRED DURING SPAWN!!')
            self.workers = self.workers.union(ids)
            future.set_exception(e)

        logging.debug('Task succesfully spawned')
        logging.debug(f'tasks after spawn: {tasks}, {len(tasks)}')
        logging.debug(f'pending after spawn: {self.pending}, {len(self.pending)}')
        return False

    def stop(self):
        """Disconnect from the communicator."""
        logging.debug('Stopping FileManager')
        # if MPI.COMM_SELF != MPI.COMM_NULL and MPI.COMM_SELF != MP:


class FunctionWorker:
    """Class for running jobs server-side."""

    def __init__(self, comm=None, sync=True):
        """Initialize a FunctionWorker."""
        logging.debug(
            f'Created FunctionWorker on local rank {MPI.COMM_WORLD.Get_rank()}'
        )
        if comm is None:
            self.spawn()
        else:
            self.main(comm, sync=sync)

    def spawn(self):
        """Get parent communicator. Called if this process was spawned."""
        comm = MPI.Comm.Get_parent()
        set_comm_server(MPI.COMM_WORLD)
        self.main(comm)

    def main(self, comm: MPI.Intercomm, sync: bool=True):
        """Synchronize options with other processes and begin execution."""
        assert comm.Is_inter()
        self.sync(comm, sync=sync)

        init_options = comm.bcast(None, 0)
        success = initialize(init_options)
        sbuf = bytearray([success])
        rbuf = bytearray([True])
        comm.Allreduce(sbuf, rbuf, op=MPI.LAND)

        self.execute(comm)
        self.stop(comm)

    def sync(self, comm: MPI.Intercomm, sync: bool):
        """Synchronize options with all other workers and import main module."""
        barrier(comm)
        options = comm.bcast(None, 0)

        if sync:
            if 'path' in options:
                sys.path.extend(options.pop('path'))
            if 'wdir' in options:
                os.chdir(options.pop('wdir'))
            if 'env' in options:
                os.environ.update(options.pop('env'))
            mod_name = options.pop('@main:mod_name', None)
            mod_path = options.pop('@main:mod_path', None)
            mod_glbs = options.pop('globals', None)
            import_main(mod_name, mod_path, mod_glbs, MAIN_RUN_NAME)

        return options

    def execute(self, comm: MPI.Intercomm):
        """Main execution loop. Wait for jobs and send back results."""
        status = MPI.Status()

        while True:
            task: Task = self.recv(comm, MPI.ANY_TAG, status)

            # None is signal to stop the loop
            if task is None:
                logging.debug(f'Worker {comm.Get_rank()}: Received End signal')
                break

            logging.debug(
                f'Executing task {task} on rank '
                f'({MPI.COMM_WORLD.Get_rank()}, {comm.Get_rank()})'
            )
            res = task()
            self.send(comm, status, res)


    @serialized
    def recv(
        self,
        comm: MPI.Intercomm,
        tag: int,
        status: MPI.Status,
    ) -> Task | BaseException:
        """Wait for and receive work from manager.

        Arguments:
            comm (MPI.Intercomm): Communicator that task is sent through.
            tag (int): Tag to be used in recv.
            status (MPI.Status): Status to use with recv.


        Returns:
            (Task | BaseException): The received task. If an error occurs returns the
                exception instead.
        """
        # Wait for a message to be sent
        logging.debug(f'Worker {comm.rank}: Waiting for work...')
        while not comm.iprobe(MPI.ANY_SOURCE, MPI.ANY_TAG, status):
            time.sleep(SLEEP_TIME)

        # Actually receive the work
        id, tag = status.source, status.tag
        try:
            task: Task = comm.recv(None, source=id, tag=tag, status=status)
            logging.debug(f'Worker {comm.rank}: Received task {task}')
        except BaseException as e:
            return e
        return task

    def send(
            self,
            comm: MPI.Intercomm,
            status: MPI.Status,
            result: tuple[Any | None, None | BaseException]
    ):
        """Send completed task back to manager.

        Arguments:
            comm (MPI.Intercomm): Communicator to send message through.
            status (MPI.Status): Status to use in message.
            result (tuple[Any | None, None | BaseException]): The result of the task
                execution. If no error occured, equal to (ret_value, None). If an error
                happened (either during execution or sending) equal to (None, exc).
        """
        id, tag = status.source, status.tag
        logging.debug(f'Worker {comm.rank}: Sending completed {result}')
        try:
            request = comm.issend(result, id, tag)
        except BaseException as e:
            result = (None, e)
            request = comm.issend(result, id, tag)

        # Wait for the message to be received
        while not request.test()[0]:
            time.sleep(SLEEP_TIME)

    def stop(self, comm: MPI.Intercomm):
        """Disconnect this process from the communicator."""
        logging.debug(f'Disconnecting FunctionWorker {comm.Get_rank()}')
        comm.Free()

class FunctionManager:
    """Class for running jobs; client-side."""

    def __init__(self):
        """Initialize a FunctionManager."""
        logging.debug(f'Creating FunctionManager on rank {MPI.COMM_WORLD.Get_rank()}')
        self.workers = set()
        self.pending: dict[int, tuple[Future, MPI.Request]] = {}  # type: ignore

    def intialize(self, comm: MPI.Intercomm, options):
        """Send initializer options to all worker processes."""
        keys = ('initializer', 'initargs', 'initkwargs')
        default_vals: tuple[str | None, Iterable[str], dict] = (None, (), {})

        # Get all the keys from options, using defaults where necessary
        data = {k: options.pop(k, v) for k, v in zip(keys, default_vals, strict=False)}
        serialized(MPI.Comm.bcast)(comm, data, MPI.ROOT)

        sbuf = bytearray([False])
        rbuf = bytearray([False])
        serialized(MPI.Comm.Allreduce)(comm, sbuf, rbuf, op=MPI.LAND)
        return bool(rbuf[0])

    def _sync_data(self, options: dict) -> dict:
        """Get synchronization data."""
        main = sys.modules['__main__']
        sys.modules.setdefault(MAIN_RUN_NAME, main)
        import_main_module = options.pop('main', True)

        data = options.copy()
        data.pop('initializer', None)
        data.pop('initargs', None)
        data.pop('initkwargs', None)

        if import_main_module:
            spec = getattr(main, '__spec__', None)
            name = getattr(spec, 'name', None)
            path = getattr(main, '__file__', None)
            if name is not None:  # pragma: no cover
                data['@main:mod_name'] = name
            if path is not None:  # pragma: no branch
                data['@main:mod_path'] = path

        return data


    def sync(self, comm: MPI.Intracomm, options: dict, sync: bool):
        """Synchronize options with all workers."""
        serialized(barrier)(comm)
        if sync:
            options = self._sync_data(options)
        serialized(MPI.Comm.bcast)(comm, options, MPI.ROOT)

    @staticmethod
    @serialized
    def spawn(
        python_exe: str | None=None,
        python_args: list[str] | None=None,
        nprocs: int | None=None,
        mpi_info: dict | None=None
    ) -> MPI.Intercomm:
        """Spawn worker processes.

        Arguments:
            python_exe (str): Python executable target.
            python_args (str): Arguments to pass to python call. Will always prepend
                '-m vipdopt.server'.
            nprocs (int): Number of worker processes to spawn. Will attempt to calculate
                from environment variables if None.
            mpi_info (dict): Info to pass to MPI.Comm.Spawn call.

        Returns:
            (MPI.Intercomm): Intercomm created by spawn.
        """
        max_workers = get_max_workers()
        if nprocs is None:
            nprocs = max_workers
        elif nprocs > max_workers:
            raise ValueError(f'Expected <= {max_workers} to spawn. Received {nprocs}')

        # Use dummy ecxecutable if none provided
        pyexe = sys.executable if python_exe is None else python_exe

        # Add vipdopt.mpi.server module to be run with same log level
        pyargs = ['-m','vipdopt.mpi.server', str(logging.getLogger().level)]
        given_args = [] if python_args is None else list(python_args)

        pyargs.extend(given_args)

        # Create MPI.Info object
        info = MPI.Info.Create()
        if mpi_info is None:
            mpi_info = {'soft': f'1:{nprocs}'}

        info.update(mpi_info)


        comm = MPI.COMM_SELF.Spawn(pyexe, pyargs, maxprocs=nprocs, info=info)
        info.Free()

        return comm

    def execute(
            self,
            comm: MPI.Intercomm,
            tag: int,
            workers: set[int],
            tasks: WorkQueue,
    ):
        """Main execution loop."""
        self.workers = workers
        status = MPI.Status()

        while True:
            # If there are tasks to do and workers available send a job
            if len(tasks) > 0 and self.workers:
                stop = self.send(comm, tag, tasks)
                logging.debug(f'Stop the loop? {stop}')
                if stop:
                    break
            # If there is a completed job, receive it
            if self.pending and self.iprobe(comm, tag, status):
                self.recv(comm, tag, status)
            time.sleep(SLEEP_TIME)
        logging.debug(f'Done sending tasks. Waiting on {len(self.pending)} jobs...')

        # Wait for remaining jobs to finish
        while self.pending:
            self.recv(comm, tag, status)

    def probe(self, comm: MPI.Intercomm, tag: int, status: MPI.Status):
        """Wait until a job is complete."""
        while not self.iprobe(comm, tag, status):
            time.sleep(SLEEP_TIME)

    @serialized
    def iprobe(self, comm: MPI.Intercomm, tag: int, status: MPI.Status) -> bool:
        """Return whether there is an incoming message to receive."""
        return comm.iprobe(MPI.ANY_SOURCE, tag, status)

    @serialized
    def issend(self, comm: MPI.Intercomm, obj: Any, dest: int, tag: int) -> MPI.Request:
        """Nonblocking send a message."""
        return comm.issend(obj, dest, tag)

    def recv(self, comm: MPI.Intercomm, tag: int, status: MPI.Status):
        """Receive a completed task from a worker.

        Arguments:
            comm (MPI.Intercomm): Intercomm to receive message through.
            tag (int): Tag to use when receiving.
            status (MPI.Status): Status to use when receiving.
        """
        # Receive completed task
        logging.debug('Receiving completed task...')
        try:
            item = serialized(MPI.Comm.recv)(comm, None, MPI.ANY_SOURCE, tag, status)
        except BaseException as e:
            item = (None, e)

        # Check which worker did the task and add it back to the WorkerSet
        source_id = status.source
        self.workers.add(source_id)
        logging.debug(
            f'Received completed task with results "{item}" from worker {source_id}')

        future, request = self.pending.pop(source_id)
        logging.debug(f'Num pending after receipt: {len(self.pending)}')

        serialized(MPI.Request.Free)(request)

        # Set future appropriately
        res, exception = item
        if exception is None:
            future.set_result(res)
        else:
            future.set_exception(exception)

    def send(
            self,
            comm: MPI.Intercomm,
            tag: int,
            tasks: WorkQueue,
    ) -> bool:
        """Send a task to a worker.

        Arguments:
            comm (MPI.Intercomm): Intercomm to spawn the worker
            tag (int): Tag to use with sent message.
            tasks (WorkQueue): Queue of tasks being executed.

        Returns:
            (bool): Whether the main execution loop should finish, i.e. whether all
                tasks have been started.
        """
        logging.debug(f'tasks before send: {tasks}, {len(tasks)}')
        logging.debug(f'pending before send: {self.pending}, {len(self.pending)}')

        # If no tasks left keep looping
        try:
            item = tasks.pop()
        except LookupError:
            return False

        # None signals that the Pool is finished
        if item is None:
            return True

        future, task = item

        # If the future was canceled we're done
        if not future.set_running_or_notify_cancel():
            return False

        worker_id = self.workers.pop()

        # Try sending the task, catching the exception if one occurs
        try:
            logging.debug(
                f'Client {comm.Get_rank()}: Sending task {task} to rank {worker_id}'
            )
            # Non-blocking send so we can start other jobs while waiting for this one
            request = self.issend(comm, task, worker_id, tag)
            self.pending[worker_id] = (future, request)
        except BaseException as e:
            self.workers.add(worker_id)
            future.set_exception(e)

        logging.debug(f'tasks after send: {tasks}, {len(tasks)}')
        logging.debug(f'pending after send: {self.pending}, {len(self.pending)}')
        return False

    @serialized
    def send_to_all(self, comm: MPI.Intercomm, obj: Any, tag=0):
        """Send a message to all worker processes."""
        size = comm.Get_remote_size()
        requests = [self.issend(comm, obj, s, tag) for s in range(size)]
        MPI.Request.waitall(requests)

    def stop(self, comm: MPI.Intercomm):
        """Send stop signal to all workers and disconnect."""
        logging.debug('Stopping FunctionManager')
        self.send_to_all(comm, None)  # None is the stop signal
        logging.debug('All Stop signals received')
        serialized(MPI.Comm.Free)(comm)
        time.sleep(1)  # To make sure that all children are freed before continuing code


if __name__ == '__main__':
    # Example usage

    log_level = logging.DEBUG
    logging.basicConfig(level=log_level)

    logging.debug(MPI.Get_library_version())

    rank = MPI.COMM_WORLD.Get_rank()

    with FunctionExecutor() as ex:
        logging.debug(f'Number of workers: {ex.num_workers}')
        logging.debug(f'Maximum workers: {get_max_workers()}')

        for res in ex.map(abs, (-1, -2, 3, 4, -5, 6), ordered=True, timeout=5):
            logging.info(res)

        for res in ex.starmap(sum, ((range(3),), (range(6),), (range(9),)), timeout=5):
            logging.info(res)

    with FunctionExecutor(max_workers=3) as ex:
        future = ex.submit(max, range(3), default=0)
        logging.info(future.result())

    with FileExecutor() as ex:
        res = ex.submit(
            'python',
            ['testing/mpitest.py', '-l', f'{log_level}'],
            num_workers=2,
        )
        logging.info(f'Result of running mpitest.py: {res.result()}')
