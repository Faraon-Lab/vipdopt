import os
import sys
import threading
from collections import deque as Deque
from typing import Any, Callable, Iterable
from concurrent.futures import Future, as_completed, wait
import time
import logging
from copy import deepcopy


import numpy as np
from mpi4py import MPI

from vipdopt.utils import R, T



SERIALIZED = None
_thread_level = None
MAIN_RUN_NAME = '__worker__'
_setup_threads_lock = threading.Lock()
_tls = threading.local()
SLEEP_TIME = 0.001


def identity(x: Any) -> Any:
    return x

class Task:
    """Wrapper for pool tasks and associated data."""
    def __init__(self, func: Callable, *args: Any, callback: Callable=identity, **kwargs):
        self._func = func
        self._args = args
        self._kwargs = kwargs
        self._callback = callback
    
    def __str__(self) -> str:
        return f'"Task: {self._func.__name__}({self._args}, {self._kwargs})"'

    def __call__(self) -> tuple[Any | None, None | BaseException]:
        logging.debug(f'Calling Task {self._func}({self._args}, {self._kwargs})')
        try:
            res = self._func(*self._args, **self._kwargs)
            self._callback(res)
            return (res, None)
        except BaseException as e:
            return (None, e)

    def __iter__(self) -> Iterable:
        yield self._func
        yield self._args
        yield self._kwargs
    

def get_max_workers():
    max_workers = os.environ.get('MAX_WORKERS')
    if max_workers is not None:
        logging.debug('non None max_workers')
        return int(max_workers)
    if MPI.UNIVERSE_SIZE != MPI.KEYVAL_INVALID:
        usize = MPI.COMM_WORLD.Get_attr(MPI.UNIVERSE_SIZE)
        if usize is not None:
            wsize = MPI.COMM_WORLD.Get_size()
            logging.debug(f'usize: {usize}; wsize: {wsize}')
            return max(usize - wsize, 1)
    return 1

def set_comm_server(intracomm: MPI.Intracomm):
    global _tls
    _tls.comm_server = intracomm

def initialize(options):
    initializer = options.pop('initializer', None)
    initargs = options.pop('initargs', ())
    initkwargs = options.pop('initkwargs', {})
    if initializer is not None:
        try:
            initializer(*initargs, **initkwargs)
            return True
        except BaseException:
            return False
    return True

def import_main(mod_name: str, mod_path: str, init_globals: dict, run_name: str):
    import types
    import runpy

    module = types.ModuleType(run_name)
    if init_globals is not None:
        module.__dict__.update(init_globals)
        module.__name__ = run_name

    class TempModulePatch(runpy._TempModule):
        def __init__(self, mod_name):
            super().__init__(mod_name)
            self.module = module

    TempModule = runpy._TempModule 
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
        runpy._TempModule = TempModule

def serialized(function):
    def wrapper(*args, **kwargs):
        with SERIALIZED:
            return function(*args, **kwargs)
    if SERIALIZED is None:
        return function
    else:
        return wrapper

def comm_split(comm: MPI.Intracomm, root: int) -> tuple[MPI.Intercomm, MPI.Intracomm]:
    if comm.Get_size() == 1:
        return MPI.Intercomm(MPI.COMM_NULL), MPI.Intracomm(MPI.COMM_NULL)

    rank = comm.Get_rank()
    full_group = comm.Get_group()
    group = full_group.Incl([root]) if rank == root else full_group.Excl([root])
    full_group.Free()
    intracomm = comm.Create(group)
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

class Executor:
    """Job executor."""

    def __init__(self, comm: MPI.Intracomm=None, root: int=0, max_workers: int=None, initializer=None, initargs=(), **kwargs) -> None:
        if comm is None:
            comm = MPI.COMM_WORLD
        if comm.Is_inter():
            raise ValueError(f'Expected an intracommunicator, received {comm}')
        if root < 0 or root >= comm.Get_size():
            raise ValueError(f'Expected a root rank in range'
                             f'[0, ..., {comm.Get_size() - 1}]. Got {root}')
        
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
        self._lock = threading.Lock()
        self._pool = None
    
    def _bootstrap(self):
        if self._pool is None:
            self._pool = Pool(self)
    
    def bootup(self, wait=True):
        with self._lock:
            if self._shutdown:
                raise RuntimeError('Cannot bootu pafter shutdown')
            if wait:
                self._pool.wait()
            return self
    
    def setup(self):
        if self.is_manager():
            self._pool = Pool(self, self._comm, False, self._root)
        else:
            comm, intracomm = comm_split(self._comm, self._root)
            logging.debug(f'Seting up worker thread {comm.Get_rank()}')
            set_comm_server(intracomm)
            ServerWorker(comm, sync=False)
            intracomm.Free()
    
    @property
    def num_workers(self):
        with self._lock:
            if self._shutdown:
                return 0
            self._bootstrap()
            self._pool.wait()
            return self._pool.size

    def is_manager(self):
        return self._comm.Get_rank() == self._root
    
    def submit(self, fn: Callable, *args, num_workers=1, callback: Callable[[list], Any]=identity, **kwargs) -> Future:
        if num_workers > self.num_workers:
            raise ValueError('Cannot request more workers for a task than are available')
        with self._lock:
            if self._shutdown:
                raise RuntimeError('Cannot submit jobs after shutdown')
            self._bootstrap()
            future = Future()
            task = Task(fn, *args, callback=callback, **kwargs)
            logging.debug(f'Pushed new task {task} to pool')
            self._pool.push((future, task))
            return future
    
    def map(self,
            fn: Callable[..., R],
            *iterable: Iterable,
            ordered: bool=True,
            timeout: float | None=None,
    ) -> Iterable[R]:
        return self.starmap(fn, zip(*iterable), ordered, timeout)
    
    def starmap(self,
                fn: Callable[..., R],
                iterables: Iterable[Iterable],
                ordered: bool=True,
                timeout: float | None=None,
    ) -> Iterable[R]:
        if timeout is not None:
            timer = time.monotonic
            end_time = timeout + timer()

        futures = [self.submit(fn, *args) for args in iterables]
        if not ordered:
            futures = set(futures)

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
                futures.reverse()
                while futures:
                    res = result(futures.pop()) if timeout is None else result(futures.pop(), end_time - timer())
                    yield res
            else:
                iterator = as_completed(futures) if timeout is None else as_completed(futures, end_time - timer())
                for f in iterator:
                    futures.remove(f)
                    yield result(f)
        finally:
            while futures:
                futures.pop().cancel()

    
    def __enter__(self):
        ex = self if self.is_manager() else None
        self.setup()
        self._executor = ex

        return ex
    
    def __exit__(self, *args):
        ex = self._executor
        self._executor = None

        if ex is not None:
            self.shutdown(wait=True)
            return False
        return True
    
    def shutdown(self, wait: bool=True, cancel_futures: bool=False):
        with self._lock:
            if not self._shutdown:
                self._shutdown = True
                if self._pool is not None:
                    self._pool.done()
            if cancel_futures:
                if self._pool is not None:
                    self._pool.cancel()
            pool = None
            if wait:
                pool = self._pool
                self._pool = None
        if pool is not None:
            pool.join()

def barrier(comm: MPI.Intercomm):
    request = comm.Ibarrier()
    while not request.Test():
        time.sleep(SLEEP_TIME)

class Pool:
    """Worker pool, handles job assignment etc."""

    def __init__(self, executor: Executor, comm: MPI.Comm=None, sync: bool=True, *args) -> None:
        self.size = None
        self.queue: Deque[tuple[Future, Task]] = Deque()

        self.event = threading.Event()

        self.thread = threading.Thread(target=manager, args=(self, executor._options, comm, sync, *args))
        self.setup_threads()
        self.thread.daemon = not hasattr(threading, '_register_atexit')
        self.thread.start()

    def setup_queue(self, n) -> Deque[tuple[Future, Task]]:
        self.size = n
        self.event.set()
        return self.queue

    def setup_threads(self):
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
        self.event.wait()

    def push(self, item):
        self.queue.appendleft(item)
    
    def done(self):
        self.push(None)
    
    def join(self):
        self.thread.join()

    def cancel(self, handler=None):
        while True:
            try:
                item = self.queue.pop()
            except LookupError as e:
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
            del future, item, task

def manager(pool: Pool, options: dict, comm: MPI.Intracomm=None, sync=True, *args):
    client = ClientWorker()

    if comm is None:
        pyexe = options.pop('python_exe')
        args = options.pop('python_args')
        nprocs = options.pop('num_workers')
        mpi_info = options.pop('mpi_info')

        comm = client.spawn(pyexe, args, nprocs, mpi_info)
    else:
        logging.debug(f'comm provided; size={comm.Get_size()}')
        if comm.Get_size() == 1:
            options['num_workers'] = 1
            set_comm_server(MPI.COMM_SELF)
            manager_thread(pool, options)
            return
        root = args[0]
        comm, _ = serialized(comm_split)(comm, root)

    # Synchronize comm
    client.sync(comm, options, sync)
    if not client.intialize(comm, options):
        client.stop(comm)
        return

    size = comm.Get_remote_size()
    queue = pool.setup_queue(size)
    workers = set(range(size))
    logging.debug(f'Created pool of size {size} with workers: {workers}')
    client.execute(comm, options, 0, workers, queue)
    client.stop(comm)

def manager_thread(pool: Pool, options: dict):
    logging.debug(f'Creating manager_thread on rank {MPI.COMM_WORLD.Get_rank()}')
    size = options.pop('num_workers', 1)
    queue = pool.setup_queue(size)
    threads: Deque[threading.Thread] = Deque()
    max_threads = size - 1
    
    def adjust():
        if len(threads) < max_threads:
            thread = threading.Thread(target=worker)
            thread.start()
            threads.append(thread)
    
    def execute(future: Future, task: tuple[Callable, tuple[Any, ...], dict]):
        func, args, kwargs = task
        res = None
        try:
            res = func(*args, **kwargs)
            future.set_result(res)
        except BaseException as e:
            future.set_exception(e)
        del res, func, args, kwargs, future, task
    
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


class ServerWorker:
    """Class for running jobs server-side."""

    def __init__(self, comm=None, sync=True):
        logging.debug(f'Creating ServerWorker on rank {MPI.COMM_WORLD.Get_rank()}')
        if comm is None:
            self.spawn()
        else:
            self.main(comm, sync=sync)
    
    def spawn(self):
        comm = MPI.Comm.Get_parent()
        set_comm_server(MPI.COMM_WORLD)
        self.main(comm)

    def main(self, comm: MPI.Intercomm, sync: bool=True):
        options = self.sync(comm, sync=sync)

        init_options = comm.bcast(None, 0)
        success = initialize(init_options)
        sbuf = bytearray([success])
        rbuf = bytearray([True])
        comm.Allreduce(sbuf, rbuf, op=MPI.LAND)

        self.execute(comm)
        self.stop(comm)
    
    def sync(self, comm: MPI.Intracomm, sync: bool):
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

        # mod_name = options.pop('@main:mod_name', None)
        # mod_path = options.pop('@main:mod_path', None)
        # mod_glbs = options.pop('globals', None)
        # import_main(mod_name, mod_path, mod_glbs, MAIN_RUN_NAME)

        return options
    
    def execute(self, comm: MPI.Intercomm):
        status = MPI.Status()

        while True:
            task: Task = self.recv(comm, MPI.ANY_TAG, status)
            if task is None:
                logging.debug(f'Worker {comm.Get_rank()}: Received End signal')
                break
            logging.debug(f'Executing task {task} on rank ({MPI.COMM_WORLD.Get_rank()}, {comm.Get_rank()})')
            res = task()
            self.send(comm, status, res)

    @serialized
    def recv(self, comm: MPI.Intercomm, tag: int, status: MPI.Status) -> Task | BaseException:
        logging.debug(f'Worker {comm.rank}: Waiting for work...')
        while not comm.iprobe(MPI.ANY_SOURCE, MPI.ANY_TAG, status):
            time.sleep(SLEEP_TIME)

        id, tag = status.source, status.tag
        try:
            task: Task = comm.recv(None, source=id, tag=tag, status=status)
            logging.debug(f'Worker {comm.rank}: Received task {task}')
        except BaseException as e:
            task = e
        return task
    
    def send(self, comm: MPI.Intercomm, status: MPI.Status, result: tuple[Any | None, None | BaseException]):
        id, tag = status.source, status.tag
        logging.debug(f'Worker {comm.rank}: Sending completed {result}')
        try:
            request = comm.issend(result, id, tag)
        except BaseException as e: 
            result = (None, e)
            request = comm.issend(result, id, tag)
        while not request.test()[0]:
            time.sleep(SLEEP_TIME)

    def stop(self, comm: MPI.Intercomm):
        comm.Disconnect()

class ClientWorker:
    """Class for running jobs; client-side."""

    def __init__(self):
        logging.debug(f'Creating ClientWorker on rank {MPI.COMM_WORLD.Get_rank()}')
        self.workers = set()
        self.pending: dict[int, tuple[Future, list[MPI.Request]]] = dict()
    
    def intialize(self, comm: MPI.Intercomm, options):
        keys = ('initializer', 'initargs', 'initkwargs')
        vals = (None, (), {})
        data = {k: options.pop(k, v) for k, v in zip(keys, vals)}
        serialized(MPI.Comm.bcast)(comm, data, MPI.ROOT)

        sbuf = bytearray([False])
        rbuf = bytearray([False])
        serialized(MPI.Comm.Allreduce)(comm, sbuf, rbuf, op=MPI.LAND)
        return bool(rbuf[0])
    
    def _sync_data(self, options):
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
        serialized(barrier)(comm)
        if sync:
            options = self._sync_data(options)
        serialized(MPI.Comm.bcast)(comm, options, MPI.ROOT)
    
    @serialized
    def spawn(self, python_exe=None, python_args=None, nprocs: int=None, mpi_info: dict={}) -> MPI.Intercomm:
        # Create use dummy ecxecutable if none provided
        pyexe = sys.executable if python_exe is None else python_exe

        pyargs = [] if python_args is None else list(python_args)
        
        # Add vipdopt.server module to be run
        pyargs.extend(['-m', __spec__.parent + '.server'])

        # Create MPI.Info object
        info = MPI.Info()
        info.update(mpi_info)

        if nprocs is None:
            nprocs = get_max_workers()

        comm = MPI.COMM_SELF.Spawn(pyexe, pyargs, maxprocs=nprocs, info=info)
        info.Free()

        return comm
    
    def execute(self, comm: MPI.Intercomm, options: dict, tag: int, workers: set[int], tasks: Deque[tuple[Future, Task]]):
        self.workers = workers
        status = MPI.Status()

        while True:
            if len(tasks) > 0 and workers:
                stop = self.send(comm, tag, tasks)
                logging.debug(f'Stop the loop? {stop}')
                if stop:
                    break
            # if self.pending:
            #     logging.debug('Pending is not empty!')
            if self.pending and self.iprobe(comm, tag, status):
                self.recv(comm, tag, status)
            time.sleep(SLEEP_TIME)
        logging.debug(f'Done sending tasks. Waiting on {len(self.pending)} jobs...')
        while self.pending:
            logging.debug('Client waiting for results...')
            self.probe()
            self.recv()

    def probe(self, comm: MPI.Intercomm, tag: int, status: MPI.Status):
        while not self.iprobe(comm, tag, status):
            time.sleep(SLEEP_TIME)

    @serialized
    def iprobe(self, comm: MPI.Intercomm, tag: int, status: MPI.Status) -> bool:
        return comm.iprobe(MPI.ANY_SOURCE, tag, status)

    @serialized
    def issend(self, comm: MPI.Intercomm, obj: Any, dest: int, tag: int) -> MPI.Request:
        return comm.issend(obj, dest, tag)
    
    def recv(self, comm: MPI.Intercomm, tag: int, status: MPI.Status):
        logging.debug(f'Waiting for completed task...')
        try:
            task = serialized(MPI.Comm.recv)(comm, None, MPI.ANY_SOURCE, tag, status)
        except BaseException as e:
            task = (None, e)

        source_id = status.source
        self.workers.add(source_id)
        logging.debug(f'Received completed task: {task} from worker {source_id}')

        future, request = self.pending.pop(source_id)
        logging.debug(f'Num pending after receipt: {len(self.pending)}')
        
        serialized(MPI.Request.Free)(request)
        res, exception = task
        if exception is None:
            future.set_result(res)
        else:
            future.set_exception(exception)
        
        # del res, exception, future, task
    
    def send(self, comm: MPI.Intercomm, tag: int, tasks: Deque[tuple[Future, Task]]) -> bool:
        logging.debug(f'tasks before send: {tasks}, {len(tasks)}')
        logging.debug(f'pending before send: {self.pending}, {len(self.pending)}')
        try:
            item = tasks.pop()
        except LookupError:
            logging.debug('1')
            return False

        if item is None:
            logging.debug('2')
            return True

        future, task = item

        if not future.set_running_or_notify_cancel():
            logging.debug('3')
            tasks.appendleft(item)
            return False
        
        worker_id = self.workers.pop()
        
        try:
            logging.debug(f'Client {comm.Get_rank()}: Sending task {task} to rank {worker_id}')
            request = self.issend(comm, task, worker_id, tag)
            self.pending[worker_id] = (future, request)
        except BaseException as e:
            self.workers.add(worker_id)
            future.set_exception(e)
        
        del future, task, item
        logging.debug(f'tasks after send: {tasks}, {len(tasks)}')
        logging.debug(f'pending after send: {self.pending}, {len(self.pending)}')
        logging.debug('5')
        return False
    
    @serialized
    def send_to_all(self, comm: MPI.Intercomm, obj: Any, tag=0):
        size = comm.Get_remote_size()
        requests = [self.issend(comm, obj, s, tag) for s in range(size)]
        MPI.Request.waitall(requests)
    
    def stop(self, comm: MPI.Intercomm):
        logging.debug('Stopping Client')
        self.send_to_all(comm, None)
        serialized(MPI.Comm.Disconnect)(comm)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    rank = MPI.COMM_WORLD.Get_rank()

    with Executor() as ex:
        logging.debug(f'Number of workers: {ex.num_workers}')
        logging.debug(f'Maximum workers: {get_max_workers()}')

        for res in ex.map(abs, (-1, -2, 3, 4, -5, 6), ordered=True, timeout=5):
            print(res)
        
        for res in ex.starmap(sum, ((range(3),), (range(6),), (range(9),)), timeout=5):
            print(res)
        
        future = ex.submit(max, range(3), default=0)
        print(future.result())
    
