"""Code for handling the connection to the FDTD solver."""

from __future__ import annotations

import abc
import contextlib
import functools
import time
import typing
from collections.abc import Callable
from functools import partial
from typing import Any, Concatenate, overload

import numpy as np
import numpy.typing as npt
from overrides import override

import vipdopt
from vipdopt.simulation import LumericalSimObjectType, LumericalSimulation, Simulation
from vipdopt.utils import (
    P,
    Path,
    R,
    ensure_path,
)


class ISolver(abc.ABC):
    """Class representing FDTD solver software."""

    @abc.abstractmethod
    def connect(self, *args, **kwargs) -> None:
        """Connect to the FDTD solver software."""

    @abc.abstractmethod
    def addjob(self, *args, **kwargs) -> None:
        """Enqueue a job to run."""

    @abc.abstractmethod
    def clearjobs(self, *args, **kwargs):
        """Remove all queued jobs."""

    @abc.abstractmethod
    def runjobs(self, *args, **kwargs):
        """Run all queued jobs."""

    @abc.abstractmethod
    def run(self, *args, **kwargs):
        """Run the FDTD solver."""

    @abc.abstractmethod
    def close(self, *args, **kwargs) -> None:
        """Close the connection with the FDTD solver software."""

    @abc.abstractmethod
    @ensure_path
    def load(self, path: Path):
        """Load data into the solver from a file."""

    @abc.abstractmethod
    @overload
    @ensure_path
    def save(self, path: Path): ...

    @abc.abstractmethod
    @overload
    @ensure_path
    def save(self, path: Path, sim: Simulation): ...

    @abc.abstractmethod
    @ensure_path
    def save(self, path: Path, sim: Simulation | None = None):
        """Save a simulation using the FDTD solver software."""


def _check_lum_fdtd(
    func: Callable[Concatenate[LumericalFDTD, P], R],
) -> Callable[Concatenate[LumericalFDTD, P], R]:
    @functools.wraps(func)
    def wrapped(fdtd: LumericalFDTD, *args: P.args, **kwargs: P.kwargs):
        if fdtd.fdtd is None:
            raise UnboundLocalError(
                f'Cannot call {func.__name__} before `fdtd` is instantiated.'
                ' Has `connect()` been called?'
            )
        assert fdtd.fdtd is not None
        return func(fdtd, *args, **kwargs)

    return wrapped


def _sync_lum_fdtd_solver(
    func: Callable[Concatenate[LumericalFDTD, P], R],
) -> Callable[Concatenate[LumericalFDTD, P], R]:
    @functools.wraps(func)
    def wrapped(fdtd: LumericalFDTD, *args: P.args, **kwargs: P.kwargs):
        fdtd._sync_fdtd()  # noqa: SLF001
        return func(fdtd, *args, **kwargs)

    return wrapped


class LumericalFDTD(ISolver):
    """Class interfacing with the lumapi FDTD class."""

    def __init__(self) -> None:
        """Initialize a LumericalFDTD."""
        self.fdtd: vipdopt.lumapi.FDTD | None = None  # type: ignore
        self._synced: bool = False
        self._env_vars: dict | None = None
        self.current_sim: LumericalSimulation | None = None

    @override
    def connect(self, hide: bool = True) -> None:
        if vipdopt.lumapi is None:
            raise ModuleNotFoundError(
                'Module "vipdopt.lumapi" has not yet been instatiated.'
            )
        while self.fdtd is None:
            try:
                self.fdtd = vipdopt.lumapi.FDTD(hide=hide)
                vipdopt.logger.info('Verified license with Lumerical servers.\n')
            except (AttributeError, vipdopt.lumapi.LumApiError) as e:  # noqa: PERF203
                vipdopt.logger.exception(
                    'Licensing server error - can be ignored.',
                    exc_info=e,
                )
                continue
        self.fdtd.newproject()

    @_check_lum_fdtd
    def _sync_fdtd(self):
        """Sync local environment variables with those of `vipdopt.lumapi.FDTD`."""
        if self._synced:
            return
        if self._env_vars is not None:
            self.setup_env_resources(**self._env_vars)
            self._env_vars = None
        self._synced = True
        vipdopt.logger.debug('Resynced LumericalFDTD.')

    @_check_lum_fdtd
    @override
    def addjob(self, fname: str):
        self.fdtd.addjob(fname, 'FDTD')  # type: ignore

    @_check_lum_fdtd
    @override
    def clearjobs(self):
        self.fdtd.clearjobs('FDTD')  # type: ignore

    @_sync_lum_fdtd_solver
    @override
    def runjobs(self, option: int = 1):
        """Run all simulations in the job queue.

        Arguments:
            option (int): Indicates the resources to use when runnin simulations.
                0: Run jobs in single process mode using only the local machine.
                1: Run jobs using the resources and parallel settings specified in
                    the resource manager. (default)
        """
        self.fdtd.runjobs('FDTD', option)  # type: ignore

    @_sync_lum_fdtd_solver
    @override
    def run(self):
        self.fdtd.run()

    @override
    def close(self):
        if self.fdtd is not None:
            vipdopt.logger.debug('Closing connection with Lumerical...')
            self.fdtd.close()
            vipdopt.logger.debug('Succesfully closed connection with Lumerical.')
            self.fdtd = None
            # self._synced = False

    @_check_lum_fdtd
    @override
    @ensure_path
    def load(self, path: Path):
        """Load a simulation from a Lumerical .fsp file."""
        fname = str(path)
        self.fdtd.load(fname)  # type: ignore
        vipdopt.logger.debug(f'Succesfully loaded simulation from {fname}.\n')

    @overload
    @ensure_path
    def save(self, path: Path): ...

    @overload
    @ensure_path
    def save(self, path: Path, sim: Simulation): ...

    @_check_lum_fdtd
    @ensure_path
    def save(self, path: Path, sim: Simulation | None = None):
        """Save a LumericalSimulation using the FDTD solver software."""
        if sim is not None:
            self.load_simulation(sim)
        if self.current_sim is not None:
            self.current_sim.info['path'] = path
        self.fdtd.save(str(path))  # type: ignore
        vipdopt.logger.debug(f'Succesfully saved simulation to {path}.\n')

    @_check_lum_fdtd
    @typing.no_type_check
    def get_env_resources(self) -> dict:
        """Return a dictionary containing all job manager resources."""
        resources = {}
        for resource in self.fdtd.setresource('FDTD', 1).splitlines():
            resources[resource] = self.fdtd.getresource('FDTD', 1, resource)
        return resources

    def promise_env_setup(self, **kwargs):
        """Setup the environment settings as soon as the fdtd is instantiated."""
        if self.fdtd is None:
            self._env_vars = kwargs if len(kwargs) > 0 else None
        else:
            self.setup_env_resources(**kwargs)

    @_check_lum_fdtd
    def set_resource(self, resource_num: int, resource: str, value: Any):
        """Set the specified job manager resource for this simulation."""
        self.fdtd.setresource('FDTD', resource_num, resource, value)  # type: ignore

    def setup_env_resources(self, **kwargs):
        """Configure the environment resources for running this simulation.

        **kwargs:
            mpi_exe (Path): Path to the mpi executable to use
            nprocs (int): The number of processes to run the simulation with
            hostfile (Path | None): Path to a hostfile for running distributed jobs
            solver_exe (Path): Path to the fdtd-solver to run
            nsims (int): The number of simulations being run
        """
        self.set_resource(1, 'mpi no default options', '1')

        nsims = kwargs.pop('nsims', 1)
        self.set_resource(1, 'capacity', nsims)

        mpi_exe = kwargs.pop(
            'mpi_exe',
            Path('/central/software/mpich/4.0.0/bin/mpirun'),
        )
        self.set_resource(1, 'mpi executable', str(mpi_exe))

        nprocs = kwargs.pop('nprocs', 8)
        hostfile = kwargs.pop('hostfile', None)
        mpi_opt = f'-n {nprocs}'
        if hostfile is not None:
            mpi_opt += f' --hostfile {hostfile}'
        self.set_resource(1, 'mpi extra command line options', mpi_opt)

        solver_exe = kwargs.pop(
            'solver_exe',
            Path('/central/home/tmcnicho/lumerical/v232/bin/fdtd-engine-mpich2nem'),
        )
        self.set_resource(1, 'solver executable', str(solver_exe))

        self.set_resource(
            1,
            'submission script',
            '#!/bin/sh\n'
            f'{mpi_exe} {mpi_opt} {solver_exe} {{PROJECT_FILE_PATH}}\n'
            'exit 0',
        )
        vipdopt.logger.debug('Updated simulation resources:')
        for resource, value in self.get_env_resources().items():
            vipdopt.logger.debug(f'\t"{resource}" = {value}')

    @_check_lum_fdtd
    @typing.no_type_check
    def getresult(
        self,
        object_name: str,
        property_name: str | None = None,
        dataset_value: str | None = None,
    ) -> Any:
        """Get a result from the FDTD solver, accesing a specific value if desired."""
        if property_name is None:
            res = self.fdtd.getresult(object_name)  # Gets all named results
            vipdopt.logger.debug(f'Available results from {object_name}:\n{res}')
            return res

        res = self.fdtd.getresult(object_name, property_name)
        if dataset_value is not None:
            res = res[dataset_value]
        vipdopt.logger.debug(f'Got "{property_name}" from "{object_name}": {res}')
        return res

    @_check_lum_fdtd
    @typing.no_type_check
    def get_shape(
        self,
        monitor_name: str,
        value: str,
        dataset_value: str | None = None,
    ) -> tuple[int, ...]:
        """Return the shape of a property returned from this simulation's monitors."""
        prop = self.getresult(monitor_name, value, dataset_value)
        if isinstance(prop, dict):
            raise ValueError(  # noqa: TRY004
                'Requested property is a dataset, and does not have a '
                '`shape` attribute.'
                f' Args passed: {monitor_name}, {value}, {dataset_value}.'
            )
        return np.squeeze(prop).shape

    @_check_lum_fdtd
    @typing.no_type_check
    def get_transmission_magnitude(self, monitor_name: str) -> npt.NDArray:
        """Return the magnitude of the transmission for a given monitor."""
        # Ensure this is a transmission monitor
        monitor_name = monitor_name.replace('focal', 'transmission')
        transmission = self.getresult(monitor_name, 'T', 'T')
        return np.abs(transmission)

    @_check_lum_fdtd
    @typing.no_type_check
    def get_transmission_shape(self, monitor_name: str) -> npt.NDArray:
        """Get the shape of the transmission field from the given monitor."""
        return self.get_shape(monitor_name, 'T', 'T')

    @_check_lum_fdtd
    @typing.no_type_check
    def get_field(self, monitor_name: str, field_indicator: str) -> npt.NDArray:
        """Return the E or H field from a monitor."""
        if field_indicator not in 'EHP':
            raise ValueError(
                f'Expected field_indicator to be "E", "H" or "P"; got {field_indicator}'
            )
        polarizations = [field_indicator + c for c in 'xyz']

        start = time.time()
        vipdopt.logger.debug(f'Getting {polarizations} from monitor "{monitor_name}"')
        fields = np.array(
            list(map(partial(self.fdtd.getdata, monitor_name), polarizations)),
            dtype=np.complex128,
        )
        data_xfer_size_mb = fields.nbytes / (1024**2)
        elapsed = time.time() - start + 1e-8  # avoid zero error

        vipdopt.logger.debug(f'Transferred {data_xfer_size_mb} MB')
        vipdopt.logger.debug(f'Data rate = {data_xfer_size_mb / elapsed} MB/sec')

        return fields

    def get_hfield(self, monitor_name: str) -> npt.NDArray:
        """Return the H field from a monitor."""
        return self.get_field(monitor_name, 'H')

    def get_efield(self, monitor_name: str) -> npt.NDArray:
        """Return the E field from a monitor."""
        return self.get_field(monitor_name, 'E')

    @_check_lum_fdtd
    def get_poynting(self, monitor_name: str) -> npt.NDArray:
        """Return the Poynting vector from a monitor."""
        return self.get_field(monitor_name, 'P')

    @_check_lum_fdtd
    def get_transmission(self, monitor_name: str) -> npt.NDArray:
        """Return the transmission as a function of wavelength."""
        return self.fdtd.transmission(monitor_name)  # type: ignore

    def get_efield_magnitude(self, monitor_name: str) -> npt.NDArray:
        """Return the magnitude of the E field from a monitor."""
        enorm_squared = self.fdtd.getelectric(monitor_name)  # type: ignore
        return np.sqrt(enorm_squared)

    @_check_lum_fdtd
    @typing.no_type_check
    def get_source_power(self, monitor_name: str) -> npt.NDArray:
        """Return the source power of a given monitor."""
        f = self.getresult(monitor_name, 'f')  # Get frequency vector
        return self.fdtd.sourcepower(f)

    def get_overall_power(self, monitor_name) -> npt.NDArray:
        """Return the overall power from a given monitor."""
        sp = self.get_source_power(monitor_name)
        t = self.get_transmission_magnitude(monitor_name)
        return t.T * sp

    @_check_lum_fdtd
    def load_simulation(self, sim: Simulation):
        """Load a simulation into the FDTD solver."""
        if not isinstance(sim, LumericalSimulation):
            raise TypeError(
                'LumericalFDTD can only load simulations of type "LumericalSimulation"'
                f'; Received "{type(sim)}"'
            )
        self.fdtd.switchtolayout()  # type: ignore
        self.fdtd.deleteall()  # type: ignore
        for obj in sim.objects.values():
            # Create an object for each of those in the simulation
            with contextlib.suppress(BaseException):
                LumericalSimObjectType.get_add_function(obj.obj_type)(
                    self.fdtd,
                    **obj.properties,
                )
        self.current_sim = sim

    @_check_lum_fdtd
    def reformat_monitor_data(self, sims: list[LumericalSimulation]):
        """Reformat simulation data so it can be loaded independent of the solver.

        This method does the following for each provided simulation.
            * Loads the simulation from the path indicated by sim.info['path']
            * Creates a .npz file for each monitor in the simulation, containing all
                of the returned values (E, H, P, T, Source Power)

        Arguments:
            sims (list[LumericalSimulation]): The simulation to load data from. Must
                have the `info['path']` field populated.
        """
        for sim in sims:
            sim_path: Path = sim.info['path']
            self.load(sim_path)
            for monitor in sim.monitors():
                mname = monitor.name
                e = self.get_efield(mname)
                h = self.get_hfield(mname)
                p = self.get_poynting(mname)
                t = self.get_transmission(mname)
                sp = self.get_source_power(mname)

                output_path = sim_path.with_suffix('') / f'_{mname}.npz'
                monitor.set_src(output_path)

                np.savez(output_path, e=e, h=h, p=p, t=t, sp=sp)


ISolver.register(LumericalFDTD)
