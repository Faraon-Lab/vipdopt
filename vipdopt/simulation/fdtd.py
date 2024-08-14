"""Code for handling the connection to the FDTD solver."""

from __future__ import annotations

import abc
import contextlib
import functools
import os
import time
import typing
from collections.abc import Callable
from functools import partial
from typing import Any, Concatenate, overload

import numpy as np
import numpy.typing as npt

import vipdopt
from vipdopt.simulation.simobject import Import, LumericalSimObjectType
from vipdopt.simulation.simulation import ISimulation, LumericalSimulation
from vipdopt.utils import (
    P,
    Path,
    PathLike,
    R,
    convert_path,
    ensure_path,
    import_lumapi,
    setup_logger,
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
    @overload
    @ensure_path
    def load(self, path: Path): ...

    @abc.abstractmethod
    @overload
    def load(self, sim: ISimulation): ...

    @abc.abstractmethod
    @overload
    @ensure_path
    def load(self, path: Path, sim: ISimulation): ...

    @abc.abstractmethod
    def load(self, path: Path | None, sim: ISimulation | None):
        """Load data into the solver from a file into a simulation object.."""

    @abc.abstractmethod
    @overload
    @ensure_path
    def save(self, path: Path): ...

    @abc.abstractmethod
    @overload
    @ensure_path
    def save(self, path: Path, sim: ISimulation): ...

    @abc.abstractmethod
    @ensure_path
    def save(self, path: Path, sim: ISimulation | None = None):
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

    # @override
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
    @ensure_path
    # @override
    def addjob(self, fname: Path):
        self.fdtd.addjob(str(fname.absolute()), 'FDTD')  # type: ignore

    @_check_lum_fdtd
    # @override
    def clearjobs(self):
        self.fdtd.clearjobs('FDTD')  # type: ignore

    @_sync_lum_fdtd_solver
    # @override
    def runjobs(self, option: int = 1):
        """Run all simulations in the job queue.

        Arguments:
            option (int): Indicates the resources to use when running simulations.
                0: Run jobs in single process mode using only the local machine.
                1: Run jobs using the resources and parallel settings specified in
                    the resource manager. (default)
        """
        vipdopt.logger.info(f'Running simulations: {self.fdtd.listjobs("FDTD")}')
        self.fdtd.runjobs('FDTD', option)  # type: ignore
        self.current_sim = None
        vipdopt.logger.info('Finished running job queue')

    @_sync_lum_fdtd_solver
    # @override
    def run(self):
        self.fdtd.run()

    # @override
    def close(self):
        if self.fdtd is not None:
            vipdopt.logger.debug('Closing connection with Lumerical...')
            self.fdtd.close()
            vipdopt.logger.debug('Successfully closed connection with Lumerical.')
            self.fdtd = None
            # self._synced = False

    @overload
    @ensure_path
    def load(self, path: Path): ...

    @overload
    def load(self, sim: ISimulation): ...

    @overload
    @ensure_path
    def load(self, path: Path, sim: ISimulation): ...

    @_check_lum_fdtd
    def load(self, path: PathLike | None, sim: ISimulation | None):
        """Load data into the solver from a file or a simulation object.."""
        if path is not None:
            path = convert_path(path)
        if sim is not None:
            self.fdtd.switchtolayout()  # type: ignore
            self.fdtd.deleteall()  # type: ignore
            if path is not None:  # Load file into sim object
                path = path.absolute()
                self.fdtd.load(str(path))
                sim.set_path(path)
            elif sim.get_path() is not None:  # Load data from sim's path
                self.fdtd.load(str(sim.get_path()))
            else:
                self.load_simulation(sim)
            self.current_sim = sim
        elif path is not None:  # Just load data from file
            self.fdtd.switchtolayout()  # type: ignore
            self.fdtd.deleteall()  # type: ignore
            path = path.absolute()
            self.fdtd.load(str(path))
            if self.current_sim is not None:
                self.current_sim.set_path(path)
        else:
            raise ValueError('Both arguments `path` and `sim` cannot be `None`.')

    @_check_lum_fdtd
    def load_simulation(self, sim: ISimulation):
        """Load a simulation into the FDTD solver."""
        if not isinstance(sim, LumericalSimulation):
            raise TypeError(
                'LumericalFDTD can only load simulations of type "LumericalSimulation"'
                f'; Received "{type(sim)}"'
            )
        vipdopt.logger.debug(
            f'Loading LumericalSimulation "{sim.info["name"]}" into LumericalFDTD...'
        )
        self.fdtd.switchtolayout()  # type: ignore
        self.fdtd.deleteall()  # type: ignore
        for obj in sim.objects.values():
            # Create an object for each of those in the simulation
            with contextlib.suppress(BaseException):
                # LumericalSimObjectType.get_add_function(obj.obj_type)(
                #     self.fdtd,
                #     **obj.properties,
                # )     
                # Some property in the obj.properties dictionary is inactive, which breaks everything after it.
                # So instead we must add just the object, then set each property individually.
                LumericalSimObjectType.get_add_function(obj.obj_type)(
                    self.fdtd, {'name': obj.name}
                )
                for p, r in obj.properties.items():
                    try:
                        self.fdtd.setnamed(obj.name, p, r)
                    except Exception as err:
                        # vipdopt.logger.debug(err + f": {p}")
                        pass
                    
                # Import nk2 if possible
                if isinstance(obj, Import) and obj.n is not None:
                    self.importnk2(obj.name, *obj.get_nk2())
        self.current_sim = sim

    # @_check_lum_fdtd
    # # @override
    # @ensure_path
    # def load(self, path: Path):
    #     """Load a simulation from a Lumerical .fsp file."""
    #     vipdopt.logger.debug(f'Loading simulation from {path!s} into Lumerical...')
    #     fname = str(path)
    #     self.fdtd.load(fname)  # type: ignore
    #     vipdopt.logger.debug(f'Successfully loaded simulation from {fname}.\n')

    @overload
    @ensure_path
    def save(self, path: Path): ...

    @overload
    @ensure_path
    def save(self, path: Path, sim: ISimulation): ...

    @_check_lum_fdtd
    @ensure_path
    def save(self, path: Path, sim: ISimulation | None = None):
        """Save a LumericalSimulation using the FDTD solver software."""
        path = path.absolute()
        if sim is not None:
            self.load_simulation(sim)

        if self.current_sim is not None:
            self.current_sim.set_path(path)
            # self.current_sim.info['path'] = path
        
        # Sometimes saving fails due to some sort of perms issue depending on machine and OS.
        # Attempt saving with 2s between attempts, and hard fails after 4 attempts.
        for x in range(0, 4):  # try 4 times
            str_error = None
            try:
                self.fdtd.save(str(path))  # type: ignore
                str_error = None
            except Exception as e:
                str_error = True
                
            if str_error:
                time.sleep(2)  # wait for 2 seconds before trying to fetch the data again
            else:
                break
        vipdopt.logger.debug(f'Successfully saved simulation to {path}.\n')

    @_check_lum_fdtd
    @typing.no_type_check
    def get_env_resources(self, resource_num: int = 1) -> dict:
        """Return a dictionary containing all job manager resources."""
        resources = {}
        for resource in self.fdtd.getresource('FDTD', resource_num).splitlines():
            resources[resource] = self.fdtd.getresource('FDTD', resource_num, resource)
        return resources

    def promise_env_setup(self, **kwargs):
        """Setup the environment settings as soon as the fdtd is instantiated."""
        if self.fdtd is None:
            self._env_vars = kwargs if len(kwargs) > 0 else None
        else:
            if self._env_vars is None:
                self._env_vars = kwargs
            self.setup_env_resources(**kwargs)

    def get_env_vars(self) -> dict:
        """Return the current pending environment variables to be set.

        Returns:
            dict: The current pending environment variables to be set. If
                `self._env_vars` is None, returns an empty dictionary.
        """
        return {} if self._env_vars is None else self._env_vars

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
        if not os.getenv('SLURM_JOB_NODELIST') is None:     
            # 20240729 Ian - Just for automatic switching between SLURM HPC and otherwise
            mpi_exe = "/central/home/ifoo/lumerical/2021a_r22/mpich2/nemesis/bin/mpiexec"
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
        if not os.getenv('SLURM_JOB_NODELIST') is None:     
            # 20240729 Ian - Just for automatic switching between SLURM HPC and otherwise
            solver_exe = "/central/home/ifoo/lumerical/2021a_r22/bin/fdtd-engine-mpich2nem"
        self.set_resource(1, 'solver executable', str(solver_exe))

        self.set_resource(
            1,
            'submission script',
            '#!/bin/sh\n'
            f'{mpi_exe} {mpi_opt} {solver_exe} {{PROJECT_FILE_PATH}}\n'
            'exit 0',
        )
        vipdopt.logger.debug('Updated lumerical job manager resources:')
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
        vipdopt.logger.debug(f'Got "{property_name}" from "{object_name}"')
        return res

    @_check_lum_fdtd
    @typing.no_type_check
    def get_field(self, monitor_name: str, field_indicator: str) -> npt.NDArray:
        """Return the E or H field or Poynting vector (P) from a monitor."""
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
    def transmission(self, monitor_name: str) -> npt.NDArray:
        """Return the transmission as a function of wavelength."""
        return np.squeeze(self.fdtd.transmission(monitor_name))  # type: ignore

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
        t = np.abs(self.transmission(monitor_name))
        return t.T * sp

    @_check_lum_fdtd
    def reformat_monitor_data(self, sims: list[LumericalSimulation]):
        """Reformat simulation data so it can be loaded independent of the solver.

        This method does the following for each provided simulation.
            * Loads the simulation from the path indicated by sim.info['path']
            * Creates a .npz file for each monitor in the simulation, containing all
                of the returned values (E, H, P, T, Source Power)

        Arguments:
            sims (list[LumericalSimulation]): The simulations to load data from. Must
                have the `info['path']` field populated.
        """
        vipdopt.logger.info('Reformatting monitor data...')
        for sim in sims:
            self.fdtd.switchtolayout()
            sim_path: Path | None = sim.get_path()
            if sim_path is None:
                continue
            self.load(sim_path, None)
            vipdopt.logger.debug(f'Reformatting monitor data from {sim_path}...')
            sim.link_monitors()
            for monitor in sim.monitors():
                mname = monitor.name
                # vipdopt.logger.debug(mname)
                # vipdopt.logger.debug(self.fdtd.getdata(mname))
                data = self.fdtd.getdata(mname).split()
                # vipdopt.logger.debug(data)
                e = self.get_efield(mname) if 'Ex' in data else None
                h = self.get_hfield(mname) if 'Hx' in data else None
                p = self.get_poynting(mname) if 'Px' in data else None
                # if monitor['monitor type'] == '2D Z-normal':
                #     t = self.get_transmission(mname)
                # else:
                #     t = None
                try:
                    t = self.transmission(mname)
                except vipdopt.lumapi.LumApiError:
                    t = None
                sp = self.get_source_power(mname)
                power = self.fdtd.getdata(mname, 'power') if 'power' in data else None

                with monitor.src.open('wb') as f:
                    np.savez(f, e=e, h=h, p=p, t=t, sp=sp, power=power)
                monitor.reset()

                # vipdopt.logger.debug(f'E field: {monitor.e}')
                # monitor.reset()
                # vipdopt.logger.debug(f'E field after (should be None): {monitor._e}')
                # vipdopt.logger.debug('Should print loading again')
                # monitor.e
                # vipdopt.logger.debug('Should NOT print loading again')
                # monitor.h
                # return
        vipdopt.logger.info('Finished reformatting monitor data.')

    @_check_lum_fdtd
    def importnk2(
        self,
        import_name: str,
        n: npt.NDArray,
        x: npt.NDArray,
        y: npt.NDArray,
        z: npt.NDArray,
    ):
        """Import the refractive index (n and k) over an entire volume / surface.

        Arguments:
            import_name (str): Name of the import primitive to import to.
            n (npt.NDArray): Refractive index. Must be of dimension NxMxP or NxMxPx3,
                depending on whether the material isotropic or not, with N, M, P >= 2.
            x (npt.NDArray): If n is NxMxP, then x should be Nx1. Values must be
                uniformly spaced.
            y (npt.NDArray): If n is NxMxP, then y should be Mx1. Values must be
                uniformly spaced.
            z (npt.NDArray): If n is NxMxP, then z should be Px1. Values must be
                uniformly spaced.
        """
        self.fdtd.select(import_name)
        self.fdtd.importnk2(n, x, y, z)
    
    @_check_lum_fdtd
    def exportnk2(
        self,
        indexmonitor_name: str,
        component: str='x'
    ) -> npt.NDArray:
        """Return the index values returned from this simulation's design index monitors."""
        index_prev = self.fdtd.getresult(indexmonitor_name, 'index preview')
        return index_prev[f'index_{component}']     # might need np.squeeze()


ISolver.register(LumericalFDTD)

if __name__ == '__main__':
    vipdopt.logger = setup_logger('logger', 0)
    vipdopt.lumapi = import_lumapi(
        'C:\\Program Files\\Lumerical\\v221\\api\\python\\lumapi.py'
    )
    # fdtd = LumericalFDTD()
    # sim = LumericalSimulation('test_data\\sim.json')
    # # sim.info['path'] = Path('testing\\monitor_data\\sim.fsp')
    # fdtd.connect(hide=True)
    # # fdtd.load_simulation(sim)
    # fdtd.save('testing\\monitor_data\\sim.fsp', sim)
    # fdtd.addjob('testing\\monitor_data\\sim.fsp')
    # fdtd.runjobs(0)
    # # fdtd.run()
    # # vipdopt.logger.debug(sim.info)
    # # print(sim.info)
    # fdtd.reformat_monitor_data([sim])
    # fdtd.close()

    # Creating the FDTD hook

    fdtd = LumericalFDTD()
    sim = LumericalSimulation('docs\\notebooks\\simulation_example.json')
    sim_file = 'sim.fsp'  # Where Lumerical will save simulation data

    fdtd.connect(hide=False)  # This starts a Lumerical session

    fdtd.load(path=None, sim=sim)  # Load simulation into Lumerical
    fdtd.save(sim_file)  # Lumerical must save to a file before running

    fdtd.addjob(sim_file)  # Add the simulation file to the job queue

    fdtd.runjobs()

    # Create individual data files for each Monitor
    fdtd.reformat_monitor_data([sim])

    fdtd.close()  # End Lumerical session
