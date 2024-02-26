"""FDTD Simulation Interface Code."""
from __future__ import annotations

import abc
import functools
import json
import logging
import time
from argparse import ArgumentParser
from collections import OrderedDict
from collections.abc import Callable
from copy import copy
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Any, Concatenate

import numpy as np
import numpy.typing as npt
from overrides import override

import vipdopt
from vipdopt.utils import P, PathLike, R, ensure_path, read_config_file


def _check_fdtd(
        func: Callable[Concatenate[LumericalSimulation, P], R],
)-> Callable[Concatenate[LumericalSimulation, P], R]:
    @functools.wraps(func)
    def wrapped(sim: LumericalSimulation, *args: P.args, **kwargs: P.kwargs):
        if sim.fdtd is None:
            raise UnboundLocalError(
                f'Cannot call {func.__name__} before `fdtd` is instantiated.'
                ' Has `connect()` been called?'
            )
        return func(sim, *args, **kwargs)
    return wrapped


def _sync_fdtd_solver(
        func: Callable[Concatenate[LumericalSimulation, P], R],
)-> Callable[Concatenate[LumericalSimulation, P], R]:
    @functools.wraps(func)
    def wrapped(sim: LumericalSimulation, *args: P.args, **kwargs: P.kwargs):
        sim._sync_fdtd()  # noqa: SLF001
        return func(sim, *args, **kwargs)
    return wrapped

class ISimulation(abc.ABC):
    """Abstract class for an FDTD simulation."""

    @abc.abstractmethod
    def __init__(self, source: PathLike | dict | None) -> None:
        """Create simulation object."""

    @abc.abstractmethod
    def connect(self):
        """Create hook to simulation software."""

    @abc.abstractmethod
    def load(self, source: PathLike | dict):
        """Load simulation from a file or dictionary."""

    @abc.abstractmethod
    def save(self, fname: PathLike):
        """Save simulation data to a file."""

    @abc.abstractmethod
    def run(self):
        """Run the simulation."""




class LumericalSimObjectType(StrEnum):
    """Types of simulation objects in Lumerical."""
    FDTD = 'fdtd'
    MESH = 'mesh'
    GAUSSIAN = 'gaussian'
    TFSF = 'tfsf'
    DIPOLE = 'dipole'
    POWER = 'power'
    PROFILE = 'profile'
    INDEX = 'index'
    IMPORT = 'import'
    RECT = 'rect'

    def get_add_function(self) -> Callable:
        """Get the corerct lumapi function to add an object."""
        return getattr(vipdopt.lumapi.FDTD, f'add{self.value}')

SOURCE_TYPES = [
    LumericalSimObjectType.DIPOLE,
    LumericalSimObjectType.TFSF,
    LumericalSimObjectType.GAUSSIAN,
]

MONITOR_TYPES = [
    LumericalSimObjectType.POWER,
    LumericalSimObjectType.PROFILE,
    LumericalSimObjectType.INDEX,
]

OBJECT_TYPE_NAME_MAP = {
    'FDTD': LumericalSimObjectType.FDTD,
    'GaussianSource': LumericalSimObjectType.GAUSSIAN,
    'DipoleSource': LumericalSimObjectType.DIPOLE,
    'Rectangle': LumericalSimObjectType.RECT,
    'Mesh': LumericalSimObjectType.MESH,
    'Import': LumericalSimObjectType.IMPORT,
    'IndexMonitor': LumericalSimObjectType.INDEX,
}

class LumericalEncoder(json.JSONEncoder):
    """Encodes LumericalSim objects in JSON format."""
    @override
    def default(self, o: Any) -> Any:
        if isinstance(o, LumericalSimObjectType):
            return {'obj_type': str(o)}
        if isinstance(o, LumericalSimObject):
            return copy(vars(o))
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


class LumericalSimObject:
    """Lumerical Simulation Object.

    Attributes:
        name (str): name of the object
        obj_type (LumericalSimObjectType): the type of object
        properties (OrderedDict[str, Any]): Map of named properties and their values
    """
    def __init__(self, name: str, obj_type: LumericalSimObjectType) -> None:
        """Crearte a LumericalSimObject."""
        self.name = name
        self.obj_type = obj_type
        self.properties: OrderedDict[str, Any] = OrderedDict()
        if obj_type != LumericalSimObjectType.FDTD:
            self.properties['name'] = name

    def __str__(self) -> str:
        """Return string representation of object."""
        return json.dumps(
            vars(self),
            indent=4,
            ensure_ascii=True,
        )

    def __setitem__(self, key: str, val: Any) -> None:
        """Set the value of a property of the object."""
        self.properties[key] = val

    def __getitem__(self, key: str) -> Any:
        """Retrieve a property from an object."""
        return self.properties[key]

    def update(self, **vals):
        """Update properties with values in a dictionary."""
        self.properties.update(vals)

    def __eq__(self, __value: object) -> bool:
        """Test equality of LumericalSimObjects."""
        if isinstance(__value, LumericalSimObject):
            return self.obj_type == __value.obj_type and \
                self.properties == __value.properties
        return super().__eq__(__value)



class LumericalSimulation(ISimulation):
    """Lumerical FDTD Simulation Code.

    Attributes:
        fdtd (lumapi.FDTD): Lumerical session
        objects (OrderedDict[str, LumericalSimObject]): The objects within
            the simulation
    """

    def __init__(self, source: PathLike | dict | None=None) -> None:
        """Create a LumericalSimulation.

        Arguments:
            source (PathLike | dict | None): optional source to load the simulation from
        """
        self.fdtd: vipdopt.lumapi.FDTD | None = None
        self._clear_objects()
        self._synced = False  # Flag for whether fdtd needs to be synced
        # Parameters to pass into set_env_vars in the future
        self._env_vars: dict | None = None

        if source:
            self.load(source)

    @override
    def __str__(self) -> str:
        return self.as_json()

    def get_env_vars(self) -> dict:
        """Return the current pending environment variables to be set.

        Returns:
            dict: The current pending environment variables to be set. If
                `self._env_vars` is None, returns an empty dictionary.
        """
        return {} if self._env_vars is None else self._env_vars

    @override
    def connect(self) -> None:
        vipdopt.logger.debug('Attempting to connect to Lumerical servers...')

        # This loop handles the problem of
        # "Failed to start messaging, check licenses..." that Lumerical often has.
        while self.fdtd is None:
            try:
                self.fdtd = vipdopt.lumapi.FDTD(hide=True)
            except (AttributeError, vipdopt.lumapi.LumApiError) as e:
                vipdopt.logger.exception(
                    'Licensing server error - can be ignored.',
                    exc_info=e,
                )
                continue

        vipdopt.logger.debug('Verified license with Lumerical servers.\n')
        self.fdtd.newproject()
        self._sync_fdtd()

    @override
    def load(self, source: PathLike | dict):
        if isinstance(source, dict):
            self._load_dict(source)
        elif source.suffix == '.fsp':
            self._load_fsp(source)
        else:
            self._load_file(source)
    
    @_check_fdtd
    def _load_fsp(self, fname: PathLike):
        """Load a simulation from a Lumerical .fsp file."""
        vipdopt.logger.debug(f'Loading simulation from {fname}...')
        self._clear_objects()
        self.fdtd.load(str(fname))
        self.fdtd.selectall()
        objects = self.fdtd.getAllSelectedObjects()
        # vipdopt.logger.debug(list(vars(objects[0]).keys()))
        # vipdopt.logger.debug(list(objects[0].__dict__.keys()))
        # print(objects[0]['type'])
        for o in objects:
            otype = o['type']
            if otype == 'DFTMonitor':
                if o['spatial interpolation'] == 'specified position':
                    obj_type = LumericalSimObjectType.PROFILE
                else:
                    obj_type = LumericalSimObjectType.POWER
            else:
                obj_type = OBJECT_TYPE_NAME_MAP[otype]
            oname = o._id.name.split('::')[-1]
            sim_obj = LumericalSimObject(oname, obj_type)
            for name in o._nameMap:
                sim_obj[name] = o[name]
            
            self.objects[oname] = sim_obj
        vipdopt.logger.debug(self.as_json())


    def _load_file(self, fname: PathLike):
        """Load a simulation from a JSON file."""
        vipdopt.logger.debug(f'Loading simulation from {fname}...')
        sim = read_config_file(fname)
        self._load_dict(sim)
        vipdopt.logger.info(f'Succesfully loaded {fname}\n')

    def _load_dict(self, d: dict):
        """Load a simulation from a dictionary of objects."""
        self._clear_objects()
        for obj in d['objects'].values():
            self.new_object(
                obj['name'],
                LumericalSimObjectType(obj['obj_type']),
                **obj['properties'],
            )

    @_check_fdtd
    def _sync_fdtd(self):
        """Sync local objects with `lumapi.FDTD` objects."""
        if self._synced:
            return
        self.fdtd.switchtolayout()
        for obj_name, obj in self.objects.items():
            # Create object if needed
            if self.fdtd.getnamednumber(obj_name) == 0:
                LumericalSimObjectType.get_add_function(obj.obj_type)(
                    self.fdtd,
                    **obj.properties,
                )
            for key, val in obj.properties.items():
                self.fdtd.setnamed(obj_name, key, val)
        if self._env_vars is not None:
            self.setup_env_resources(**self._env_vars)
            self._env_vars = None
        self._synced = True
        vipdopt.logger.debug('Resynced LumericalSimulation with fdtd solver')


    @ensure_path
    @override
    def save(self, fname: Path):
        if self.fdtd is not None:
            self.save_lumapi(fname)
        content = self.as_json()
        with open(fname.with_suffix('.json'), 'w', encoding='utf-8') as f:
            f.write(content)
        vipdopt.logger.debug(f'Succesfully saved simulation to {fname}.\n')

    @_sync_fdtd_solver
    @ensure_path
    def save_lumapi(self, fname: Path):
        """Save using Lumerical's simulation file type (.fsp)."""
        self.fdtd.save(str(fname.with_suffix('')))

    def as_dict(self) -> dict:
        """Return a dictionary representation of this simulation."""
        return {'objects': {name: vars(obj) for name, obj in self.objects.items()}}

    def as_json(self) -> str:
        """Return a JSON representation of this simulation."""
        return json.dumps(
            self.as_dict(),
            indent=4,
            ensure_ascii=True,
            cls=LumericalEncoder,
        )

    @_sync_fdtd_solver
    @override
    def run(self):
        self.fdtd.run()

    @_check_fdtd
    def set_resource(self, resource_num: int, resource: str, value: Any):
        """Set the specified job manager resource for this simulation."""
        self.fdtd.setresource('FDTD', resource_num, resource, value)

    def _clear_objects(self):
        """Clear all existing objects and create a new project."""
        self.objects = OrderedDict()
        if self.fdtd is not None:
            self.fdtd.newproject()
            self._synced = True

    def copy(self) -> LumericalSimulation:
        """Return a copy of this simulation."""
        new_sim = LumericalSimulation()
        for obj_name, obj in self.objects.items():
            new_sim.new_object(obj_name, obj.obj_type, **obj.properties)

        new_sim.promise_env_setup(**self.get_env_vars())

        return new_sim

    def enable(self, objs: list[str]):
        """Enable all objects in provided list."""
        for obj in objs:
            self.update_object(obj, enabled=1)

    def with_enabled(self, objs: list[str]) -> LumericalSimulation:
        """Return copy of this simulation with only objects in objs enabled."""
        new_sim = self.copy()
        new_sim.disable_all_sources()
        new_sim.enable(objs)
        return new_sim

    def disable(self, objs: list[str]):
        """Disable all objects in provided list."""
        for obj in objs:
            self.update_object(obj, enabled=0)

    def with_disabled(self, objs: list[str]) -> LumericalSimulation:
        """Return copy of this simulation with only objects in objs disabled."""
        new_sim = self.copy()
        new_sim.enable_all_sources()
        new_sim.disable(objs)
        return new_sim

    def enable_all_sources(self):
        """Enable all sources in this simulation."""
        self.enable(self.source_names())

    def disable_all_sources(self):
        """Disable all sources in this simulation."""
        self.disable(self.source_names())

    def sources(self) -> list[LumericalSimObject]:
        """Return a list of all source objects."""
        return [obj for _, obj in self.objects.items() if obj.obj_type in SOURCE_TYPES]

    def source_names(self) -> list[str]:
        """Return a list of all source object names."""
        return [obj.name for obj in self.sources()]

    def monitors(self) -> list[LumericalSimObject]:
        """Return a list of all monitor objects."""
        return [obj for _, obj in self.objects.items() if obj.obj_type in MONITOR_TYPES]

    def monitor_names(self) -> list[str]:
        """Return a list of all montor object names."""
        return [obj.name for obj in self.monitors()]

    def new_object(
        self,
        obj_name: str,
        obj_type: LumericalSimObjectType,
        **properties,
    ) -> LumericalSimObject:
        """Create a new object and add it to the simulation.

        Arguments:
            obj_name (str): Name of the object
            obj_type (LumericalSimObjectType): type of the object
            properties (dict[str, Any]): optional dictionary to populate
                the new object with
        """
        vipdopt.logger.debug(f"Creating new object: '{obj_name}'...")
        obj = LumericalSimObject(obj_name, obj_type)
        obj.update(**properties)
        self.add_object(obj)
        return obj

    def add_object(self, obj: LumericalSimObject) -> None:
        """Add an existing object to the simulation."""
        # Add copy to the lumapi.FDTD
        if self.fdtd is not None:
            LumericalSimObjectType.get_add_function(obj.obj_type)(
                self.fdtd,
                **obj.properties,
            )
        else:
            self._synced = False
        self.objects[obj.name] = obj

    def update_object(self, name: str, **properties):
        """Update lumerical object with new property values."""
        obj = self.objects[name]
        if self.fdtd is not None:
            for key, val in properties.items():
                self.fdtd.setnamed(obj.name, key, val)
        else:
            self._synced = False
        obj.update(**properties)

    def close(self):
        """Close fdtd conenction."""
        if self.fdtd is not None:
            vipdopt.logger.debug('Closing connection with Lumerical...')
            self.fdtd.close()
            vipdopt.logger.debug('Succesfully closed connection with Lumerical.')
            self.fdtd = None
            self._synced = False

    def __eq__(self, __value: object) -> bool:
        """Test equality of simulations."""
        if isinstance(__value, LumericalSimulation):
            return self.objects == __value.objects
        return super().__eq__(__value)

    def __enter__(self):
        """Startup for context manager usage."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup after exiting a context."""
        self.close()

    @_check_fdtd
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

    @_check_fdtd
    def getresult(
        self,
        object_name: str,
        property_name: str | None=None,
        dataset_value: str | None=None,
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

    def get_shape(
            self,
            monitor_name: str,
            value: str,
            dataset_value: str | None=None,
    ) -> tuple[int, ...]:
        """Return the shape of a property returned from this simulation's monitors."""
        prop = self.getresult(monitor_name, value, dataset_value)
        return np.squeeze(prop).shape

    def get_field_shape(self) -> tuple[int, ...]:
        """Return the shape of the fields returned from this simulation's monitors."""
        return self.get_shape('design_index_monitor', 'index', 'x')
        for s in self.fdtd.getresult('design_index_monitor').splitlines():
            vipdopt.logger.debug(s)
            vipdopt.logger.debug(self.fdtd.getresult('design_index_monitor', s))
        if self.fdtd.haveresult('design_index_monitor', 'index preview'):
            index_prev = self.fdtd.getresult('design_index_monitor', 'index preview')
            return np.squeeze(index_prev['index_x']).shape
        index = self.fdtd.getresult('design_index_monitor', 'index_x')
            
        return np.squeeze(index).shape
    
    def get_transimission_shape(self, monitor_name: str) -> npt.NDArray:
        """Return the magnitude of the transmission for a given monitor."""
        # Ensure this is a transmission monitor
        monitor_name = monitor_name.replace('focal', 'transmission')
        return self.get_shape(monitor_name, 'T', 'T')

    def get_transimission_magnitude(self, monitor_name: str) -> npt.NDArray:
        """Return the magnitude of the transmission for a given monitor."""
        # Ensure this is a transmission monitor
        monitor_name = monitor_name.replace('focal', 'transmission')
        transmission = self.getresult(monitor_name, 'T', 'T')
        return np.abs(transmission)

    @_check_fdtd
    def get_field(self, monitor_name: str, field_indicator: str) -> npt.NDArray:
        """Return the E or H field from a monitor."""
        if field_indicator not in 'EH':
            raise ValueError('Expected field_indicator to be "E" or "H";'
                             f' got {field_indicator}')
        polarizations = [field_indicator + c for c in 'xyz']

        start = time.time()
        vipdopt.logger.debug(f'Getting {polarizations} from monitor "{monitor_name}"')
        fields = np.array(
            list(map(partial(self.fdtd.getdata, monitor_name), polarizations)),
            dtype=np.complex128,
        )
        data_xfer_size_mb = fields.nbytes / (1024 ** 2)
        elapsed = time.time() - start

        vipdopt.logger.debug(f'Transferred {data_xfer_size_mb} MB')
        vipdopt.logger.debug(f'Data rate = {data_xfer_size_mb / elapsed} MB/sec')
        return fields.squeeze()

    def get_hfield(self, monitor_name: str) -> npt.NDArray:
        """Return the H field from a monitor."""
        return self.get_field(monitor_name, 'H')

    def get_efield(self, monitor_name: str) -> npt.NDArray:
        """Return the E field from a monitor."""
        return self.get_field(monitor_name, 'E')

    @_check_fdtd
    def get_pfield(self, monitor_name: str) -> npt.NDArray:
        """Returns power as a function of space and wavelength, with xyz components."""
        return self.getresult(monitor_name, 'P', 'P')

    def get_efield_magnitude(self, monitor_name: str) -> npt.NDArray:
        """Return the magnitude of the E field from a monitor."""
        efield = self.get_efield(monitor_name)
        enorm_squared = np.sum(np.square(np.abs(efield)), axis=0)
        return np.sqrt(enorm_squared)

    @_check_fdtd
    def get_source_power(self) -> npt.NDArray:
        """Return the source power of a given monitor."""
        f = self.getresult('focal_monitor_0', 'f')
        return self.fdtd.sourcepower(f)

    def get_overall_power(self, monitor_name) -> npt.NDArray:
        """Return the overall power from a given monitor."""
        source_power = self.get_source_power()
        t = self.get_transimission_magnitude(monitor_name)
        return source_power * t.T

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        'simulation_json',
        type=str,
        help='Simulation JSON to create a simulation from',
    )
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Enable verbose output.')

    args = parser.parse_args()

    # Set verbosity
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level)

    # with LumericalSimulation(Path(args.simulation_json)) as sim:
    with LumericalSimulation() as sim:
        # sim.promise_env_setup()
        sim._load_fsp('test_project/.tmp/sim_0.fsp')
        exit()

        vipdopt.logger.info('Saving Simulation')
        sim.save(Path('test_sim'))
        vipdopt.logger.info('Running simulation')
        sim.run()
