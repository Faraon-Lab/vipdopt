"""FDTD Simulation Interface Code."""
from __future__ import annotations

import abc
import json
import logging
import time
from argparse import ArgumentParser
from collections import OrderedDict
from collections.abc import Callable
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from overrides import override

from vipdopt import lumapi
from vipdopt.utils import PathLike, ensure_path, read_config_file


class ISimulation(abc.ABC):
    """Abstract class for an FDTD simulation."""

    @abc.abstractmethod
    def __init__(self, fname: PathLike | None) -> None:
        """Create simulation object."""

    @abc.abstractmethod
    def connect(self):
        """Create hook to simulation software."""

    @abc.abstractmethod
    def load(self, fname: PathLike):
        """Load simulation from a file."""

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
        return getattr(lumapi.FDTD, f'add{self.value}')

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

class LumericalEncoder(json.JSONEncoder):
    """Encodes LumericalSim objects in JSON format."""
    @override
    def default(self, o: Any) -> Any:
        if isinstance(o, LumericalSimulation):
            d = o.__dict__.copy()
            del d['fdtd']
            return d
        if isinstance(o, LumericalSimObjectType):
            return {'obj_type': str(o)}
        if isinstance(o, LumericalSimObject):
            d = o.__dict__.copy()
            del d['lumapi_obj']
            return d
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
        self.lumapi_obj = None

    def __str__(self) -> str:
        """Return string representation of object."""
        return json.dumps(
            self.__dict__,
            indent=4,
            ensure_ascii=True,
        )

    def __setitem__(self, key: str, val: Any) -> None:
        """Set the value of a property of the object."""
        self.properties[key] = val
        if self.lumapi_obj is not None:
            self.lumapi_obj[key] = val

    def __getitem__(self, key: str) -> Any:
        """Retrieve a property from an object."""
        return self.properties[key]

    def update(self, **vals):
        """Update properties with values in a dictionary."""
        self.properties.update(vals)
        if self.lumapi_obj is not None:
            for k, v in vals.items():
                self.lumapi_obj[k] = v

    def __eq__(self, __value: object) -> bool:
        """Test equality of LumericalSimObjects."""
        if isinstance(__value, LumericalSimObject):
            return self.obj_type == __value.obj_type and \
                self.properties == __value.properties
        return super().__eq__(__value)


class LumericalSimulation(ISimulation):
    """Lumerical FDTD Simulation Code.

    Attributes:
        config (vipdopt.Config): configuration object
        fdtd (lumapi.FDTD): Lumerical session
        objects (OrderedDict[str, LumericalSimObject]): The objects within
            the simulation
    """

    def __init__(self, fname: PathLike | None=None) -> None:
        """Create a LumericalSimulation.

        Arguments:
            fname (str): optional filename to load the simulation from
        """
        self.fdtd: lumapi.FDTD = None
        self.connect()
        self._clear_objects()
        if fname:
            self.load(fname)

    @override
    def __str__(self) -> str:
        return json.dumps(
            dict(self.__dict__),
            indent=4,
            ensure_ascii=True,
            cls=LumericalEncoder,
        )

    @override
    def connect(self) -> None:
        logging.info('Attempting to connect to Lumerical servers...')

        # This loop handles the problem of
        # "Failed to start messaging, check licenses..." that Lumerical often has.
        while self.fdtd is None:
            try:
                self.fdtd = lumapi.FDTD(hide=True)
            except (AttributeError, lumapi.LumApiError) as e:
                logging.exception(
                    'Licensing server error - can be ignored.',
                    exc_info=e,
                )
                continue

        logging.info('Verified license with Lumerical servers.\n')

    @override
    def load(self, fname: PathLike):
        fpath = ensure_path(fname)
        logging.info(f'Loading simulation from {fpath}...')
        self._clear_objects()
        sim = read_config_file(fpath)
        for obj in sim['objects'].values():
            self.new_object(
                obj['name'],
                LumericalSimObjectType(obj['obj_type']),
                **obj['properties'],
            )

        logging.info(f'Succesfully loaded {fpath}\n')

    @override
    def save(self, fname: PathLike):
        fpath = ensure_path(fname)
        logging.info(f'Saving simulation to {fpath}...')
        self.fdtd.save(str(fpath.with_suffix('')))
        with open(fpath.with_suffix('.json'), 'w', encoding='utf-8') as f:
            json.dump(
                {'objects': self.objects},
                f,
                indent=4,
                ensure_ascii=True,
                cls=LumericalEncoder,
            )
        logging.info(f'Succesfully saved simulation to {fpath}.\n')

    @override
    def run(self):
        self.fdtd.run()

    def set_resource(self, resource_num: int, resource: str, value: Any):
        """Set the specified job manager resource for this simulation."""
        self.fdtd.setresource('FDTD', resource_num, resource, value)

    def _clear_objects(self):
        """Clear all existing objects and create a new project."""
        self.objects = OrderedDict()
        self.fdtd.newproject()

    def copy(self) -> LumericalSimulation:
        """Return a copy of this simulation."""
        new_sim = LumericalSimulation()
        for obj_name, obj in self.objects.items():
            new_sim.new_object(obj_name, obj.obj_type, **obj.properties)

        return new_sim

    def enable(self, objs: list[str]):
        """Enable all objects in provided list."""
        self.fdtd.switchtolayout()
        for obj in objs:
            self.update_object(obj, enabled=1)

    def enabled(self, objs: list[str]) -> LumericalSimulation:
        """Return copy of this simulation with all objects in objs enabled."""
        new_sim = self.copy()
        new_sim.enable(objs)
        return new_sim

    def disable(self, objs: list[str]):
        """Disable all objects in provided list."""
        self.fdtd.switchtolayout()
        for obj in objs:
            self.update_object(obj, enabled=0)

    def disabled(self, objs: list[str]) -> LumericalSimulation:
        """Return copy of this simulation with all objects in objs disabled."""
        new_sim = self.copy()
        new_sim.disable(objs)
        return new_sim

    def disable_all_sources(self):
        """Disable all sources in this simulation."""
        to_disable = []

        for obj_name, obj in self.objects.items():
            if obj.obj_type in SOURCE_TYPES:
                to_disable.append(obj_name)

        self.disable(to_disable)

    def sources(self):
        """Return a list of all source objects."""
        return [obj for _, obj in self.objects.items() if obj.type in SOURCE_TYPES]

    def monitors(self):
        """Return a list of all monitors objects."""
        return [obj for _, obj in self.objects.items() if obj.type in MONITOR_TYPES]

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
        logging.debug(f"Creating new object: '{obj_name}'...")
        obj = LumericalSimObject(obj_name, obj_type)
        obj.update(**properties)
        self.add_object(obj)
        return obj

    def add_object(self, obj: LumericalSimObject) -> None:
        """Add an existing object to the simulation."""
        if self.fdtd is None:
            self.objects[obj.name] = obj
            return

        lumapi_obj = LumericalSimObjectType.get_add_function(obj.obj_type)(
            self.fdtd,
            **obj.properties,
        )
        obj.lumapi_obj = lumapi_obj
        self.objects[obj.name] = obj

    def update_object(self, name: str, **properties):
        """Update lumerical object with new property values."""
        obj = self.objects[name]
        assert self.fdtd is not None
        for key, val in properties.items():
            self.fdtd.setnamed(obj.name, key, val)
        obj.update(**properties)

    def close(self):
        """Close fdtd conenction."""
        if self.fdtd is not None:
            logging.info('Closing connection with Lumerical...')
            self.fdtd.close()
            logging.info('Succesfully closed connection with Lumerical.')


    def __eq__(self, __value: object) -> bool:
        """Test equality  of simulations."""
        if isinstance(__value, LumericalSimulation):
            return self.objects == __value.objects
        return super().__eq__(__value)

    def __enter__(self):
        """Startup for context manager usage."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup after exiting a context."""
        self.close()

    def get_hpc_resources(self) -> dict:
        """Return a dictionary containing all job manager resources."""
        resources = {}
        for resource in self.fdtd.setresource('FDTD', 1).splitlines():
            resources[resource] = self.fdtd.getresource('FDTD', 1, resource)
        return resources

    def setup_hpc_resources(self, **kwargs):
        """Set the job manager resources for runnign this simulation.

        **kwargs:
            mpi_exe (Path): Path to the mpi executable to use
            nprocs (int): The number of processes to run the simulation with
            hostfile (Path | None): Path to a hostfile for running distributed jobs
            solver_exe (Path): Path to the fdtd-solver to run
        """
        self.set_resource(1, 'mpi no default options', '1')

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
        logging.debug('Updated simulation resources:')
        for resource, value in self.get_hpc_resources().items():
            logging.debug(f'\t"{resource}" = {value}')

    def get_field_shape(self) -> tuple[int, ...]:
        """Return the shape of the fields returned from this simulation's monitors."""
        index_prev = self.fdtd.getresult('design_index_monitor', 'index preview')
        return np.squeeze(index_prev['index_x']).shape

    def get_transimission_magnitude(self, monitor_name: str) -> npt.NDArray:
        """Return the magnitude of the transmission for a given monitor."""
        # Ensure this is a transmission monitor
        monitor_name = monitor_name.replace('focal', 'transmission')
        return np.abs(self.fdtd.getresult(monitor_name, 'T')['T'])

    def get_field(self, monitor_name: str, field_indicator: str) -> npt.NDArray:
        """Return the E or H field from a monitor."""
        if field_indicator not in 'EH':
            raise ValueError('Expected field_indicator to be "E" or "H";'
                             f' got {field_indicator}')
        polarizations = [field_indicator + c for c in 'xyz']

        start = time.time()
        logging.debug(f'Getting {polarizations} from monitor "{monitor_name}"')
        fields = np.array(
            map(partial(self.fdtd.getdata(monitor_name)), polarizations),
            dtype=np.complex128,
        )
        data_xfer_size_mb = fields.nbytes / (1024 ** 2)
        elapsed = time.time() - start

        logging.debug(f'Transferred {data_xfer_size_mb} MB')
        logging.debug(f'Data rate = {data_xfer_size_mb / elapsed} MB/sec')
        return fields

    def get_hfield(self, monitor_name: str) -> npt.NDArray:
        """Return the H field from a monitor."""
        return self.get_field(monitor_name, 'H')

    def get_efield(self, monitor_name: str) -> npt.NDArray:
        """Return the E field from a monitor."""
        return self.get_field(monitor_name, 'E')

    def get_pfield(self, monitor_name: str) -> npt.NDArray:
        """Returns power as a function of space and wavelength, with xyz components."""
        return self.fdtd.getresult(monitor_name, 'P')['P']

    def get_efield_magnitude(self, monitor_name: str) -> npt.NDArray:
        """Return the magnitude of the E field from a monitor."""
        efield = self.get_efield(monitor_name)
        enorm_squared = np.sum(np.square(np.abs(efield)), axis=0)
        return np.sqrt(enorm_squared)

    def get_source_power(self) -> npt.NDArray:
        """Return the source power of a given monitor."""
        f = self.fdtd.getresult('focal_monitor_0', 'f')
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

    with LumericalSimulation(Path(args.simulation_json)) as sim:
        sim.setup_hpc_resources()

        logging.info('Saving Simulation')
        sim.save(Path('test_sim'))
        logging.info('Running simulation')
        sim.run()
