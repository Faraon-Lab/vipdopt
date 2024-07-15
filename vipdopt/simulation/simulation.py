"""FDTD Simulation Interface Code."""

from __future__ import annotations

import abc
import json
import logging
from argparse import ArgumentParser
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from copy import copy
from pathlib import Path
from typing import Any, overload

import numpy as np
from overrides import override

import vipdopt

# from vipdopt.optimization import Device
from vipdopt.simulation.monitor import Monitor
from vipdopt.simulation.simobject import (
    IMPORT_TYPES,
    MONITOR_TYPES,
    SOURCE_TYPES,
    Import,
    LumericalSimObject,
    LumericalSimObjectType,
)
from vipdopt.simulation.source import Source
from vipdopt.utils import PathLike, ensure_path, read_config_file

# TODO: Create simulation subpackage and add this, monitors, sources, and maybe devices


class ISimulation(abc.ABC):
    """Abstract class for an FDTD simulation."""

    @abc.abstractmethod
    def __init__(self, source: PathLike | dict | None) -> None:
        """Create simulation object."""

    @abc.abstractmethod
    def load(self, source: PathLike | dict):
        """Load simulation from a file or dictionary."""

    @abc.abstractmethod
    def save(self, fname: PathLike):
        """Save simulation data to a file."""


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


class LumericalSimulation(ISimulation):
    """Lumerical FDTD Simulation Code.

    Attributes:
        info (OrderedDict[str]): Info about this simulation. Contains the keys
            "filename", "path", "simulator", and "coordinates".
        objects (OrderedDict[str, LumericalSimObject]): The objects within
            the simulation
    """

    def __init__(self, source: PathLike | dict | None = None) -> None:
        """Create a LumericalSimulation.

        Arguments:
            source (PathLike | dict | None): optional source to load the simulation from
        """
        self.info: OrderedDict[str, Any] = OrderedDict([
            ('name', ''),
            ('path', None),
        ])
        self.clear_objects()

        if source:
            self.load(source)

    @override
    def __str__(self) -> str:
        return self.as_json()

    @override
    def load(self, source: PathLike | dict):
        if isinstance(source, dict):
            self._load_dict(source)
        else:
            self._load_file(source)

    @ensure_path
    def _load_file(self, fname: Path):
        """Load a simulation from a JSON file."""
        vipdopt.logger.debug(f'Loading simulation from {fname}...')
        sim = read_config_file(fname)
        self._load_dict(sim)
        vipdopt.logger.info(f'Succesfully loaded {fname}\n')

    def _load_dict(self, d: dict):
        """Load a simulation from a dictionary."""
        self._clear_info()
        self.info.update(d.get('info', {}))

        self.clear_objects()
        for obj in d['objects'].values():
            self.new_object(
                obj['name'],
                LumericalSimObjectType(obj['obj_type']),
                **obj['properties'],
            )

    @ensure_path
    @override
    def save(self, fname: Path):
        content = self.as_json()
        with open(fname.with_suffix('.json'), 'w', encoding='utf-8') as f:
            f.write(content)
        vipdopt.logger.debug(f'Succesfully saved simulation to {fname}.\n')

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

    def _clear_info(self):
        """Clear all existing simulation info and create a new project."""
        self.info = OrderedDict()

    def clear_objects(self):
        """Clear all existing objects and create a new project."""
        self.objects: OrderedDict[str, LumericalSimObject] = OrderedDict()

    @ensure_path
    def set_path(self, path: Path):
        """Set the save path of the simulation."""
        self.info['path'] = path.absolute()
    
    def get_path(self) -> Path | None:
        """Get the save path of the simulation."""
        return self.info.get('path', None)

    def copy(self) -> LumericalSimulation:
        """Return a copy of this simulation."""
        new_sim = LumericalSimulation()
        new_sim.info = self.info.copy()
        for obj_name, obj in self.objects.items():
            new_sim.new_object(obj_name, obj.obj_type, **obj.properties)

        return new_sim

    def with_monitors(
        self, objs: Iterable[str] | Iterable[Monitor], name: str | None = None
    ):
        """Return a copy of this simulation with only the specified monitors."""
        new_sim = self.copy()  # Unlinked
        if name is not None:
            # Create the simulation with the provided name
            new_sim.info['name'] = name
        names = [m.name if isinstance(m, Monitor) else m for m in objs]
        for name in new_sim.monitor_names():
            if name not in names:
                del self.objects[name]
        return new_sim

    def enable(self, names: Iterable[str]):
        """Enable all objects in provided list."""
        for name in names:
            self.update_object(name, enabled=1)

    def with_enabled(
        self,
        objs: Iterable[Source] | Iterable[str],
        name: str | None = None,
    ) -> LumericalSimulation:
        """Return copy of this simulation with only objects in objs enabled."""
        new_sim = self.copy()  # Unlinked
        if name is not None:
            # Create the simulation with the provided name
            new_sim.info['name'] = name
        new_sim.disable_all_sources()
        names = [o.name if isinstance(o, LumericalSimObject) else o for o in objs]
        new_sim.enable(names)
        return new_sim

    def disable(self, names: Iterable[str]):
        """Disable all objects in provided list."""
        for name in names:
            self.update_object(name, enabled=0)

    def with_disabled(
        self,
        objs: Iterable[Source] | Iterable[str],
        name: str | None = None,
    ) -> LumericalSimulation:
        """Return copy of this simulation with only objects in objs disabled."""
        new_sim = self.copy()  # Unlinked
        if name is not None:
            # Create the simulation with the provided name
            new_sim.info['name'] = name
        new_sim.enable_all_sources()
        names = [o.name if isinstance(o, LumericalSimObject) else o for o in objs]
        new_sim.disable(names)
        return new_sim

    def enable_all_sources(self):
        """Enable all sources in this simulation."""
        self.enable(self.source_names())

    def disable_all_sources(self):
        """Disable all sources in this simulation."""
        self.disable(self.source_names())

    def sources(self) -> list[LumericalSimObject]:
        """Return a list of all source objects."""
        return [obj for _, obj in self.objects.items() if isinstance(obj, Source)]

    def source_names(self) -> Iterator[str]:
        """Return a list of all source object names."""
        for obj in self.sources():
            yield obj.name

    def monitors(self) -> list[Monitor]:
        """Return a list of all monitor objects."""
        return [obj for _, obj in self.objects.items() if isinstance(obj, Monitor)]
        # return [obj for _, obj in self.objects.items() if obj.obj_type in MONITOR_TYPES]

    def monitor_names(self) -> Iterator[str]:
        """Return a list of all monitor object names."""
        for obj in self.monitors():
            yield obj.name

    @overload
    def link_monitors(self): ...

    @overload
    def link_monitors(self, monitors: list[Monitor]): ...

    def link_monitors(self, monitors: list[Monitor] | None = None):
        """Link all of the given monitors to this simulation.

        Requires self.info['path'] to have been set already!!
        """
        sim_path: Path = self.info['path']
        if monitors is None:
            monitors = self.monitors()
        for mon in monitors:
            output_path = sim_path.parent / (sim_path.stem + f'_{mon.name}.npz')
            mon.set_source(output_path)

    def imports(self) -> list[Import]:
        """Return a list of all import objects."""
        return [obj for _, obj in self.objects.items() if isinstance(obj, Import)]

    def import_names(self) -> Iterator[str]:
        """Return a list of all import object names."""
        for obj in self.imports():
            yield obj.name

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
        obj: LumericalSimObject
        if obj_type in MONITOR_TYPES:
            obj = Monitor(obj_name, obj_type)
        elif obj_type in SOURCE_TYPES:
            obj = Source(obj_name, obj_type)
        elif obj_type in IMPORT_TYPES:
            obj = Import(obj_name)
        else:
            obj = LumericalSimObject(obj_name, obj_type)
        obj.update(**properties)
        self.add_object(obj)
        return obj

    def add_object(self, obj: LumericalSimObject) -> None:
        """Add an existing object to the simulation."""
        # Add copy to the vipdopt.lumapi.FDTD
        self.objects[obj.name] = obj

    def update_object(self, name: str, **properties):
        """Update object with new property values."""
        obj = self.objects[name]
        obj.update(**properties)

    def __eq__(self, __value: object) -> bool:
        """Test equality of simulations."""
        if isinstance(__value, LumericalSimulation):
            return self.objects == __value.objects
        return super().__eq__(__value)


ISimulation.register(LumericalSimulation)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        'simulation_json',
        type=str,
        help='Simulation JSON to create a simulation from',
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        default=False,
        help='Enable verbose output.',
    )

    args = parser.parse_args()

    # Set verbosity
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level)

    # with LumericalSimulation(Path(args.simulation_json)) as sim:
    # with LumericalSimulation() as sim:
    #     # sim.promise_env_setup()
    #     sim.load('test_project/.tmp/sim_0.fsp')
    #     sys.exit()

    #     vipdopt.logger.info('Saving Simulation')
    #     sim.save(Path('test_sim'))
    #     vipdopt.logger.info('Running simulation')
    #     sim.run()
