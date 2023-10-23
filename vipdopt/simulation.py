"""FDTD Simulation Interface Code."""

import abc
import json
import logging
import os
import sys
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from overrides import override

if __name__ == '__main__':
    f = sys.modules[__name__].__file__
    if not f:
        raise ModuleNotFoundError('SHOULD NEVER REACH HERE')

    path = os.path.dirname(f)
    path = os.path.join(path, '..')
    sys.path.insert(0, path)


# TODO: Handle import for lumapi


class ISimulation(abc.ABC):
    """Abstract class for an FDTD simulation."""

    @abc.abstractmethod
    def __init__(self, fname: str | None) -> None:
        """Create simulation object."""

    @abc.abstractmethod
    def connect(self):
        """Create hook to simulation software."""

    @abc.abstractmethod
    def load(self, fname: str):
        """Load simulation from a file."""

    @abc.abstractmethod
    def save(self, fname: str):
        """Save simulation data to a file."""

    # def update_obj(self, objname: str, properties: dict[str, Any]) -> Any:
    #     """Update a simulation object."""


class LumericalSimObjectType(StrEnum):
    """Types of simulation objects in Lumerical."""
    fdtd = 'fdtd'
    mesh = 'mesh'
    gaussian = 'gaussian'
    tfsf = 'tfsf'
    dipole = 'dipole'
    power = 'power'
    profile = 'profile'
    lum_index = 'lum_index'
    lum_import = 'lum_import'
    rect = 'rect'

    def get_add_function(self) -> Callable:
        """Get the corerct lumapi function to add an object."""
        return lambda o: o
        # TODO: ADD the following
        # match self.value
        #     case LumericalSimObjectType.fdtd:
        #     case LumericalSimObjectType.mesh:
        #     case LumericalSimObjectType.gaussian:
        #     case LumericalSimObjectType.tfsf:
        #     case LumericalSimObjectType.dipole:
        #     case LumericalSimObjectType.power:
        #     case LumericalSimObjectType.profile:
        #     case LumericalSimObjectType.lum_index:
        #     case LumericalSimObjectType.lum_import:
        #     case LumericalSimObjectType.rect:


class LumericalEncoder(json.JSONEncoder):
    """Encodes LumericalSim objects in JSON format."""
    @override
    def default(self, o: Any) -> Any:
        if isinstance(o, LumericalSimulation):
            d = o.__dict__.copy()
            del d['fdtd']
            return d
        if isinstance(o, LumericalSimObjectType):
            return {'__type__': str(o)}
        if isinstance(o, LumericalSimObject):
            return o.__dict__
        return super().default(o)


class LumericalSimObject:
    """Lumerical Simulation Object.

    Attributes:
        name (str): name of the object
        obj_type (LumericalSimObjectType): the type of object
        properties (dict[str, Any]): Map of named properties and their values
    """
    def __init__(self, name: str, obj_type: LumericalSimObjectType) -> None:
        """Crearte a LumericalSimObject."""
        self.name = name
        self.obj_type = obj_type
        self.properties: dict[str, Any] = {}

    def __str__(self) -> str:
        """Return string representation of object."""
        return json.dumps(
            self.__dict__,
            indent=4,
            ensure_ascii=True,
        )

    def __setitem__(self, key: str, val: Any) -> Any:
        """Set the value of a property of the object."""
        old_val = self.properties.get(key)
        self.properties[key] = val
        return old_val

    def __getitem__(self, key: str) -> Any:
        """Retrieve a property from an object."""
        return self.properties[key]

    def update(self, vals: dict):
        """Update properties with values in a dictionary."""
        self.properties.update(vals)

    def __eq__(self, __value: object) -> bool:
        """Test equality of LumericalSimObjects."""
        if isinstance(__value, LumericalSimObject):
            return self.name == __value.name and \
                self.obj_type == __value.obj_type and \
                self.properties == __value.properties
        return False


class LumericalSimulation(ISimulation):
    """Lumerical FDTD Simulation Code.

    Attributes:
        config (vipdopt.Config): configuration object
        fdtd (lumapi.FDTD): Lumerical session
        objects (dict[str, LumericalSimObject]): The objects within
            the simulation
    """

    def __init__(self, fname: str | None) -> None:
        """Create a LumericalSimulation.

        Arguments:
            fname (str): optional filename to load the simulation from
        """
        self.fdtd = None
        self.objects: dict[str, LumericalSimObject] = {}
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
        # This loop handles the problem of
        # "Failed to start messaging, check licenses..." that Lumerical often has.
        while self.fdtd is None:
            try:
                # TODO: self.fdtd = lumapi.FDTD()
                pass
            except Exception:
                logging.exception('Licensing server error - can be ignored.')
                continue
        logging.info('Verified license with Lumerical servers.')

    @override
    def load(self, fname: str):
        self.fdtd = None
        self.objects = {}
        with open(fname) as f:
            sim = json.load(f)
            for obj in sim['objects'].values():
                self.new_object(obj['name'], obj['obj_type'], obj['properties'])

    @override
    def save(self, fname: str):
        with open(fname, 'w', encoding='utf-8') as f:
            json.dump(
                {'objects': self.objects},
                f,
                indent=4,
                ensure_ascii=True,
                cls=LumericalEncoder,
            )

    def new_object(
        self,
        obj_name: str,
        obj_type: LumericalSimObjectType,
        properties: dict[str, Any] | None=None
    ) -> LumericalSimObject:
        """Create a new object and add it to the simulation.

        Arguments:
            obj_name (str): Name of the object
            obj_type (LumericalSimObjectType): type of the object
            properties (dict[str, Any]): optional dictionary to populate
                the new object with
        """
        if properties is None:
            properties = {}
        obj = LumericalSimObject(obj_name, obj_type)
        obj.update(properties)
        self.add_object(obj)
        return obj

    def add_object(self, obj: LumericalSimObject) -> None:
        """Add an existing object to the simulation."""
        if self.fdtd is None:
            self.objects[obj.name] = obj
            return

        obj.obj_type.get_add_function()(properties=obj.properties)
        self.objects[obj.name] = obj

    def update_object(self, name: str, properties: dict):
        """Update lumerical object with new property values."""
        obj = self.objects[name]
        assert self.fdtd is not None
        lum_obj = self.fdtd.getnamed(obj.name)
        for key, val in properties.items():
            lum_obj[key] = val
        obj.update(properties)

    def __enter__(self):
        """Startup for context manager usage."""
        return self

    def __exit__(self, *args):
        """Cleanup after exiting a context."""
        if self.fdtd:
            self.fdtd.close()


if __name__ == '__main__':
    with LumericalSimulation(fname=None) as sim:
        props1 = {'width': 2, 'height': 5}
        obj1 = sim.new_object('obj1', LumericalSimObjectType.mesh, props1)
        props2 = {'x span': 10, 'y span': 7, 'simulation time': 300}
        obj2 = sim.new_object('obj2', LumericalSimObjectType.fdtd, props2)
        sim.save('testout.json')

    sim2 = LumericalSimulation('testout.json')
