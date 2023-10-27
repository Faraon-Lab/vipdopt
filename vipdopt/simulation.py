"""FDTD Simulation Interface Code."""

import abc
import json
import logging
import os
import sys
from collections import OrderedDict
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

from vipdopt import lumapi


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
        match self.value:
            case LumericalSimObjectType.lum_index:
                return lumapi.FDTD.addindex
            case LumericalSimObjectType.lum_import:
                return lumapi.FDTD.addimport
            case _:
                return getattr(lumapi.FDTD, f'add{self.value}')


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
        self.lumapi_obj[key] = val

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
        return super.__eq__(__value)


class LumericalSimulation(ISimulation):
    """Lumerical FDTD Simulation Code.

    Attributes:
        config (vipdopt.Config): configuration object
        fdtd (lumapi.FDTD): Lumerical session
        objects (OrderedDict[str, LumericalSimObject]): The objects within
            the simulation
    """

    def __init__(self, fname: str | None=None) -> None:
        """Create a LumericalSimulation.

        Arguments:
            fname (str): optional filename to load the simulation from
        """
        self.fdtd = None
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
    def load(self, fname: str):
        logging.info(f'Loading simulation from {fname}...')
        self._clear_objects()
        with open(fname) as f:
            sim = json.load(f)
            for obj in sim['objects'].values():
                self.new_object(obj['name'], LumericalSimObjectType[obj['obj_type']], **obj['properties'])

        logging.info(f'Succesfully loaded {fname}\n')

    @override
    def save(self, fname: str):
        logging.info(f'Saving simulation to {fname}...')
        with open(fname, 'w', encoding='utf-8') as f:
            json.dump(
                {'objects': self.objects},
                f,
                indent=4,
                ensure_ascii=True,
                cls=LumericalEncoder,
            )
        logging.info(f'Succesfully saved simulation to {fname}.\n')
    
    def _clear_objects(self):
        """Clear all existing objects and create a new project."""
        self.objects: OrderedDict[str, LumericalSimObject] = OrderedDict()
        self.fdtd.newproject()

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
        logging.debug(f'Creating new object: \'{obj_name}\'...')
        obj = LumericalSimObject(obj_name, obj_type)
        obj.update(**properties)
        self.add_object(obj)
        return obj

    def add_object(self, obj: LumericalSimObject) -> None:
        """Add an existing object to the simulation."""
        if self.fdtd is None:
            self.objects[obj.name] = obj
            return

        lumapi_obj = LumericalSimObjectType.get_add_function(obj.obj_type)(self.fdtd, **obj.properties)
        obj.lumapi_obj = lumapi_obj
        self.objects[obj.name] = obj

    def update_object(self, name: str, **properties):
        """Update lumerical object with new property values."""
        obj = self.objects[name]
        assert self.fdtd is not None
        lum_obj = self.fdtd.getnamed(obj.name)
        for key, val in properties.items():
            lum_obj[key] = val
        obj.update(properties)
    
    def close(self):
        """Close fdtd conenction."""
        if self.fdtd is not None:
            logging.info('Closing connection with Lumerical...')
            self.fdtd.close()
            logging.info('Succesfully closed connection with Lumerical.')

    
    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, LumericalSimulation):
            return self.objects == __value.objects
        return super().__eq__(__value)

    def __enter__(self):
        """Startup for context manager usage."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup after exiting a context."""
        self.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    sim1 = LumericalSimulation()
    props1 = {'x': 2, 'y': 5}
    obj1 = sim1.new_object('obj1', LumericalSimObjectType.mesh, **props1)
    props2 = {'x span': 10, 'y span': 7, 'simulation time': 300}
    obj2 = sim1.new_object('obj2', LumericalSimObjectType.fdtd, **props2)
    sim1.save('testout.json')

    with LumericalSimulation('testout.json') as sim:
        assert sim == sim1
    
    sim1.close()
