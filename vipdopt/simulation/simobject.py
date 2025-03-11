"""Abstractions of simulation objects."""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Callable
from copy import copy
from enum import Enum
from pathlib import Path
from typing import Any
from overrides import override

import numpy as np
import numpy.typing as npt

import vipdopt


class SimObjectType(str, Enum):
    """Types of generic simulation objects."""

    FDTD = 'fdtd'
    MESH = 'mesh'
    GAUSSIAN = 'gaussian'
    TFSF = 'tfsf'
    PLANE = 'plane'
    DIPOLE = 'dipole'
    POWER = 'power'
    PROFILE = 'profile'
    INDEX = 'index'
    IMPORT = 'import'
    RECT = 'rect'
    CIRCLE = 'circle'

    def get_add_lumerical_function(self) -> Callable:
        """Get the correct lumapi function to add an object."""
        if vipdopt.lumapi is None:
            raise ModuleNotFoundError(
                'Module "vipdopt.lumapi" has not yet been instantiated.'
            )
        return getattr(vipdopt.lumapi.FDTD, f'add{self.value}')


SOURCE_TYPES = [
    SimObjectType.DIPOLE,
    SimObjectType.TFSF,
    SimObjectType.GAUSSIAN,
    SimObjectType.PLANE,
]
MONITOR_TYPES = [
    SimObjectType.POWER,
    SimObjectType.PROFILE,
]
IMPORT_TYPES = [SimObjectType.IMPORT]


OBJECT_TYPE_NAME_MAP = {
    'FDTD': SimObjectType.FDTD,
    'GaussianSource': SimObjectType.GAUSSIAN,
    'DipoleSource': SimObjectType.DIPOLE,
    'Rectangle': SimObjectType.RECT,
    'Mesh': SimObjectType.MESH,
    'Import': SimObjectType.IMPORT,
    'IndexMonitor': SimObjectType.INDEX,
}


class SimObject:
    """Generic Simulation Object.

    Attributes:
        name (str): name of the object
        obj_type (SimObjectType): the type of object
        properties (OrderedDict[str, Any]): Map of named properties and their values
    """

    def __init__(self, name: str, obj_type: SimObjectType) -> None:
        """Create a SimObject."""
        self.name = name
        self.obj_type = obj_type
        self.info: OrderedDict[str, Any] = OrderedDict([('name', '')])
        self.properties: OrderedDict[str, Any] = OrderedDict()
        if obj_type != SimObjectType.FDTD:
            self.properties['name'] = name

    def __repr__(self) -> str:
        """Return string representation of the object."""
        return json.dumps(
            self.as_dict(),
            indent=4,
            ensure_ascii=True,
        )

    def __str__(self) -> str:
        """Return a string version of the object."""
        return f'{self.obj_type} "{self.name}"'

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
        """Test equality of SimObjects."""
        if isinstance(__value, SimObject):
            return (
                self.obj_type == __value.obj_type
                and self.properties == __value.properties
            )
        return super().__eq__(__value)

    def __lt__(self, obj2: SimObject) -> bool:
        """Test if this object comes before another alphabetically."""
        return self.name < obj2.name

    def __gt__(self, obj2: SimObject) -> bool:
        """Test if this object comes after another alphabetically."""
        return self.name > obj2.name

    def __le__(self, obj2: SimObject) -> bool:
        """Test if this object is less than or equal to another alphabetically."""
        return self.name <= obj2.name

    def __ge__(self, obj2: SimObject) -> bool:
        """Test if this object is greater than or equal to another alphabetically."""
        return self.name >= obj2.name

    def as_dict(self) -> dict:
        """Return a dictionary representation of this object."""
        return vars(self)

    @classmethod
    def from_lumerical(cls, obj) -> SimObject:
        """Return a SimObject from the Lumerical equivalent."""
        otype = obj['type']
        if otype == 'DFTMonitor':
            if obj['spatial interpolation'] == 'specified position':
                obj_type = SimObjectType.PROFILE
            else:
                obj_type = SimObjectType.POWER
        else:
            obj_type = OBJECT_TYPE_NAME_MAP[otype]
        oname = obj._id.name.split('::')[-1]
        sim_obj = SimObject(oname, obj_type)
        sim_obj.update(**obj._nameMap)

        return sim_obj


class Import(SimObject):
    """Class representing a freeform import primitive of a Device."""

    def __init__(self, name: str) -> None:
        super().__init__(name, SimObjectType.IMPORT)
        # Create dummy values until otherwise
        self.n = None
        self.x = np.ones(1)
        self.y = np.ones(1)
        self.z = np.ones(1)

    def as_dict(self) -> dict:
        data = copy(vars(self))
        del data['n'], data['x'], data['y'], data['z']
        return data

    def set_nk2(self, n: npt.NDArray, x: npt.NDArray, y: npt.NDArray, z: npt.NDArray):
        """Set the value of the nk of this import primitive."""
        self.n = n
        self.x = x
        self.y = y
        self.z = z

    def get_nk2(self) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """Get the relevant data from in order to use `LumericalFDTD.importnk2`."""
        assert self.n is not None
        return (self.n, self.x, self.y, self.z)

class IndexMonitor(SimObject):
    """Class representing an index monitor in Lumerical."""

    def __init__(self, name: str) -> None:
        super().__init__(name, SimObjectType.INDEX)




class SimEncoder(json.JSONEncoder):
    """Encodes SimObjects in JSON format."""

    @override
    def default(self, o: Any) -> Any:
        if isinstance(o, SimObjectType):
            return {'obj_type': str(o)}
        if isinstance(o, SimObject):
            return copy(vars(o))
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.generic):
            return o.item()
        if isinstance(o, complex) and np.imag(o)==0:
            # We purposely want it to break for actual complex numbers
            return np.real(o)
        if isinstance(o, Path):
            return str(o)
        return super().default(o)

