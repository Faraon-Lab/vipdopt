"""Abstractions of Lumerical simulation objects."""

from __future__ import annotations
import json
from collections import OrderedDict
from collections.abc import Callable
from enum import Enum
from typing import Any

import vipdopt


class LumericalSimObjectType(str, Enum):
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
        """Get the correct lumapi function to add an object."""
        if vipdopt.lumapi is None:
            raise ModuleNotFoundError(
                'Module "vipdopt.lumapi" has not yet been instatiated.'
            )
        return getattr(vipdopt.lumapi.FDTD, f'add{self.value}')


SOURCE_TYPES = [
    LumericalSimObjectType.DIPOLE,
    LumericalSimObjectType.TFSF,
    LumericalSimObjectType.GAUSSIAN,
]
MONITOR_TYPES = [
    LumericalSimObjectType.POWER,
    LumericalSimObjectType.PROFILE,
]
IMPORT_TYPES = [LumericalSimObjectType.IMPORT]


OBJECT_TYPE_NAME_MAP = {
    'FDTD': LumericalSimObjectType.FDTD,
    'GaussianSource': LumericalSimObjectType.GAUSSIAN,
    'DipoleSource': LumericalSimObjectType.DIPOLE,
    'Rectangle': LumericalSimObjectType.RECT,
    'Mesh': LumericalSimObjectType.MESH,
    'Import': LumericalSimObjectType.IMPORT,
    'IndexMonitor': LumericalSimObjectType.INDEX,
}


class LumericalSimObject:
    """Lumerical Simulation Object.

    Attributes:
        name (str): name of the object
        obj_type (LumericalSimObjectType): the type of object
        properties (OrderedDict[str, Any]): Map of named properties and their values
    """

    def __init__(self, name: str, obj_type: LumericalSimObjectType) -> None:
        """Create a LumericalSimObject."""
        self.name = name
        self.obj_type = obj_type
        self.info: OrderedDict[str, Any] = OrderedDict([('name', '')])
        self.properties: OrderedDict[str, Any] = OrderedDict()
        if obj_type != LumericalSimObjectType.FDTD:
            self.properties['name'] = name

    def __str__(self) -> str:
        """Return string representation of object."""
        return json.dumps(
            self.as_dict(),
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
            return (
                self.obj_type == __value.obj_type
                and self.properties == __value.properties
            )
        return super().__eq__(__value)

    def as_dict(self) -> dict:
        """Return a dictionary representation of this object."""
        return vars(self)

    @classmethod
    def from_fdtd(cls, obj: dict) -> LumericalSimObject:
        """Return a LumericalSimObject from the fdtd equivalent."""
        otype = obj['type']
        if otype == 'DFTMonitor':
            if obj['spatiazl interpolation'] == 'specified position':
                obj_type = LumericalSimObjectType.PROFILE
            else:
                obj_type = LumericalSimObjectType.POWER
        else:
            obj_type = OBJECT_TYPE_NAME_MAP[otype]
        oname = obj._id.name.split('::')[-1]
        sim_obj = LumericalSimObject(oname, obj_type)
        sim_obj.update(**obj._nameMap)

        return sim_obj


