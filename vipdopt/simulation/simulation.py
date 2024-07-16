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
from scipy import interpolate  # type: ignore

import vipdopt

# from vipdopt.optimization import Device
from vipdopt.simulation.monitor import Monitor
from vipdopt.simulation.simobject import (
    IMPORT_TYPES,
    MONITOR_TYPES,
    SOURCE_TYPES,
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
        self.info: OrderedDict[str, Any] = OrderedDict([('name', '')])
        self.clear_objects()
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
    def load(self, source: PathLike | dict):
        if isinstance(source, dict):
            self._load_dict(source)
        else:
            self._load_file(source)

    # @_check_fdtd
    @ensure_path
    def _load_fsp(self, fname: Path):
        """Load a simulation from a Lumerical .fsp file."""
        vipdopt.logger.debug(f'Loading simulation from {fname}...')
        self._clear_objects()
        self.fdtd.load(str(fname))  # type: ignore
        self.fdtd.selectall()  # type: ignore
        objects = self.fdtd.getAllSelectedObjects()  # type: ignore
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
            oname = o._id.name.split('::')[-1]  # noqa: SLF001
            sim_obj = LumericalSimObject(oname, obj_type)
            for name in o._nameMap:  # noqa: SLF001
                sim_obj[name] = o[name]

            self.objects[oname] = sim_obj
        vipdopt.logger.debug(self.as_json())

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

    def get_path(self) -> Path:
        """Get the save path of the simulation."""
        p = self.info['path']
        if not isinstance(p, Path):
            p = Path(p)
        return p

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
            mon.set_src(output_path)

    def imports(self) -> list[LumericalSimObject]:
        """Return a list of all import objects."""
        return [obj for _, obj in self.objects.items() if obj.obj_type in IMPORT_TYPES]

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

    def import_nk_material(
        #! TODO: 20240702 - MOVE TO FDTD. Add the input argument of the device name as string.
        self,
        cur_index,
        device_region_import_x,
        device_region_import_y,
        device_region_import_z,
    ):
        """Import the nk2 material."""
        device_name = list(self.import_names())[0]

        #! TODO: eRASE THIS
        self.fdtd = vipdopt.fdtd.fdtd

        self.fdtd.select(device_name)
        self.fdtd.importnk2(
            cur_index,
            device_region_import_x,
            device_region_import_y,
            device_region_import_z,
        )

    def import_cur_index(
        self,
        design: Any,
        reinterpolate_permittivity_factor=1,
        reinterpolate_permittivity=False,
        binary_design=False,
    ):
        """Reinterpolate and import cur_index into design regions."""
        if reinterpolate_permittivity_factor != 1:
            reinterpolate_permittivity = True

        # Get current density after being passed through current extant filters.
        # TODO: Might have to rewrite this to be performed on the entire device and NOT
        # the subdesign / device. In which case.
        # TODO: partitioning must happen after reinterpolating and converting to index.

        # Here cur_density will have values between 0 and 1
        cur_density = design.get_density()

        # Reinterpolate
        if reinterpolate_permittivity:
            cur_density_import = np.repeat(
                np.repeat(
                    np.repeat(
                        cur_density,
                        int(np.ceil(reinterpolate_permittivity_factor)),
                        axis=0,
                    ),
                    int(np.ceil(reinterpolate_permittivity_factor)),
                    axis=1,
                ),
                int(np.ceil(reinterpolate_permittivity_factor)),
                axis=2,
            )

            # Create the axis vectors for the reinterpolated 3D density region
            design_region_reinterpolate_x = 1e-6 * np.linspace(
                design.coords['x'][0],
                design.coords['x'][-1],
                len(design.coords['x']) * reinterpolate_permittivity_factor,
            )
            design_region_reinterpolate_y = 1e-6 * np.linspace(
                design.coords['y'][0],
                design.coords['y'][-1],
                len(design.coords['y']) * reinterpolate_permittivity_factor,
            )
            design_region_reinterpolate_z = 1e-6 * np.linspace(
                design.coords['z'][0],
                design.coords['z'][-1],
                len(design.coords['z']) * reinterpolate_permittivity_factor,
            )
            # Create the 3D array for the 3D density region for final import
            design_region_import_x = 1e-6 * np.linspace(
                design.coords['x'][0], design.coords['x'][-1], 300
            )
            design_region_import_y = 1e-6 * np.linspace(
                design.coords['y'][0], design.coords['y'][-1], 300
            )
            design_region_import_z = 1e-6 * np.linspace(
                design.coords['z'][0], design.coords['z'][-1], 306
            )
            design_region_import = np.array(
                np.meshgrid(
                    design_region_import_x,
                    design_region_import_y,
                    design_region_import_z,
                    indexing='ij',
                )
            ).transpose((1, 2, 3, 0))

            # Perform reinterpolation so as to fit into Lumerical mesh.
            # NOTE: The imported array does not need to have the same size as the
            # Lumerical mesh. Lumerical will perform its own reinterpolation
            # (params. of which are unclear).
            cur_density_import_interp = interpolate.interpn(
                (
                    design_region_reinterpolate_x,
                    design_region_reinterpolate_y,
                    design_region_reinterpolate_z,
                ),
                cur_density_import,
                design_region_import,
                method='linear',
            )

        else:
            cur_density_import_interp = cur_density.copy()

            design_region_x = 1e-6 * np.linspace(
                design.coords['x'][0],
                design.coords['x'][-1],
                cur_density_import_interp.shape[0],
            )
            design_region_y = 1e-6 * np.linspace(
                design.coords['y'][0],
                design.coords['y'][-1],
                cur_density_import_interp.shape[1],
            )
            design_region_z = 1e-6 * np.linspace(
                design.coords['z'][0],
                design.coords['z'][-1],
                cur_density_import_interp.shape[2],
            )

            design_region_import_x = 1e-6 * np.linspace(
                design.coords['x'][0],
                design.coords['x'][-1],
                design.field_shape[0],
            )
            design_region_import_y = 1e-6 * np.linspace(
                design.coords['y'][0],
                design.coords['y'][-1],
                design.field_shape[1],
            )
            try:
                design_region_import_z = 1e-6 * np.linspace(
                    design.coords['z'][0], design.coords['z'][-1], design.field_shape[2]
                )
            except IndexError:  # 2D case
                design_region_import_z = 1e-6 * np.linspace(
                    design.coords['z'][0], design.coords['z'][-1], 3
                )
            design_region_import = np.array(
                np.meshgrid(
                    design_region_import_x,
                    design_region_import_y,
                    design_region_import_z,
                    indexing='ij',
                )
            ).transpose((1, 2, 3, 0))

            cur_density_import_interp = interpolate.interpn(
                (design_region_x, design_region_y, design_region_z),
                cur_density_import_interp,
                design_region_import,
                method='linear',
            )

        # # IF YOU WANT TO BINARIZE BEFORE IMPORTING:
        if binary_design:
            cur_density_import_interp = design.binarize(cur_density_import_interp)

        # TODO: Add dispersion. Account for dispersion by using the dispersion_model
        for _dispersive_range_idx in range(1):
            # max_device_permittivity
            # dispersive_max_permittivity = dispersion_model.average_permittivity(
            #     dispersive_ranges_um[dispersive_range_idx]
            # )
            dispersive_max_permittivity = design.permittivity_constraints[-1]
            design.index_from_permittivity(dispersive_max_permittivity)

            # Convert device to permittivity and then index
            cur_permittivity = design.density_to_permittivity(
                cur_density_import_interp,
                design.permittivity_constraints[0],
                dispersive_max_permittivity,
            )
            # TODO: Another way to do this that can include dispersion is to update the
            # Scale filter with new permittivity constraint maximum
            cur_index = design.index_from_permittivity(cur_permittivity)

            # TODO (maybe): cut cur_index by region
            # Import to simulation
            self.import_nk_material(
                cur_index,
                design_region_import_x,
                design_region_import_y,
                design_region_import_z,
            )
            # Disable device index monitor to save memory
            self.disable(['design_index_monitor'])

            return cur_density, cur_permittivity
        return None

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
