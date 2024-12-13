"""FDTD Simulation Interface Code."""

from __future__ import annotations

import abc
import os
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
        elif isinstance(o, np.generic):
            return o.item()
        if isinstance(o, complex) and np.imag(o)==0:
            # We purposely want it to break for actual complex numbers
            return np.real(o)
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
        vipdopt.logger.info(f'Successfully loaded {fname}\n')

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
        vipdopt.logger.debug(f'Successfully saved simulation to {fname}.\n')

    def as_dict(self) -> dict:
        """Return a dictionary representation of this simulation."""
        # We could use vars() here but as_dict() gives more customizability
        return {'info': {name: obj for name, obj in self.info.items()},
                'objects': {name: obj.as_dict() for name, obj in self.objects.items()}
                }

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

    # def get_path(self) -> Path:
    #     """Get the save path of the simulation."""
    #     p = self.info['path']
    #     if not isinstance(p, Path):
    #         p = Path(p)
    #     return p

    def copy(self) -> LumericalSimulation:
        """Return a copy of this simulation."""
        new_sim = LumericalSimulation()
        new_sim.info = self.info.copy()
        for obj_name, obj in self.objects.items():
            new_sim.new_object(obj_name, obj.obj_type, **obj.properties)

        try:    # Copy-paste the device imports from the previous simulation into the new simulation.
            for i, imp in enumerate(new_sim.imports()):
                imp.set_nk2(*self.imports()[i].get_nk2())
        except Exception:
            pass

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
                del new_sim.objects[name]
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
    
    def monitors_by_name(self, string_list) -> list[Monitor]:
        """Return a list of all monitor objects, with names filtered by a string."""
        if isinstance(string_list, str):
            string_list = [string_list]
        return [m for m in self.monitors() if any(substring in m.name for substring in string_list)]

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

    def indexmonitors(self) -> list[LumericalSimObject]:
        '''Return a list of all indexmonitor objects.'''
        return [obj for _, obj in self.objects.items() if obj.obj_type == LumericalSimObjectType.INDEX]

    def indexmonitor_names(self) -> Iterator[str]:
        """Return a list of all indexmonitor object names."""
        for obj in self.indexmonitors():
            yield obj.name
    
    def crosssection_monitors(self) -> list[LumericalSimObject]:
        '''Return a list of all cross-section monitor objects.'''
        return self.monitors_by_name('cross_monitor')
    
    def crosssection_monitor_names(self) -> Iterator[str]:
        """Return a list of all cross-section monitor object names."""
        for obj in self.crosssection_monitors():
            yield obj.name

    def import_field_shape(self) -> tuple[int, ...]:
        """Return the shape of the fields returned from this simulation's design index monitors."""
        index_prev = vipdopt.fdtd.getresult(
            list(self.indexmonitor_names())[0], 'index preview'
        )
        return np.squeeze(index_prev['index_x']).shape

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
    
    @classmethod
    def upload_device_index_to_sim(cls, 
                                   base_sim:LumericalSimulation,
                                   device,      # Device
                                   fdtd,        # LumericalFDTD
                                   simulator_dimension:str,
                                ):
        '''Takes a device and:
        - calculates its field shape;
        - performs interpolation to match the Lumerical mesh;
        - syncs the permittivity with the device's design variable.
        '''
        # Sync up base sim LumericalSimObject with FDTD in order to get device index monitor shape.
        fdtd.save(base_sim.get_path(), base_sim)
        # Reassign field shape now that the device has been properly imported into Lumerical.
        device.field_shape = base_sim.import_field_shape()
        # Handle 2D exception
        if simulator_dimension=='2D' and len(device.field_shape) == 2:
            device.field_shape += tuple([3])
        
        device.update_density()
        # Import device index now into base simulation
        import_primitives = base_sim.imports()
        # Hard-code reinterpolation size as this seems to be what works for accurate Lumerical imports.
        reinterpolation_size = (300,306,3) if simulator_dimension=='2D' else (300,300,306)

        for import_primitive in import_primitives:
            cur_density, cur_permittivity = device.import_cur_index(
                import_primitive,
                reinterpolation_factors=(1,1,1),    # For 2D the last entry of the tuple must always be 1.
                reinterpolation_size=reinterpolation_size,   # For 2D the last entry of the tuple must be 3.
                binarize=False,
            )

    def run_sims(self, program, sim_list, file_dir, add_job_to_fdtd=True):
        # program is either a LumericalOptimization or a LumericalEvaluation.
        
        self.save_to_fsp(program, sim_list, file_dir, add_job_to_fdtd=add_job_to_fdtd)
        vipdopt.logger.info(
            'In-Progress Step 1: All Simulations Setup and Jobs Added.'
        )
        self.run_jobs_to_completion(program, sim_list)
    
    @classmethod
    def save_to_fsp(cls, program, sim_list, file_dir, add_job_to_fdtd=True):
        # program is either a LumericalOptimization or a LumericalEvaluation
        for sim in sim_list:
            sim_file = program.dirs['eval_temp'] / f'{sim.info["name"]}.fsp'
            # sim.link_monitors()

            program.fdtd.save(sim_file, sim)  # Saving also sets the path
            if add_job_to_fdtd:
                program.fdtd.addjob(sim_file)
    
    @classmethod
    def run_jobs_to_completion(cls, program, sim_list):
        # program is either a LumericalOptimization or a LumericalEvaluation
        
        # If true, we're in debugging mode and it means no simulations are run.
        # Data is instead pulled from finished simulation files in the debug folder.
        # If false, run jobs and check that they all ran to completion.
        if program.cfg['pull_sim_files_from_debug_folder']:
            for sim in sim_list:
                sim_file = (
                    program.dirs['debug_completed_jobs']
                    / f'{sim.info["name"]}.fsp'
                )
                sim.set_path(sim_file)
        else:
            while program.fdtd.fdtd.listjobs(
                'FDTD'
            ):  # Existing job list still occupied
                # Run simulations from existing job list
                use_GUI_license = program.cfg['use_GUI_license'] if os.getenv('SLURM_JOB_NODELIST') is None else False
                program.fdtd.runjobs( use_GUI_license )

                # Check if there are any jobs that didn't run
                for sim in sim_list:
                    sim_file = sim.get_path()
                    # program.fdtd.load(sim_file)
                    # if program.fdtd.layoutmode():
                    cond = [program.cfg['simulator_dimension'], sim_file.stat().st_size]
                    if (cond[0]=='3D' and cond[1]<=2e7) or (cond[0]=='2D' and cond[1]<=5e5):
                        # Arbitrary 500KB filesize for 2D sims, 20MB filesize for 3D sims. That didn't run completely
                        program.fdtd.addjob(sim_file)
                        vipdopt.logger.info(
                            f'Failed to run: {sim_file.name}. Re-adding ...'
                        )

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

    with LumericalSimulation(Path(args.simulation_json)) as sim:
        # with LumericalSimulation() as sim:
        sim.fdtd.promise_env_setup()
        sim.load('test_project/.tmp/sim_0.fsp')
        sys.exit()

        vipdopt.logger.info('Saving Simulation')
        sim.save(Path('test_sim'))
        vipdopt.logger.info('Running simulation')
        sim.run()
