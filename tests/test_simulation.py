"""Tests for vipdopt.simulation"""

import pytest
import json

from testing import assert_equal
from vipdopt.simulation import Simulation
from vipdopt.simulation.simobject import SimObject, SimObjectType

# TODO: How to test if I need Lumerical? Probably use a marker and ignore or mark xfail
# TODO: I should MOCK lumapi! see
# TODO:https://stackoverflow.com/questions/43162722/mocking-a-module-import-in-pytest

@pytest.mark.lumapi()
def test_load_sim(simulation_json):
    source_aperture = SimObject('source_aperture', SimObjectType.RECT)
    source_aperture.update(**{
        'name': 'source_aperture',
        'x': 0,
        'x span': 1.5e-6,
        'y': 0,
        'y span': 1.5e-6,
        'index': 0.3e-6,
    })

    device_mesh = SimObject('device_mesh', SimObjectType.MESH)
    device_mesh.update(**{
        'name': 'device_mesh',
        'x': 0,
        'x span': 1.5e-6,
        'y': 0,
        'y span': 1.5e-6,
    })

    s = Simulation(source=simulation_json)
    assert 'source_aperture' in s.objects
    assert_equal(s.objects['source_aperture'], source_aperture)
    assert 'device_mesh' in s.objects
    assert_equal(s.objects['device_mesh'], device_mesh)



@pytest.mark.lumapi()
def test_save_sim(tmp_path, sim_file):
    path = tmp_path / 'sim.json'

    source_aperture = SimObject('source_aperture', SimObjectType.RECT)
    source_aperture.update(**{
        'name': 'source_aperture',
        'x': 0,
        'x span': 1.5e-6,
        'y': 0,
        'y span': 1.5e-6,
        'index': 0.3e-6,
    })

    device_mesh = SimObject('device_mesh', SimObjectType.MESH)
    device_mesh.update(**{
        'name': 'device_mesh',
        'x': 0,
        'x span': 1.5e-6,
        'y': 0,
        'y span': 1.5e-6,
    })

    s = Simulation(source=None)
    s.add_object(source_aperture)
    s.add_object(device_mesh)
    s.save(path)

    # Nia 20241003 ==================
    # assert len(list(tmp_path.iterdir())) == 2  # noqa: PLR2004 Only two files written
    # assert_equal(path.read_text(), sim_file)  # Correct contents

    # Ian 20250403 ==================
    assert len(list(tmp_path.iterdir())) == 1

    j = json.loads(path.read_text())
    j.pop('info')                  # This is created and added on top of the input JSON.
    j_txt = json.dumps(j, indent=4, ensure_ascii=True, cls=s.encoder)
    # input args to json.dumps() are taken from s.as_json()
    
    assert_equal(j_txt, sim_file)  # Correct contents
    #! BEWARE OF COMMAS, EXTRA SPACES, AND \N
    # for instance, if you put a comma after the last element of a dictionary, this will break.


@pytest.mark.lumapi()
def test_new_object():
    props = {
        'name': 'device_mesh',
        'x': 0,
        'x span': 1.5e-6,
        'y': 0,
        'y span': 1.5e-6,
    }

    correct_mesh = SimObject('correct_mesh', SimObjectType.MESH)
    correct_mesh.update(**props)

    s = Simulation(source=None)
    s.new_object('device_mesh', SimObjectType.MESH, **props)

    assert 'device_mesh' in s.objects
    assert_equal(s.objects['device_mesh'], correct_mesh)