import pytest

from testing import assert_equal
from vipdopt.simulation import (
    LumericalSimObject,
    LumericalSimObjectType,
    LumericalSimulation,
)

# TODO: How to test if I need lumerical? Porbably use a marker and ignore or mark xfail
# TODO: I should MOCK lumapi! see
# TODO:https://stackoverflow.com/questions/43162722/mocking-a-module-import-in-pytest


@pytest.mark.smoke()
@pytest.mark.lumapi()
def test_load_sim(simulation_json):
    s = LumericalSimulation(fname=simulation_json)

    source_aperture = LumericalSimObject('source_aperture', LumericalSimObjectType.rect)
    source_aperture.update({
        'x': 0,
        'x span': 1.5e-6,
        'y': 0,
        'y span': 1.5e-6,
        'index': 0.3e-6,
    })

    device_mesh = LumericalSimObject('device_mesh', LumericalSimObjectType.mesh)
    device_mesh.update({
        'x': 0,
        'x span': 1.5e-6,
        'y': 0,
        'y span': 1.5e-6,
    })


    assert 'source_aperture' in s.objects
    assert_equal(s.objects['source_aperture'], source_aperture)
    assert 'device_mesh' in s.objects
    assert_equal(s.objects['device_mesh'], device_mesh)


@pytest.mark.smoke()
@pytest.mark.lumapi()
def test_save_sim(tmp_path, mock_sim_file):
    path = tmp_path / 'sim.json'

    source_aperture = LumericalSimObject('source_aperture', LumericalSimObjectType.rect)
    source_aperture.update({
        'x': 0,
        'x span': 1.5e-6,
        'y': 0,
        'y span': 1.5e-6,
        'index': 0.3e-6,
    })

    device_mesh = LumericalSimObject('device_mesh', LumericalSimObjectType.mesh)
    device_mesh.update({
        'x': 0,
        'x span': 1.5e-6,
        'y': 0,
        'y span': 1.5e-6,
    })

    with LumericalSimulation(fname=None) as s:
        s.add_object(source_aperture)
        s.add_object(device_mesh)
        s.save(path)

    assert len(list(tmp_path.iterdir())) == 1  # Only one file written
    assert_equal(path.read_text(), mock_sim_file)  # Correct contents


@pytest.mark.smoke()
@pytest.mark.lumapi()
def test_new_object():
    s = LumericalSimulation(fname=None)
    props = {
        'x': 0,
        'x span': 1.5e-6,
        'y': 0,
        'y span': 1.5e-6,
    }
    s.new_object('device_mesh', LumericalSimObjectType.mesh, properties=props)

    device_mesh = LumericalSimObject('device_mesh', LumericalSimObjectType.mesh)
    device_mesh.update({
        'x': 0,
        'x span': 1.5e-6,
        'y': 0,
        'y span': 1.5e-6,
    })

    assert 'device_mesh' in s.objects
    assert_equal(s.objects['device_mesh'], device_mesh)
