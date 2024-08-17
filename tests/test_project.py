"""Tests for project.py"""

import pytest

from testing import assert_equal
from vipdopt.optimization import Device, FoM
from vipdopt.project import Project


@pytest.mark.xfail()
@pytest.mark.usefixtures('_mock_project_json')
def test_load(device_dict: dict, fom_dict: dict):
    project = Project()
    project.load_project('fakefile.json')

    assert_equal(project.device, Device.from_source(device_dict))
    src_to_sim_map = {
        src: project.base_sim.with_enabled([src])
        for src in project.base_sim.source_names()
    }

    foms = [
        FoM.from_dict(name, data, src_to_sim_map) for name, data in fom_dict.items()
    ]
    assert_equal(project.foms[0], foms[0])

# TODO: Write more tests e.g. the below:
# def save_and_load():
#     project_dir = Path('./test_project/')
#     output_dir = Path('./test_output_optimization/')

#     project = Project()
#     project.load_project(project_dir, config_name='test_config.yml')
#     #output_dir = project.subdirectories['checkpoints']
#     project.save_as(output_dir)
#     # Test that the saved format is loadable
#     project2 = Project.from_dir(output_dir)
