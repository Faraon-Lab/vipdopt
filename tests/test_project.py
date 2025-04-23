"""Tests for project.py"""

import pytest
from pathlib import Path

from testing import assert_equal, assert_close
from vipdopt.optimization import Device, FoM
from vipdopt.project import Project

# @pytest.mark.xfail()
# def test_save(device_dict)


# @pytest.mark.xfail()
# @pytest.mark.usefixtures('_mock_project_json')
# def test_load(mock_device_dict: dict, fom_dict: dict):
#     project = Project()
#     project.load_project('fakefile.json')

#     assert_equal(project.device, Device.from_source(mock_device_dict))
#     src_to_sim_map = {
#         src: project.base_sim.with_enabled([src])
#         for src in project.base_sim.source_names()
#     }

#     foms = [
#         FoM.from_dict(name, data, src_to_sim_map) for name, data in fom_dict.items()
#     ]
#     assert_equal(project.foms[0], foms[0])

@pytest.mark.xfail()
def test_save_and_load():
    project_dir = Path('./testing/test_project/')
    output_dir = Path('./testing/test_output_project/')
    # output_dir = project.subdirectories['checkpoints']

    project1 = Project()
    project1.load_project(project_dir, config_name='test_config.json')
    project1.save_as(output_dir)
    # Test that the saved format is loadable
    project2 = Project.from_dir(output_dir)
    
    for attr in vars(project1):
        assert_close(project2.__getattribute__(attr), project1.__getattribute__(attr))

# def test_entire_optimization():
        
#     args = {
#         'verbose': False,
#         'log': Path('dev.log'),
#         'command': 'optimize',
#         'directory': Path('runs/test_run'),
#         'config': 'processed_config.yml'
#     }

