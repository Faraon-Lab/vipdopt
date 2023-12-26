"""Tests for vipdopt.configuration.template"""


import numpy as np
import pytest
import yaml

from testing import assert_close
from vipdopt.configuration.template import TemplateRenderer
from vipdopt.utils import read_config_file

TEST_TEMPLATE_FILE = 'derived_simulation_properties.j2'


@pytest.mark.smoke()
def rest_render():
    pass

@pytest.mark.usefixtures('_mock_example_config')
def test_render_example_config(
    template_renderer: TemplateRenderer,
    example_derived_properties,
):
    template_renderer.set_template(TEST_TEMPLATE_FILE)

    data = read_config_file('fakefile.yaml')
    data['pi'] = np.pi
    output = template_renderer.render(
        data=data,
        pi=np.pi,
        trim_blocks=True,
        lstrip_blocks=True,
    )

    rendered_data = yaml.safe_load(output)

    for k, v in example_derived_properties.items():
        assert k in rendered_data
        assert_close(rendered_data[k], v)



