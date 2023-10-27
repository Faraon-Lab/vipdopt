"""Library of code for working with Jinja and rendering templates."""

from jinja2 import Environment, FileSystemLoader


if __name__ == '__main__':
    env = Environment(loader=FileSystemLoader('jinja_templates/'))

    template = env.get_template('mesh_properties.jinja')
    print(template.render(
        min_feature_size_um=1,
        geometry_spacing_lateral_um=2,
    ))
