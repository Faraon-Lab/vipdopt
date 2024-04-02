import os
import sys
import time
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use('TKAgg')

sys.path.append(os.getcwd())
current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)
import vipdopt
from vipdopt import GDS, STL
from vipdopt.utils import import_lumapi


def binarize(variable_in):
    """Assumes density - if not, convert explicitly."""
    return 1.0 * np.greater_equal(variable_in, 0.5)


current_dir = Path(current_dir)
lumapi_filepath_local = 'C:\\Program Files\\Lumerical\\v212\\api\\python\\lumapi.py'
vipdopt.lumapi = import_lumapi(lumapi_filepath_local)

full_density = np.load(current_dir / 'cur_design.npy')
full_density = binarize(np.real(full_density))
# plt.imshow(np.real(full_density[...,20]))
# plt.show()

# Convert to STL - maybe a path to generate GDSii from here ?
stl_generator = STL.STL(full_density)
stl_generator.generate_stl()
# stl_generator.viz_stl()
stl_generator.save_stl(current_dir / 'final_device.stl')

# Find out how Lumerical exports GDSii data - mirror that
# And see if the final GDSii can be imported back in to achieve the same effect

# numpy to GDSii, process each layer accordingly
# It has to be imported back in layer by layer unfortunately.
# And then also the material set layer by layer.

# Create a list of Mesh objects that will be fed into the GDS code.
# Each object is a polygonal mesh / 2D layer of the full, 3D density.
layer_mesh_array = []
for z_layer in range(full_density.shape[2]):
    density_layer = full_density[..., z_layer][..., np.newaxis]

    # Convert to STL - maybe a path to generate GDSii from here ?
    stl_generator = STL.STL(density_layer)
    stl_generator.generate_stl()
    layer_mesh = stl_generator.stl_mesh

    layer_mesh_array.append(layer_mesh)

# # Create a Layered GDS object that holds a list of GDS objects converted from each Mesh.
# Create a GDS object that contains a Library with Cells corresponding to each 2D layer in the 3D device.
gds_generator = GDS.GDS().set_layers(
    full_density.shape[2], unit=(2.04e-6) / (40e-6) * (1e-6)
)  # , unit=0.051e-9)
gds_generator.assemble_device(layer_mesh_array, listed=False)

# Directory for export
gds_layer_dir = os.path.join(current_dir, 'layers')
if not os.path.exists(gds_layer_dir):
    os.makedirs(gds_layer_dir)
# Export both GDS and SVG for easy visualization.
gds_generator.export_device(gds_layer_dir, filetype='gds')
gds_generator.export_device(gds_layer_dir, filetype='svg')
# for layer_idx in range(0, full_density.shape[2]):
#     gds_generator.export_device(gds_layer_dir, filetype='gds', layer_idx=layer_idx)
#     gds_generator.export_device(gds_layer_dir, filetype='svg', layer_idx=layer_idx)


# Now we have to test it.
# Open up Lumerical
# Delete the existing device import
# Call gdsimport on the device.gds, iterating through each cell.
# For each cell, specify material, z, and z-span.

fdtd = vipdopt.lumapi.FDTD(hide=False)
fdtd.newproject()
fdtd.load(os.path.join(current_dir, 'forward_src_x'))
dev_name = 'design_import'
num_voxel_side = [41, 41, 42]
fdtd.setnamed(dev_name, 'enabled', 0)
x_vals = np.linspace(
    fdtd.getnamed(dev_name, 'x min'),
    fdtd.getnamed(dev_name, 'x max'),
    num_voxel_side[0],
)
y_vals = np.linspace(
    fdtd.getnamed(dev_name, 'y min'),
    fdtd.getnamed(dev_name, 'y max'),
    num_voxel_side[1],
)
z_vals = np.linspace(
    fdtd.getnamed(dev_name, 'z min'),
    fdtd.getnamed(dev_name, 'z max'),
    num_voxel_side[2],
)

for layer_idx in range(num_voxel_side[2]):
    t = time.time()
    # fdtd.gdsimport(os.path.join(gds_layer_dir, 'device.gds'), f'L{layer_idx}', 0)
    fdtd.gdsimport(
        os.path.join(gds_layer_dir, 'device.gds'),
        f'L{layer_idx}',
        0,
        'Si (Silicon) - Palik',
        z_vals[layer_idx],
        z_vals[layer_idx + 1],
    )
    fdtd.set({'x': -1.02e-6, 'y': -1.02e-6})
    print(f'Layer {layer_idx} imported in {time.time() - t} seconds.')


print('End of code reached.')
