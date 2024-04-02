import os
import sys

import numpy as np
from stl import mesh

sys.path.append(os.getcwd())
current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)


def viz_stl(mesh):
    # Optionally render the rotated cube faces
    import matplotlib as mpl

    mpl.use('TKAgg')
    from matplotlib import pyplot as plt
    from mpl_toolkits import mplot3d

    # Create a new plot
    figure = plt.figure()
    axes = figure.add_subplot(projection='3d')

    # Render the cube
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh.vectors))

    # Auto scale to the mesh size
    scale = mesh.points.flatten()
    axes.auto_scale_xyz(scale, scale, scale)

    # Show the plot to the screen
    plt.show()


# Define the 8 vertices of the cube
vertices = np.array([
    [-1, -1, -1],
    [+1, -1, -1],
    [+1, +1, -1],
    [-1, +1, -1],
    [-1, -1, +1],
    [+1, -1, +1],
    [+1, +1, +1],
    [-1, +1, +1],
])
# Define the 12 triangles composing the cube
faces = np.array([
    [0, 3, 1],
    [1, 3, 2],
    [0, 4, 7],
    [0, 7, 3],
    [4, 5, 6],
    [4, 6, 7],
    [5, 1, 2],
    [5, 2, 6],
    [2, 3, 6],
    [3, 7, 6],
    [0, 1, 5],
    [0, 5, 4],
])

# Create the mesh
cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
print(3)
for i, f in enumerate(faces):
    for j in range(3):
        cube.vectors[i][j] = vertices[f[j], :]

# # Write the mesh to file "cube.stl"
# viz_stl()
# cube.save('cube.stl')

# Write to GDS
import gdstk

from vipdopt import GDS

g = GDS.GDS().from_stl_mesh(cube)

# # The GDSII file is called a library, which contains multiple cells.
# lib = gdstk.Library()

# Geometry must be placed in cells.
cell = g.lib.new_cell('FIRST')

# Create the geometry (using single triangles as polygons) and add it to the cell.
for v in cube.vectors:
    d = v[:, :-1]
    f = [tuple(e) for e in d.astype(int)]
    rect = gdstk.Polygon(f)
    cell.add(rect)

# Save the library in a GDSII or OASIS file.
g.lib.write_gds(os.path.join(current_dir, 'first.gds'))

# Optionally, save an image of the cell as SVG.
cell.write_svg(os.path.join(current_dir, 'first.svg'))


print(3)
