import os

import gdstk

#! todo: account for 3-materials.

class GDS:
    def __init__(self):
        self.lib = gdstk.Library()

    @classmethod
    def set_layers(cls, num_layers, unit=1e-6, precision=1e-9):
        """Initializes a GDS object with a certain number of layers. The library object will have the corresponding number of cells."""
        x = cls()
        x.num_layers = num_layers
        x.stl_mesh_list = [None] * x.num_layers
        x.gds_list = [None] * x.num_layers          # In case we need to contain a list of libraries instead of a single main library.
        x.lib.unit = unit
        x.lib.precision = precision
        return x

    # @classmethod
    # def from_stl_mesh(cls, base_stl_mesh):
    #     x = cls()
    #     x.base_stl_mesh = base_stl_mesh
    #     x.num_faces = len(base_stl_mesh)
    #     return x

    def assemble_device(self, stl_mesh_array, listed=False):
        """stl_mesh_array: list of Mesh objects generated from generate_stl() function in STL.py
        which converts a 3D inverse-designed device into a polygonal STL mesh. Here, each Mesh object
        should correspond to a single layer of the 3D device.

        This function takes those Mesh objects and converts them into a corresponding list of GDS objects.
        """
        if listed:
            for layer_idx, layer_mesh in enumerate(stl_mesh_array):
                # g = GDS().from_stl_mesh(layer_mesh)
                g = gdstk.Library()

                # Geometry must be placed in cells.
                cell = g.new_cell('FIRST')

                # Create the geometry (using single triangles as polygons) and add it to the cell.
                # Vectors returns a list of the vertex coordinates i.e. 3 points of 3D coordinates
                for v in layer_mesh.vectors:
                    d = v[:,:-1]                # Omit z-axis
                    f = [tuple(e) for e in d.astype(int)]
                    rect = gdstk.Polygon(f)
                    cell.add(rect)

                self.gds_list[layer_idx] = g
        else:
            for layer_idx, layer_mesh in enumerate(stl_mesh_array):
                cell = self.lib.new_cell(f'L{layer_idx}')

                # Create the geometry (using single triangles as polygons) and add it to the cell.
                # Vectors returns a list of the vertex coordinates i.e. 3 points of 3D coordinates
                for v in layer_mesh.vectors:
                    d = v[:,:-1]                # Omit z-axis
                    f = [tuple(e) for e in d.astype(int)]
                    rect = gdstk.Polygon(f)
                    cell.add(rect)

    def export_device(self, directory, filetype='gds', layer_idx=None):
        """Save the library in a GDSII or OASIS file. Optionally, save an image of the cell as SVG.
        Allows for individual layer selection and export.
        """
        if layer_idx is not None:
            g = gdstk.Library()
            cell = self.lib.cells[layer_idx]
            g.add(cell.copy(cell.name))
        else:
            g = self.lib

        file_name = 'device' if layer_idx is None else f'L{layer_idx}'

        if filetype in ['gds']:
            g.write_gds(os.path.join(directory, f'{file_name}.gds'))
        elif filetype in ['oasis']:
            g.write_oas(os.path.join(directory, f'{file_name}.oas'))
        elif filetype in ['svg']:
            if layer_idx is not None:
                # According to the above logic that means g is a library with a single cell.
                g.cells[0].write_svg(os.path.join(directory, f'{file_name}.svg'))
            else:
                for cell_idx, cell in enumerate(g.cells):
                    cell.write_svg(os.path.join(directory, f'L{cell_idx}.svg'))





class Layered_GDS:
    def __init__(self, num_layers):
        """[DEPRECATED]   This class instead contains a list of GDS objects, each with one cell corresponding to one layer. Useful for individual layer export."""
        self.num_layers = num_layers
        self.stl_mesh_list = [None] * self.num_layers
        self.stl_mesh = GDS()

    def assemble_device(self, stl_mesh_array, listed=True):
        """stl_mesh_array: list of Mesh objects generated from generate_stl() function in STL.py
        which converts a 3D inverse-designed device into a polygonal STL mesh. Here, each Mesh object
        should correspond to a single layer of the 3D device.

        This function takes those Mesh objects and converts them into a corresponding list of GDS objects.
        """
        if listed:
            for layer_idx, layer_mesh in enumerate(stl_mesh_array):
                g = GDS().from_stl_mesh(layer_mesh)

                # Geometry must be placed in cells.
                cell = g.lib.new_cell('FIRST')

                # Create the geometry (using single triangles as polygons) and add it to the cell.
                # Vectors returns a list of the vertex coordinates i.e. 3 points of 3D coordinates
                for v in layer_mesh.vectors:
                    d = v[:,:-1]                # Omit z-axis
                    f = [tuple(e) for e in d.astype(int)]
                    rect = gdstk.Polygon(f)
                    cell.add(rect)

                self.stl_mesh_list[layer_idx] = g
        else:
            for layer_idx, layer_mesh in enumerate(stl_mesh_array):
                cell = self.stl_mesh.lib.new_cell(f'L{layer_idx}')

                # Create the geometry (using single triangles as polygons) and add it to the cell.
                # Vectors returns a list of the vertex coordinates i.e. 3 points of 3D coordinates
                for v in layer_mesh.vectors:
                    d = v[:,:-1]                # Omit z-axis
                    f = [tuple(e) for e in d.astype(int)]
                    rect = gdstk.Polygon(f)
                    cell.add(rect)


    def export_layer(self, layer_idx, directory, filetype='gds'):
        """Save the library in a GDSII or OASIS file. Optionally, save an image of the cell as SVG."""
        g = self.stl_mesh_list[layer_idx]
        if filetype in ['gds']:
            g.lib.write_gds(os.path.join(directory, f'L{layer_idx}.gds'))
        elif filetype in ['oasis']:
            g.lib.write_oas(os.path.join(directory, f'L{layer_idx}.oas'))
        elif filetype in ['svg']:
            # NOTE: Only exports the first cell, although that should be all that's needed.
            g.lib.cells[0].write_svg(os.path.join(directory, f'L{layer_idx}.svg'))



