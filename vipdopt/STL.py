import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from stl import mesh


class STL:
	def __init__( self, density_3d ):
		self.density_3d = density_3d
		self.density_shape = density_3d.shape


	def generate_stl( self ):
		pad_density = np.pad( self.density_3d, ( ( 1, 1 ), ( 1, 1 ), ( 1, 1 ) ), mode='constant' )

		triangles = []
		for z in range( 1, 1 + self.density_shape[ 2 ] ):
			for y in range( 1, 1 + self.density_shape[ 1 ] ):
				for x in range( 1, 1 + self.density_shape[ 0 ] ):

					if pad_density[ x, y, z ]:

						for neighbor_x in [ -1, 1 ]:
							if not pad_density[ x + neighbor_x, y, z ]:

								v2_x = x + 0.5 * neighbor_x
								v2_y = y - 0.5
								v2_z = z - 0.5

								v1_x = x + 0.5 * neighbor_x
								v1_y = y + 0.5
								v1_z = z - 0.5

								v0_x = x + 0.5 * neighbor_x
								v0_y = y + 0.5
								v0_z = z + 0.5


								v5_x = x + 0.5 * neighbor_x
								v5_y = y - 0.5
								v5_z = z - 0.5

								v4_x = x + 0.5 * neighbor_x
								v4_y = y + 0.5
								v4_z = z + 0.5

								v3_x = x + 0.5 * neighbor_x
								v3_y = y - 0.5
								v3_z = z + 0.5

								if neighbor_x == -1:
									triangles.append( [ [ v0_x, v0_y, v0_z ], [ v1_x, v1_y, v1_z ], [ v2_x, v2_y, v2_z ] ] )
									triangles.append( [ [ v3_x, v3_y, v3_z ], [ v4_x, v4_y, v4_z ], [ v5_x, v5_y, v5_z ] ] )
								else:
									triangles.append( [ [ v2_x, v2_y, v2_z ], [ v1_x, v1_y, v1_z ], [ v0_x, v0_y, v0_z ] ] )
									triangles.append( [ [ v5_x, v5_y, v5_z ], [ v4_x, v4_y, v4_z ], [ v3_x, v3_y, v3_z ] ] )


						for neighbor_y in [ -1, 1 ]:
							if not pad_density[ x, y + neighbor_y, z ]:

								v0_y = y + 0.5 * neighbor_y
								v0_z = z - 0.5
								v0_x = x - 0.5

								v1_y = y + 0.5 * neighbor_y
								v1_z = z + 0.5
								v1_x = x - 0.5

								v2_y = y + 0.5 * neighbor_y
								v2_z = z + 0.5
								v2_x = x + 0.5


								v3_y = y + 0.5 * neighbor_y
								v3_z = z - 0.5
								v3_x = x + 0.5

								v4_y = y + 0.5 * neighbor_y
								v4_z = z - 0.5
								v4_x = x - 0.5

								v5_y = y + 0.5 * neighbor_y
								v5_z = z + 0.5
								v5_x = x + 0.5


								if neighbor_y == -1:
									triangles.append( [ [ v2_x, v2_y, v2_z ], [ v1_x, v1_y, v1_z ], [ v0_x, v0_y, v0_z ] ] )
									triangles.append( [ [ v5_x, v5_y, v5_z ], [ v4_x, v4_y, v4_z ], [ v3_x, v3_y, v3_z ] ] )
								else:
									triangles.append( [ [ v0_x, v0_y, v0_z ], [ v1_x, v1_y, v1_z ], [ v2_x, v2_y, v2_z ] ] )
									triangles.append( [ [ v3_x, v3_y, v3_z ], [ v4_x, v4_y, v4_z ], [ v5_x, v5_y, v5_z ] ] )


						for neighbor_z in [ -1, 1 ]:
							if not pad_density[ x, y, z + neighbor_z ]:

								v2_z = z + 0.5 * neighbor_z
								v2_x = x - 0.5
								v2_y = y - 0.5

								v1_z = z + 0.5 * neighbor_z
								v1_x = x + 0.5
								v1_y = y - 0.5

								v0_z = z + 0.5 * neighbor_z
								v0_x = x + 0.5
								v0_y = y + 0.5

								v5_z = z + 0.5 * neighbor_z
								v5_x = x + 0.5
								v5_y = y + 0.5

								v4_z = z + 0.5 * neighbor_z
								v4_x = x - 0.5
								v4_y = y + 0.5

								v3_z = z + 0.5 * neighbor_z
								v3_x = x - 0.5
								v3_y = y - 0.5

								if neighbor_z == -1:
									triangles.append( [ [ v0_x, v0_y, v0_z ], [ v1_x, v1_y, v1_z ], [ v2_x, v2_y, v2_z ] ] )
									triangles.append( [ [ v3_x, v3_y, v3_z ], [ v4_x, v4_y, v4_z ], [ v5_x, v5_y, v5_z ] ] )
								else:
									triangles.append( [ [ v2_x, v2_y, v2_z ], [ v1_x, v1_y, v1_z ], [ v0_x, v0_y, v0_z ] ] )
									triangles.append( [ [ v5_x, v5_y, v5_z ], [ v4_x, v4_y, v4_z ], [ v3_x, v3_y, v3_z ] ] )


		def z_compare(triangle):
			return np.min([triangle[0][2], triangle[1][2], triangle[2][2]])
		sorted_triangles = sorted( triangles, key=z_compare )

		num_triangles = len( sorted_triangles )
		mesh_data = np.zeros( num_triangles, dtype=mesh.Mesh.dtype )

		for triangle_idx in range( num_triangles ):
			mesh_data[ 'vectors' ][ triangle_idx ] = np.array( sorted_triangles[ triangle_idx ] )

		self.stl_mesh = mesh.Mesh( mesh_data, remove_empty_areas=False )


	def save_stl( self, filename ):
		self.stl_mesh.save( filename )

	def viz_stl( self ):
		figure = plt.figure()
		axes = mplot3d.Axes3D( figure )

		axes.add_collection3d( mplot3d.art3d.Poly3DCollection( self.stl_mesh.vectors ) )

		scale = self.stl_mesh.points.flatten()
		axes.auto_scale_xyz(scale, scale, scale)

		plt.show()




class Layered_STL:
	def __init__( self, density_3d, layer_increment ):
		self.density_3d = density_3d
		self.density_shape = density_3d.shape
		self.layer_increment = layer_increment

		assert ( self.density_3d.shape[ 2 ] % self.layer_increment ) == 0, 'This density does not divide equally into the given layer increment!'

		self.num_layers = self.density_3d.shape[ 2 ] // self.layer_increment


	def generate_stl( self ):
		pad_density = np.pad( self.density_3d, ( ( 1, 1 ), ( 1, 1 ), ( self.layer_increment, self.layer_increment ) ), mode='constant' )

		triangles = []
		for layered_z in range( self.num_layers ):

			z = ( 1 + layered_z ) * self.layer_increment

			for y in range( 1, 1 + self.density_shape[ 1 ] ):
				for x in range( 1, 1 + self.density_shape[ 0 ] ):

					if pad_density[ x, y, z ]:

						for neighbor_x in [ -1, 1 ]:
							if not pad_density[ x + neighbor_x, y, z ]:

								v2_x = x + 0.5 * neighbor_x
								v2_y = y - 0.5
								v2_z = z - 0.5 * self.layer_increment

								v1_x = x + 0.5 * neighbor_x
								v1_y = y + 0.5
								v1_z = z - 0.5 * self.layer_increment

								v0_x = x + 0.5 * neighbor_x
								v0_y = y + 0.5
								v0_z = z + 0.5 * self.layer_increment


								v5_x = x + 0.5 * neighbor_x
								v5_y = y - 0.5
								v5_z = z - 0.5 * self.layer_increment

								v4_x = x + 0.5 * neighbor_x
								v4_y = y + 0.5
								v4_z = z + 0.5 * self.layer_increment

								v3_x = x + 0.5 * neighbor_x
								v3_y = y - 0.5
								v3_z = z + 0.5 * self.layer_increment

								if neighbor_x == -1:
									triangles.append( [ [ v0_x, v0_y, v0_z ], [ v1_x, v1_y, v1_z ], [ v2_x, v2_y, v2_z ] ] )
									triangles.append( [ [ v3_x, v3_y, v3_z ], [ v4_x, v4_y, v4_z ], [ v5_x, v5_y, v5_z ] ] )
								else:
									triangles.append( [ [ v2_x, v2_y, v2_z ], [ v1_x, v1_y, v1_z ], [ v0_x, v0_y, v0_z ] ] )
									triangles.append( [ [ v5_x, v5_y, v5_z ], [ v4_x, v4_y, v4_z ], [ v3_x, v3_y, v3_z ] ] )


						for neighbor_y in [ -1, 1 ]:
							if not pad_density[ x, y + neighbor_y, z ]:

								v0_y = y + 0.5 * neighbor_y
								v0_z = z - 0.5 * self.layer_increment
								v0_x = x - 0.5

								v1_y = y + 0.5 * neighbor_y
								v1_z = z + 0.5 * self.layer_increment
								v1_x = x - 0.5

								v2_y = y + 0.5 * neighbor_y
								v2_z = z + 0.5 * self.layer_increment
								v2_x = x + 0.5


								v3_y = y + 0.5 * neighbor_y
								v3_z = z - 0.5 * self.layer_increment
								v3_x = x + 0.5

								v4_y = y + 0.5 * neighbor_y
								v4_z = z - 0.5 * self.layer_increment
								v4_x = x - 0.5

								v5_y = y + 0.5 * neighbor_y
								v5_z = z + 0.5 * self.layer_increment
								v5_x = x + 0.5


								if neighbor_y == -1:
									triangles.append( [ [ v2_x, v2_y, v2_z ], [ v1_x, v1_y, v1_z ], [ v0_x, v0_y, v0_z ] ] )
									triangles.append( [ [ v5_x, v5_y, v5_z ], [ v4_x, v4_y, v4_z ], [ v3_x, v3_y, v3_z ] ] )
								else:
									triangles.append( [ [ v0_x, v0_y, v0_z ], [ v1_x, v1_y, v1_z ], [ v2_x, v2_y, v2_z ] ] )
									triangles.append( [ [ v3_x, v3_y, v3_z ], [ v4_x, v4_y, v4_z ], [ v5_x, v5_y, v5_z ] ] )


						for neighbor_z in [ -self.layer_increment, self.layer_increment ]:
							if not pad_density[ x, y, z + neighbor_z ]:

								v2_z = z + 0.5 * neighbor_z
								v2_x = x - 0.5
								v2_y = y - 0.5

								v1_z = z + 0.5 * neighbor_z
								v1_x = x + 0.5
								v1_y = y - 0.5

								v0_z = z + 0.5 * neighbor_z
								v0_x = x + 0.5
								v0_y = y + 0.5

								v5_z = z + 0.5 * neighbor_z
								v5_x = x + 0.5
								v5_y = y + 0.5

								v4_z = z + 0.5 * neighbor_z
								v4_x = x - 0.5
								v4_y = y + 0.5

								v3_z = z + 0.5 * neighbor_z
								v3_x = x - 0.5
								v3_y = y - 0.5

								if neighbor_z == -self.layer_increment:
									triangles.append( [ [ v0_x, v0_y, v0_z ], [ v1_x, v1_y, v1_z ], [ v2_x, v2_y, v2_z ] ] )
									triangles.append( [ [ v3_x, v3_y, v3_z ], [ v4_x, v4_y, v4_z ], [ v5_x, v5_y, v5_z ] ] )
								else:
									triangles.append( [ [ v2_x, v2_y, v2_z ], [ v1_x, v1_y, v1_z ], [ v0_x, v0_y, v0_z ] ] )
									triangles.append( [ [ v5_x, v5_y, v5_z ], [ v4_x, v4_y, v4_z ], [ v3_x, v3_y, v3_z ] ] )


		def z_compare(triangle):
			return np.min([triangle[0][2], triangle[1][2], triangle[2][2]])
		sorted_triangles = sorted( triangles, key=z_compare )

		num_triangles = len( sorted_triangles )
		mesh_data = np.zeros( num_triangles, dtype=mesh.Mesh.dtype )

		for triangle_idx in range( num_triangles ):
			mesh_data[ 'vectors' ][ triangle_idx ] = np.array( sorted_triangles[ triangle_idx ] )

		self.stl_mesh = mesh.Mesh( mesh_data, remove_empty_areas=False )


	def save_stl( self, filename ):
		self.stl_mesh.save( filename )

	def viz_stl( self ):
		figure = plt.figure()
		axes = mplot3d.Axes3D( figure )

		axes.add_collection3d( mplot3d.art3d.Poly3DCollection( self.stl_mesh.vectors ) )

		scale = self.stl_mesh.points.flatten()
		axes.auto_scale_xyz(scale, scale, scale)

		plt.show()
