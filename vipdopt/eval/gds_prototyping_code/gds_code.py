import numpy as np
from spins.gds import gen_gds

new_pattern = np.copy( new_pattern ).repeat( upsample_factor, axis=0 ).repeat( upsample_factor, axis=1 ) # Upsample for accurate renderå


Nx = new_pattern.shape[ 0 ]
Ny = new_pattern.shape[ 1 ]
spatial_um = np.linspace( -0.5 * lattice_period_um * num_replications, 0.5 * lattice_period_um * num_replications, num_replications * Nx )

for x_rep in range( num_replications ):
    for y_rep in range( num_replications ):
        x_start = x_rep * Nx
        y_start = y_rep * Ny

        if negative:
            new_pattern[ x_start : ( x_start + Nx ), y_start : ( y_start + Ny ) ] = np.array( ~np.greater( binarize_density.repeat( upsample_factor, axis=0 ).repeat( upsample_factor, axis=1 ), 255 / 2 ) )
        else:
            new_pattern[ x_start : ( x_start + Nx ), y_start : ( y_start + Ny ) ] = binarize_density.repeat( upsample_factor, axis=0 ).repeat( upsample_factor, axis=1 )

#plt.figure()
#plt.imshow(new_pattern)
#plt.show()
#sys.exit()

contours, hierarchy = cv2.findContours( new_pattern, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )

reshape =  []

for idx in range( len( contours ) ):
    make_nd = np.zeros( ( len( contours[ idx ] ), 2 ) )

    for pt_idx in range( len( contours[ idx ] ) ):
        make_nd[ pt_idx, : ] = [ 1000. * spatial_um[ contours[ idx ][ pt_idx ][ 0 ][ 0 ] ], 1000. * spatial_um[ contours[ idx ][ pt_idx ][ 0 ][ 1 ] ] ]

    reshape.append( make_nd )

if negative:
    gen_gds( reshape, override_data_folder + '/inv_design_n.gds' )
else:
    gen_gds( reshape, override_data_folder + '/inv_design_p.gds' )

num_cell_replications = 1

if negative:
    lib = gdspy.GdsLibrary( infile=( override_data_folder + '/inv_design_n.gds' ) )
else:
    lib = gdspy.GdsLibrary( infile=( override_data_folder + '/inv_design_p.gds' ) )
main_cell = lib.top_level()[ 0 ]

lib_replicate = gdspy.GdsLibrary()
inverse_design = lib_replicate.new_cell( 'inverse_design' )
inverse_design.add( gdspy.CellArray(
        main_cell, num_cell_replications, num_cell_replications, (lattice_period_um, lattice_period_um),
        origin=(-0.5 * lattice_period_um * num_cell_replications, -0.5 * lattice_period_um * num_cell_replications)
    ) )


lib_replicate.add( inverse_design )

if negative:
    lib_replicate.write_gds( override_data_folder + '/inv_design_replicate_n.gds' )
else:
    lib_replicate.write_gds( override_data_folder + '/inv_design_replicate_p.gds' )
