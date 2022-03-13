import device as device             # Base class of SonyBayerFilter
import layering as layering         # Filters to impose fabrication constraints: See OPTICA paper supplement Section IIA, https://doi.org/10.1364/OPTICA.384228,  for details.
import sigmoid as sigmoid
import scale as scale

# import square_blur as square_blur
# import square_blur_smooth as square_blur_smooth

# import skimage.morphology as skim
# import networkx as nx
# import skimage.draw

from scipy import ndimage
import scipy.optimize

import numpy as np
import time
import sys

from SonyBayerFilterParameters import *

def compute_binarization( input_variable, set_point=0.5 ):
    total_shape = np.product( input_variable.shape )
    return ( 2. / total_shape ) * np.sum( np.sqrt( ( input_variable - set_point )**2 ) )

def compute_binarization_gradient( input_variable, set_point=0.5 ):
    total_shape = np.product( input_variable.shape )
    return ( 2. / total_shape ) * np.sign( input_variable - set_point )




class SonyBayerFilter(device.Device):

    def __init__(self, size, permittivity_bounds, init_permittivity, num_z_layers, design_height_voxels):
        super(SonyBayerFilter, self).__init__(size, permittivity_bounds, init_permittivity)

        self.num_z_layers = num_z_layers
        self.flip_threshold = 0.5
        self.minimum_design_value = 0
        self.maximum_design_value = 1
        self.current_iteration = 0
        self.design_height_voxels = design_height_voxels

        self.layer_height_voxels = self.design_height_voxels

        self.init_filters_and_variables()

        # set to the 0th epoch value
        self.update_filters( 0 )
        self.update_permittivity()

    #
    # Random initialization
    #
    def set_random_init( self, mean, std_dev ):
        '''Instead of uniform value for permittivity as an initialization, does a random distribution.'''
        self.w[0] = np.random.normal( mean, std_dev, self.w[0].shape )
        self.w[0] = np.maximum( np.minimum( self.w[0], 1.0 ), 0.0 )
        self.update_permittivity()

    #
    # Fixed initialization
    #
    def set_fixed_init( self, init_density ):
        '''Implements a different user-defined number for the initial permittivity, rather than the init_permittivity that was used to declare the device.'''
        self.w[0] = init_density * np.ones( self.w[0].shape )
        self.update_permittivity()

    #
    # Override the update_permittivity function so we can handle layer-dependent collapsing along either x- or y-dimensions
    #
    def update_permittivity(self):
        '''Hard-coded manual override of the update_permittivity function in device.py. Declares the specific filters to pass the design variable through.
        Here, we use 1) layering z0; 2) sigmoid_1; 3) scale_2 to get the final permittivity.'''
        var0 = self.w[ 0 ]

        var1 = self.layering_z_0.forward( var0 )
        self.w[ 1 ] = var1

        var2 = self.sigmoid_1.forward( var1 )
        self.w[ 2 ] = var2

        var3 = self.scale_2.forward( var2 )
        self.w[ 3 ] = var3


    def update_filters(self, epoch):
        '''Updates the second filter (sigmoid filter) with each epoch i.e. makes the sigmoid stronger.
        Then sets the filters of the SonyBayerFilter object to be:
        1) layering z0; 2) sigmoid_1; 3) scale_2
        The design_variable is then passed through these filters to get the final permittivity.'''
        
        self.sigmoid_beta = 0.0625 * (2**epoch)                                 # Sigmoid curve gets more strict with each epoch
        self.sigmoid_1 = sigmoid.Sigmoid(self.sigmoid_beta, self.sigmoid_eta)
        
        self.filters = [ self.layering_z_0, self.sigmoid_1, self.scale_2 ]


    def init_variables(self):
        '''Calls init_variables in device.py'''
        super(SonyBayerFilter, self).init_variables()

        self.w[0] = np.multiply(self.init_permittivity, np.ones(self.size, dtype=np.complex))


    def init_filters_and_variables(self):
        '''Creates filters for SonyBayerFilter object, and also the variables in device.py.'''
        self.num_filters = 3                             # The filters are defined below, and are hard-coded
        self.num_variables = 1 + self.num_filters        # One initial design_variable, passed through (num_filters) filters
        
        
        # Start the sigmoids at weak strengths
        self.sigmoid_beta = 0.0625
        self.sigmoid_eta = 0.5
        self.sigmoid_1 = sigmoid.Sigmoid(self.sigmoid_beta, self.sigmoid_eta)

        x_dimension_idx = 0
        y_dimension_idx = 1
        z_dimension_idx = 2

        z_voxel_layers = self.size[2]

        # Implement spacers between layers
        spacer_value = 0
        self.layering_z_0 = layering.Layering(z_dimension_idx, self.num_z_layers, variable_bounds=[0, 1], spacer_height_voxels=0, spacer_voxels_value=0)

        # Create scale filter with permittivity bounds as the min and max, to scale output of sigmoid filter to the physical refractive indices
        scale_min = self.permittivity_bounds[0]
        scale_max = self.permittivity_bounds[1]
        self.scale_2 = scale.Scale([scale_min, scale_max])

        # Initialize the filter chain
        self.filters = [ self.layering_z_0, self.sigmoid_1, self.scale_2 ]

        # Initializes default variables
        self.init_variables()
    
    #
    # Will also be overriding the backpropagate function in device.py
    #
    def backpropagate(self, gradient):
        gradient = self.scale_2.chain_rule(gradient, self.w[ 3 ], self.w[ 2 ])
        gradient = self.sigmoid_1.chain_rule(gradient, self.w[ 2 ], self.w[ 1 ])
        gradient = self.layering_z_0.chain_rule(gradient, self.w[ 1 ], self.w[ 0 ])

        return gradient

    # In the step function, we should update the permittivity with update_permittivity
    def step(self, gradient, step_size):
        self.w[0] = self.proposed_design_step(gradient, step_size)

        self.update_permittivity()

    def convert_to_binary_map(self, variable):
        '''Returns an array that is either the permittivity at that point, or the center of the permittivity bounds, whichever is greater.'''
        return np.greater(variable, self.mid_permittivity)