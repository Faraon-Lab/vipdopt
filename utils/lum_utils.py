# Copyright © 2023, California Institute of Technology. All rights reserved.
#
# Use in source and binary forms for nonexclusive, nonsublicenseable, commercial purposes with or without modification, is permitted provided that the following conditions are met:
# - Use of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# - Use in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the software.
# - Neither the name of the California Institute of Technology (Caltech) nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import sys
import os
import copy
import numpy as np
import logging
import time

def get_sourcepower( fdtd_hook_ ):
	f = fdtd_hook_.getresult('focal_monitor_0','f')		# Might need to replace 'focal_monitor_0' with a search for monitors & their names
	sp = fdtd_hook_.sourcepower(f)
	return sp

def get_afield( fdtd_hook_, monitor_name, field_indicator ):
	field_polarizations = [ field_indicator + 'x', 
					   	field_indicator + 'y', field_indicator + 'z' ]
	data_xfer_size_MB = 0

	start = time.time()

	pol_idx = 0
	logging.debug(f'Getting {field_polarizations[pol_idx]} from monitor {monitor_name}.')
	field_pol_0 = fdtd_hook_.getdata( monitor_name, field_polarizations[ pol_idx ] )
	logging.debug(f'Size {field_pol_0.nbytes}.')

	data_xfer_size_MB += field_pol_0.nbytes / ( 1024. * 1024. )

	total_field = np.zeros( [ len (field_polarizations ) ] + 
						list( field_pol_0.shape ), dtype=np.complex128 )
	total_field[ 0 ] = field_pol_0

	logging.info(f'Getting {field_polarizations[pol_idx]} from monitor {monitor_name}. Size: {data_xfer_size_MB}MB. Time taken: {time.time()-start} seconds.')

	for pol_idx in range( 1, len( field_polarizations ) ):
		logging.debug(f'Getting {field_polarizations[pol_idx]} from monitor {monitor_name}.')
		field_pol = fdtd_hook_.getdata( monitor_name, field_polarizations[ pol_idx ] )
		logging.debug(f'Size {field_pol.nbytes}.')
  
		data_xfer_size_MB += field_pol.nbytes / ( 1024. * 1024. )

		total_field[ pol_idx ] = field_pol

		logging.info(f'Getting {field_polarizations[pol_idx]} from monitor {monitor_name}. Size: {field_pol.nbytes/(1024*1024)}MB. Total time taken: {time.time()-start} seconds.')

	elapsed = time.time() - start

	date_xfer_rate_MB_sec = data_xfer_size_MB / elapsed
	logging.debug( "Transferred " + str( data_xfer_size_MB ) + " MB\n" )
	logging.debug( "Data rate = " + str( date_xfer_rate_MB_sec ) + " MB/sec\n\n" )

	return total_field

def get_hfield( fdtd_hook_, monitor_name ):
	return get_afield( fdtd_hook_, monitor_name, 'H' )

def get_efield( fdtd_hook_, monitor_name ):
	return get_afield( fdtd_hook_, monitor_name, 'E' )

def get_Pfield( fdtd_hook_, monitor_name ):
	'''Returns power as a function of space and wavelength, with three components Px, Py, Pz'''
	read_pfield = fdtd_hook_.getresult( monitor_name, 'P' )
	read_P = read_pfield[ 'P' ]
	return read_P
	# return get_afield( monitor_name, 'P' )

def get_power( fdtd_hook_, monitor_name ):
    '''Returns power as a function of space and wavelength, with three components Px, Py, Pz'''
    return get_Pfield( fdtd_hook_, monitor_name )

def get_efield_magnitude(fdtd_hook_, monitor_name):
	e_field = get_afield(fdtd_hook_, monitor_name, 'E')
	Enorm2 = np.square(np.abs(e_field[0,:,:,:,:]))
	for idx in [1,2]:
		Enorm2 += np.square(np.abs(e_field[idx,:,:,:,:]))
	return np.sqrt(Enorm2)

def get_transmission_magnitude( fdtd_hook_, monitor_name ):
	read_T = fdtd_hook_.getresult( monitor_name, 'T' )
	return np.abs( read_T[ 'T' ] )

def get_overall_power(fdtd_hook_, monitor_name):
	'''Returns power spectrum, power being the summed power over the monitor.'''
	# Lumerical defines all transmission measurements as normalized to the source power.
	source_power = get_sourcepower(fdtd_hook_)
	T = get_transmission_magnitude(fdtd_hook_, monitor_name)
	
	return source_power * np.transpose([T])



