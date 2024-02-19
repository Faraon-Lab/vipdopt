import numpy as np
import pickle
import copy
import logging

def load_foms(fom_arr, fname):
	f = open(fname, 'rb')
	for fom_temp in fom_arr:
		try:
			fom_obj = pickle.load(f)
			fom_temp.fom = fom_obj.fom
			fom_temp.true_fom = fom_obj.true_fom
			try:
				fom_temp.restricted_fom = fom_obj.restricted_fom
			except:
				fom_temp.restricted_fom = []
		except (EOFError, pickle.UnpicklingError):
			print('Error occurred during depickling.')
	f.close()

def write_arr_to_pickle(fom_arr, fname):
	f = open(fname, 'wb')
	for fom_curr in fom_arr:
		temp = copy.copy(fom_curr)
		temp.adj_src = None
		temp.fwd_src = None
		temp.mon = None
		temp.T_mon = None
		pickle.dump(temp, f)
	f.close()


class FoM():

	def __init__(self):
		pass

	def save_to_pickle(self, fname):
		# Pickles this object to a file fname
		f = open(fname, 'wb')
		pickle.dump(self, f)
		f.close()

	def load_fom_from_pickle(self, fname):
		'''
		The references to Lumerical objects can not be loaded from the old pickle. The user must always reinitialize this object the usual way, but
		can load the history of another fom (i.e. the 'fom' attribute) from the past.
		NOTE: This will OVERWRITE the current objects fom. This is intended to be used when initializing an object, so it assumes fom is empty.
		'''
		try:
			f = open(fname, 'rb')
			temp_obj = pickle.load(f)
			f.close()

			setattr(self,'fom',getattr(temp_obj, 'fom'))

		except:
			print("Import of pickle failed")
	
class Arbitrary_FoM(FoM):
	''' This class is intended to replace all previous classes eventually. The idea is that it is an arbitrary object, 
	with the computation of the FoM being a function that is stored as an object attribute. '''

	def __init__(self, fwd_srcs, adj_srcs, polarization, fom_type, freq, freq_index_opt=None, freq_index_restricted_opt=[], **kwargs):
		'''
		args:
		fwd_srcs: List of Forward_Source object
		adj_srcs: List of Adjoint source object
		fom_function: a method that computes the fom. This can use arguments from kwargs.
		polarization: 'TE' or 'TM' - important because source_weight needs the corresponding dipole to take the same direction as the polarization
		fom_type: 'dipole' or 'mode_overlap'
		freq: frequency points. Usually the same as the optimizer's frequency
		freq_index_opt: Which index values to optimize in the freq variable.
		freq_index_restricted_opt: Specify and array of frequency indices that should be optimized with a negative gradient. 
									This is useful for making sure light does not focus to a point, for example.

		kwargs: examples
		'''

		self.fwd_srcs = fwd_srcs
		self.adj_srcs = adj_srcs
		self.polarization = polarization
		self.freq = freq
		self.fom_type = fom_type
		if freq_index_opt is None:
			self.freq_index_opt = list(np.arange(0,len(freq)))
		else:
			self.freq_index_opt = freq_index_opt
		self.freq_index_restricted_opt = freq_index_restricted_opt

		# Add kwargs to the instance's __dict__
		self.__dict__.update(kwargs)
	
	def enable_all(self):
		'''Disables the FoM for all frequency indices (across the whole spectrum).'''
		# Boolean array specifying which frequencies are active.
		# This is a bit confusing. Almost deprecated really. enabled means of the frequencies being optimized, 
		# which are enabled. Useful in rare circumstances where some things need to be fully disabled to help catch up.
		self.enabled = np.ones((len(self.freq_index_opt)))

	def disable_all(self):
		'''Disables the FoM for all frequency indices (across the whole spectrum).'''
		self.enabled = np.zeros((len(self.freq_index_opt)))


class BayerFilter_FoM(FoM):

	def __init__(self, fwd_srcs, adj_srcs, polarization, freq, freq_index_opt=None, freq_index_restricted_opt=[]):
		'''
		fwd_src: The forward source that the FoM is associated with. This is required in order to turn on the correct source. It is a Lumerical Object type.
		adj_src: The adjoint source that the FoM is associated with. .
		freq: The frequency vector that the simulation uses for its monitors. This is used for information only - an example is when plotting information later.
		freq_index_opt: The indices of the simulation monitors' frequency vector which this fom is optimized at. In optimization.py, this indexing is used
			to retrieve the relevant information from the monitor.
		polarization: Either 'TE', 'TM', or 'TE+TM'
		freq_index_negative_opt: Specify and array of frequency indices that should be optimized with a negative gradient. This is useful for making sure light does not focus to a point, for example.
		'''

		self.fwd_srcs = fwd_srcs
		self.adj_srcs = adj_srcs
		
		self.freq = freq

		if freq_index_opt is None:
			self.freq_index_opt = list(np.arange(0,len(freq)))
		else:
			self.freq_index_opt = list(freq_index_opt)

		self.freq_index_restricted_opt = freq_index_restricted_opt

		self.fom = np.array([]) # The more convenient way to look at fom. For example, power transmission through a monitor even if you're optimizing for a point source.
		self.restricted_fom = np.array([]) # FOM that is being restricted. For instance, frequencies that should not be focused.
		self.true_fom = np.array([]) # The true FoM being used to define the adjoint source.
		self.gradient = np.array([])
		self.restricted_gradient = np.array([])
		self.polarization = polarization
		self.T = []
	
		self.tempfile_fwd_name = ''
		self.tempfile_adj_name = ''
		
		self.design_fwd_fields = None
		self.design_adj_fields = None

		# Boolean array specifying which frequencies are active.
		# This is a bit confusing. Almost deprecated really. enabled means of the frequencies being optimized, which are enabled. Useful in rare circumstances where some things need to be fully disable to help catch up.
		self.enabled = np.ones((len(self.freq_index_opt)))
		# Adding this once I started optimizing for functions we DONT want (i.e. restricted_gradient). This is of the freq_index_opt_restricted values, which do we want to optimize?
		self.enabled_restricted = np.ones((len(self.freq_index_restricted_opt)))

	def disable_all(self):
		self.enabled = np.zeros((len(self.freq_index_opt)))
	
	def enable_all(self):
		self.enabled = np.ones((len(self.freq_index_opt)))
	
	def compute_fom(self, simulator):
		for adj_src in self.adj_srcs:	# todo: this starts breaking if there's more than one adjoint source in the list
			# todo: needed to hard-code the string replace. There should be some sort of property for the adj_src, like 'alternate_monitor_dict' or something
			source_name = adj_src.monitor_dict['name'].replace('focal','transmission')

			# The "stand-in" FoM, which we use to plot device efficiency
			T = np.abs(simulator.get_transmission_magnitude(source_name))
			logging.info(f'Accessed transmission data from {source_name}.')
			# todo: should we squeeze it??
			# todo: append or replace?! Right now it's just replace
			self.fom = T[...] # T[..., self.freq_index_opt]		# todo: for now put all frequency slicing into compute_gradient()
			self.restricted_fom = T[..., self.freq_index_restricted_opt]

			# The "true" FoM, i.e. intensity at center of device, which we use for gradient calculation.
			focal_data = simulator.get_efield(adj_src.monitor_dict['name'])
			logging.info(f'Accessed adjoint E-field data from {source_name} for intensity calculation.')

			#* Conjugate of E_{old}(x_0) -field at the adjoint source of interest, with direction along the polarization
			# This is going to be the amplitude of the dipole-adjoint source driven at the focal plane
			#! Reminder that this is only applicable for dipole-based adjoint sources!!!
			pol_xy_idx = 0 if adj_src.src_dict['phi'] == 0 else 1		# x-polarized if phi = 0, y-polarized if phi = 90.	
			# todo: REDO - direction of source_weight vector potential error.

			# self.source_weight = np.squeeze( np.conj(
			# 		focal_data[pol_xy_idx, 0, 0, 0, :]			# shape: (3, nx, ny, nz, nλ)
			# 		#get_focal_data[adj_src_idx][xy_idx, 0, 0, 0, spectral_indices[0]:spectral_indices[1]:1]
			# 		) )
			self.source_weight = np.squeeze( np.conj(
					focal_data[:,0,0,0,:]
					))
			# Recall that E_adj = source_weight * what we call E_adj i.e. the Green's function[design_efield from adj_src simulation]
			# Reshape source weight (nλ) to (1, 1, 1, nλ) so it can be multiplied with (E_fwd * E_adj)
			# https://stackoverflow.com/a/30032182
			# given_axis = [0,-1]
			# dim_array = np.ones((1, self.design_fwd_fields[0].ndim),int).ravel()
			# dim_array[given_axis] = -1
			# self.source_weight = self.source_weight.reshape(dim_array)
			self.source_weight = np.expand_dims(self.source_weight, axis=(1,2,3))

			# Convert E-vector x,y,z into |E|^2 ----- i.e. (3,nx,ny,nz,nλ) -> (1,nx,ny,nz,nλ)
			# self.true_fom =  np.sum( np.abs(focal_data[..., self.freq_index_opt])**2, 
			# 						axis=0)
			self.true_fom =  np.sum( np.abs(focal_data[...])**2, 
									axis=0)
   

	def compute_gradient(self):
 
		# pos_gradient_indices = []
		# neg_gradient_indices = []
		# for i in self.freq_index_opt:
		# 	pos_gradient_indices.append((self.freq_index_opt + self.freq_index_restricted_opt).index(i))
		# for j in self.freq_index_restricted_opt:
		# 	neg_gradient_indices.append((self.freq_index_opt + self.freq_index_restricted_opt).index(j))

		# df_dev = np.real(Ex_fwd*Ex_adj + Ey_fwd*Ey_adj + Ez_fwd*Ez_adj)

		#! DEBUG: Check orthogonality and direction of E-fields in the design monitor
		logging.info((f'Forward design fields have average absolute xyz-components: '
					f'{np.mean(np.abs(self.design_fwd_fields[0]))}, {np.mean(np.abs(self.design_fwd_fields[1]))}, '
					f'{np.mean(np.abs(self.design_fwd_fields[2]))}.'
					))
		logging.info((f'Adjoint design fields have average absolute xyz-components: '
					f'{np.mean(np.abs(self.design_adj_fields[0]))}, {np.mean(np.abs(self.design_adj_fields[1]))}, '
					f'{np.mean(np.abs(self.design_adj_fields[2]))}.'
					))
		logging.info((f'Source weight has average absolute xyz-components: '
				f'{np.mean(np.abs(self.source_weight[0]))}, {np.mean(np.abs(self.source_weight[1]))}, '
				f'{np.mean(np.abs(self.source_weight[2]))}.'
					))

		self.design_adj_fields = self.design_adj_fields * self.source_weight
		df_dev = 1 * (self.design_fwd_fields[0]*self.design_adj_fields[0] + \
				self.design_fwd_fields[1]*self.design_adj_fields[1] + \
				self.design_fwd_fields[2]*self.design_adj_fields[2]
			)
		# Taking the real part comes when multiplying by Δε0 i.e. change in permittivity.

		logging.info('Computing Gradient')

		self.gradient = np.zeros(df_dev.shape, dtype=np.complex128)
		self.restricted_gradient = np.zeros(df_dev.shape, dtype=np.complex128)

		self.gradient[..., self.freq_index_opt] = df_dev[..., self.freq_index_opt] * self.enabled
		self.restricted_gradient[..., self.freq_index_restricted_opt] = df_dev[..., self.freq_index_restricted_opt] * self.enabled_restricted

		# self.gradient = df_dev[..., pos_gradient_indices] * self.enabled
		# self.restricted_gradient = df_dev[..., neg_gradient_indices] * self.enabled_restricted

		return df_dev

