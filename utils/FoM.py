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
		polarization: 'TE' or 'TM'
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

		self.fom = [] # The more convenient way to look at fom. For example, power transmission through a monitor even if you're optimizing for a point source.
		self.restricted_fom = [] # FOM that is being restricted. For instance, frequencies that should not be focused.
		self.true_fom = [] # The true FoM being used to define the adjoint source.
		self.gradient = None
		self.restricted_gradient = None
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
		for adj_src in self.adj_srcs:
			# todo: needed to hard-code the string replace. There should be some sort of property for the adj_src, like 'alternate_monitor_dict' or something
			source_name = adj_src.monitor_dict['name'].replace('focal','transmission')

			# The "stand-in" FoM, which we use to plot device efficiency
			T = np.abs(simulator.get_transmission_magnitude(source_name))
			logging.info(f'Accessed transmission data from {source_name}.')
			# todo: should we squeeze it??
			# todo: append or replace?! Right now it's just replace
			self.fom = T[..., self.freq_index_opt]
			self.restricted_fom = T[..., self.freq_index_restricted_opt]

			# The "true" FoM, i.e. intensity at center of device, which we use for gradient calculation.
			focal_data = simulator.get_efield(adj_src.monitor_dict['name'])
			logging.info(f'Accessed adjoint E-field data from {source_name} for intensity calculation.')
			# Convert E-vector x,y,z into |E|^2 ----- i.e. (3,nx,ny,nz,nλ) -> (1,nx,ny,nz,nλ)
			self.true_fom =  np.sum( np.abs(focal_data[..., self.freq_index_opt])**2, 
									axis=0)

	def compute_gradient(self):
 
		pos_gradient_indices = []
		neg_gradient_indices = []
		for i in self.freq_index_opt:
			pos_gradient_indices.append((self.freq_index_opt + self.freq_index_restricted_opt).index(i))
		for i in self.freq_index_restricted_opt:
			neg_gradient_indices.append((self.freq_index_opt + self.freq_index_restricted_opt).index(i))

		# df_dev = np.real(Ex_fwd*Ex_adj + Ey_fwd*Ey_adj + Ez_fwd*Ez_adj)
		df_dev = np.real(self.design_fwd_fields[0]*self.design_adj_fields[0] + \
						self.design_fwd_fields[1]*self.design_adj_fields[1] + \
						self.design_fwd_fields[2]*self.design_adj_fields[2]
					)

		logging.info('Computing Gradient')

		self.gradient = df_dev[..., pos_gradient_indices] * self.enabled
		self.restricted_gradient = df_dev[..., neg_gradient_indices] * self.enabled_restricted

		return df_dev

