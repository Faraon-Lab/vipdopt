import pickle


class Src:

	def __init__(self, src_object, monitor_object=None, sim_id=None):
		self.src_dict = src_object		# Every Src class instance must be tied to a corresponding simulation object / dictionary containing parameters and values.
		self.monitor_dict = monitor_object	# Every source (esp. an adjoint source) is located at a monitor, which may be a point or a surface. NOTE: This is also a SimObj/dictionary
		self.simulation = sim_id

	def save_to_pickle(self, fname):
		# Pickles this object to a file fname
		f = open(fname, 'wb')
		pickle.dump(self, f)
		f.close()

class Fwd_Src(Src):

	def __init__(self, src_object, polarization=None, monitor_object=None, sim_id=None):
		super().__init__(src_object, monitor_object=monitor_object, sim_id=sim_id)

		self.Ex_fwd = None
		self.Ey_fwd = None
		self.Ez_fwd = None
		self.Hx_fwd = None
		self.Hy_fwd = None
		self.Hz_fwd = None

		self.polarization = polarization

		# An attribute to store a temporary file name. Temporary files are used in parallel processing.
		self.temp_file_name = None


class Adj_Src(Src):

	def __init__(self, src_object, polarization=None, monitor_object=None, sim_id=None):
		super().__init__(src_object, monitor_object=monitor_object, sim_id=sim_id)

		self.Ex_adj = None
		self.Ey_adj = None
		self.Ez_adj = None
		self.Hx_adj = None
		self.Hy_adj = None
		self.Hz_adj = None

		self.polarization = polarization

		# An attribute to store a temporary file name. Temporary files are used in parallel processing.
		self.temp_file_name = None
