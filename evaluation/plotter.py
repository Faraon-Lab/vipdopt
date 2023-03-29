import os
import sys
import copy
import logging

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, PercentFormatter,
							   AutoMinorLocator)
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd

sys.path.append(os.getcwd())
from configs import global_params as gp
from utils import utility

#* Template
# The structure of the dictionaries we are passing into these functions are as follows:
# https://i.imgur.com/J1OctoM.png
# https://tree.nathanfriend.io/?s=(%27opNons!(%27fancy!true~fullPath!fqse~trailingSlash!true~rootDot!fqse)~U(%27U%27Os_dataCjob1CQCjob%202CQCsweep_OsC*OMJrJ*LMFmateriq%20indexE0B1B2XLKn_qlWpeakInd80-formatStrK%25.3fWiteraNng%22%3A*TrueJfY0FReflecNonE449B556B6G5B02B04Hmean%22Y1FAbsorpNonE549B656B3G12B09B01Hpeak%22JQJNtleZ%22device_rta_sweep%22%27)~version!%271%27)*%20%20-J**%22KZB%2C%200.C%5Cn*EWz_vqueVF-zMKG17Xz_stdevV0HXstaNsNcsKJC**K8%22Lshort_formM_nameNtiOplotQ*...Usource!Vs8%5B0.W%22-X%5D-YJ*line_Z%3A%20qalzvar%01zqZYXWVUQONMLKJHGFECB8-*
# Broadly, the plot_data has keys 'r', 'f', and 'title'.
# The values of 'r' and 'f' are a dictionary and array of dictionaries respectively, that contain plot metadata as well as values.
# Examples are given below:

TEMPLATE_R_VECTOR = {
					'var_name': 'Circle Radius',
					'var_values': [1.8, 2.4, 2.7, 3.0],
					'short_form': 'c_rad',
					'peakInd': 0,
					'formatStr': '%.3f',
					'iterating': True
				   }
sweep_parameters = [TEMPLATE_R_VECTOR]

TEMPLATE_F_VECTOR = {'var_name': 'Circle Diameter', 
					'var_values': [10.179, 18.096, 22.902, 28.274], 
					'var_stdevs': [0.1, 0.1, 0.1, 0.1], 
					'statistics': 'mean'
					}

TEMPLATE_PLOT_DATA = {'r':copy.deepcopy(sweep_parameters),
					  'f': [copy.deepcopy(TEMPLATE_F_VECTOR)],
					  'title': 'template'
					}


#* Plot Style Params
# if not running_on_local_machine:	
#     from matplotlib import font_manager
#     font_manager._rebuild()
#     fp = font_manager.FontProperties(fname=r"/central/home/ifoo/.fonts/Helvetica-Neue-Medium-Extended.ttf")
#     print('Font name is ' + fp.get_name())
#     plt.rcParams.update({'font.sans-serif':fp.get_name()}) 

plt.rcParams.update({'font.sans-serif':'Helvetica Neue',            # Change this based on whatever custom font you have installed
					 'font.weight': 'normal', 'font.size':20})              
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Helvetica Neue'
plt.rcParams['mathtext.it'] = 'Helvetica Neue:italic'
# plt.rcParams['text.usetex'] = True

# mpl.rcParams['font.sans-serif'] = 'Helvetica Neue'
# mpl.rcParams['font.family'] = 'sans-serif'

marker_style = dict(linestyle='-', linewidth=2.2, marker='o', markersize=4.5)
vline_style = {'color': 'gray', 'linestyle': '--', 'linewidth': 1}



#* Define a class called BasicPlot with basic plotting methods and data storage options.

def adjust_figure_size(ax, fig_width, fig_height=None):
	if fig_height is None:
		fig_height = fig_width*4/5

	l = ax.figure.subplotpars.left
	r = ax.figure.subplotpars.right
	t = ax.figure.subplotpars.top
	b = ax.figure.subplotpars.bottom
	figw = float(fig_width)/(r-l)
	figh = float(fig_height)/(t-b)
	ax.figure.set_size_inches(figw, figh, forward=True)

	return ax

def apply_common_plot_style(ax=None, plt_kwargs={}, show_legend=True):
	'''Sets universal axis properties for all plots. Function is called in each plot generating function.'''
	
	if ax is None:
		ax = plt.gca()
	
	# Creates legend    
	if show_legend:
		#ax.legend(prop={'size': 10})
		ax.legend(prop={'size': 10}, loc='center left', bbox_to_anchor=(1.1,.5))
	
	# Figure size
	ax = adjust_figure_size(ax, 8.0)
	
	# Minor Ticks
	ax.xaxis.set_minor_locator(AutoMinorLocator())
	ax.yaxis.set_minor_locator(AutoMinorLocator())
	
	# Y-Axis Exponent Repositioning
	# ax.get_yaxis().get_offset_text().set_position((0, 0.5))
	
	return ax

def enter_plot_data_1d(plot_config, fig=None, ax=None):
	'''Every plot function calls this function to actually put the data into the plot. The main input is plot_config, a dictionary that contains
	parameters controlling every single property of the plot that might be relevant to data.'''
	
	for plot_idx in range(0, len(plot_config['lines'])):
		data_line = plot_config['lines'][plot_idx]
		
		x_plot_data = data_line['x_axis']['factor'] * np.array(data_line['x_axis']['values']) + data_line['x_axis']['offset']
		y_plot_data = data_line['y_axis']['factor'] * np.array(data_line['y_axis']['values']) + data_line['y_axis']['offset']
		
		plt.plot(x_plot_data[data_line['cutoff']], y_plot_data[data_line['cutoff']],
				 color=data_line['color'], label=data_line['legend'], **data_line['marker_style'])
	
	
	plt.title(plot_config['title'])
	plt.xlabel(plot_config['x_axis']['label'])
	plt.ylabel(plot_config['y_axis']['label'])
	
	if plot_config['x_axis']['limits']:
		plt.xlim(plot_config['x_axis']['limits'])
	if plot_config['y_axis']['limits']:
		plt.ylim(plot_config['y_axis']['limits'])
	
	ax = apply_common_plot_style(ax, {})
	plt.tight_layout()
	
	# # plt.show(block=False)
	# # plt.show()
	
	return fig, ax

class BasicPlot():
	def __init__(self, plot_data):
		'''Initializes the plot_config variable of this class object and also the Plot object.'''
		self.r_vectors = plot_data['r'][0]
		self.f_vectors = plot_data['f']
		
		self.fig, self.ax = plt.subplots()
		self.plot_config = {'title': plot_data['title'],
							'x_axis': {'label':'', 'limits':[]},
							'y_axis': {'label':'', 'limits':[]},
							'lines': []
							}
	
	def append_line_data(self, plot_colors=None, plot_labels=None):
		'''Appends all line data stored in the f_vectors to the plot_config. Here we can assign colors and legend labels all at once.'''
		
		for plot_idx in range(0, len(self.f_vectors)):
			line_data = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
						'y_axis': {'values': None, 'factor': 1, 'offset': 0},
						'cutoff': None,		# A slice of two numbers i.e. slice(1,8) that defines the indices at which the data is truncated
						'color': None,
						'alpha': 1.0,
						'legend': None,
						'marker_style': marker_style
						}
			
			line_data['x_axis']['values'] = self.r_vectors['var_values']
			line_data['y_axis']['values'] = self.f_vectors[plot_idx]['var_values']
			line_data['cutoff'] = slice(0, len(line_data['x_axis']['values']))
			# line_data['cutoff'] = slice(8,-8)
			
			if plot_colors is not None:
				line_data['color'] = plot_colors[plot_idx]
			if plot_labels is not None:
				line_data['legend'] = plot_labels[plot_idx]
			else: line_data['legend'] = self.f_vectors[plot_idx]['var_name']
			
			self.plot_config['lines'].append(line_data)
	
	def alter_line_property(self, key, new_value, nested_keys=[]):
		'''Alters a property throughout all of the line data.
		If new_value is an array it will change all of the values index by index; otherwise it will blanket change everything.
		The nested_keys argument is to cover multiple-level dictionaries.'''
		
		for line_idx, line_data in enumerate(self.plot_config['lines']):
			if not isinstance(new_value, list):
				utility.set_by_path(line_data, nested_keys+[key], new_value)
			else: 
				utility.set_by_path(line_data, nested_keys+[key], new_value[line_idx])
	
	def assign_title(self, title_string=None):
		'''Replaces title of plot.'''
		if title_string is not None:
			self.plot_config['title'] = title_string
	
	def assign_axis_labels(self, x_label_string=None, y_label_string=None):
		'''Replaces axis labels of plot.'''
		if x_label_string is None:
			x_label_string = self.r_vectors['var_name']
		else:
			self.plot_config['x_axis']['label'] = x_label_string

		if y_label_string is None:
			y_label_string = self.f_vectors[0]['var_name']
		else:
			self.plot_config['y_axis']['label'] = y_label_string

		self.plot_config['x_axis']['label'] = x_label_string
		self.plot_config['y_axis']['label'] = y_label_string
		
	def export_plot_config(self, plot_directory_location, plot_subfolder_name, filename, close_plot=True):
		'''Creates plot using the plot config, and then exports.'''
		self.fig, self.ax = enter_plot_data_1d(self.plot_config, self.fig, self.ax)
  
		#! NOTE: This is where you do any additional adjustment of the plot before saving out
		# plot adjustment code

		SAVE_LOCATION = os.path.abspath(os.path.join(plot_directory_location, plot_subfolder_name))
		if not os.path.isdir(SAVE_LOCATION):
			os.makedirs(SAVE_LOCATION)

		plt.savefig(SAVE_LOCATION + f'/{filename}.png', bbox_inches='tight')
		
		# export_msg_string = (SAVE_LOCATION + f'/{filename}'.replace("_", " ")).title()
		export_msg_string = filename.replace("_", "").title()
		logging.info('Exported: ' + export_msg_string)
		
		if close_plot:
			plt.close()

class SpectrumPlot(BasicPlot):
	def __init__(self, plot_data, sweep_parameters_):
		'''Initializes the plot_config variable of this class object and also the Plot object. Here, accounts for sweep parameters.'''
		super(SpectrumPlot, self).__init__(plot_data)
		self.sweep_parameters = sweep_parameters_

	def assign_title(self, title_string, job_idx):
		'''Replaces title of plot. Overwrites function in BasicPlot and accounts for sweep parameters.'''
		
		for sweep_param_value in list(self.sweep_parameters.values()):
			current_value = sweep_param_value['var_values'][job_idx[2][0]]
			optimized_value = sweep_param_value['var_values'][sweep_param_value['peakInd']]
			title_string += '\n' + sweep_param_value['var_name'] + f': Current Value {current_value}, Optimized for {optimized_value}'
			
		self.plot_config['title'] = title_string	

	def append_line_data(self, plot_colors=None, plot_labels=None):
		'''Overwrites function in BasicPlot by applying a 1e6 factor to the x-axis (which is supposed to be wavelength in um.)'''
		super(SpectrumPlot, self).append_line_data(plot_colors, plot_labels)
		self.alter_line_property('factor', 1e6, ['x_axis'])

	def export_plot_config(self, plot_directory_location, plot_subfolder_name, filename, close_plot=True):
		'''Creates plot using the plot config, and then exports.'''
		self.fig, self.ax = enter_plot_data_1d(self.plot_config, self.fig, self.ax)
  
		#! NOTE: This is where you do any additional adjustment of the plot before saving out
		# plot adjustment code

		SAVE_LOCATION = os.path.abspath(os.path.join(plot_directory_location, plot_subfolder_name))
		if not os.path.isdir(SAVE_LOCATION):
			os.makedirs(SAVE_LOCATION)
		filename = utility.isolate_filename(filename).replace('.fsp', '')

		plt.savefig(SAVE_LOCATION + f'/{filename}.png', bbox_inches='tight')
		
		export_msg_string = (SAVE_LOCATION + f'/{filename}'.replace("_", " ")).title()
		logging.info('Exported: ' + export_msg_string)
		
		if close_plot:
			plt.close()

# TODO: Class SweepPlot

#* Functions that call one of the Plot classes

def plot_basic_1d(plot_data,  plot_directory_location, plot_subfolder_name, filename, 
				  title=None, xlabel_txt=None, ylabel_txt=None):
	# Calls an instance of BasicPlot, initializes with the data.
	bp = BasicPlot(plot_data)
	bp.append_line_data()		# Sets up plot config with line data.
	bp.assign_title(title_string=title)			# Assign title
	bp.assign_axis_labels(x_label_string=xlabel_txt, y_label_string=ylabel_txt)		 # Assign x and y axis labels
	bp.export_plot_config(plot_directory_location, plot_subfolder_name, filename)	 # Do actual plotting and then save out the figure.

#* Evaluation during Optimization

def plot_fom_trace(f, plot_directory_location):
	# Plot FOM trace
	trace = []
	numEpochs = len(f); numIter = len(f[0])
	upperRange = np.ceil(np.max(f))

	for epoch_fom in f:
		trace = np.concatenate((trace, epoch_fom), axis=0)

	iterations = copy.deepcopy(TEMPLATE_R_VECTOR)
	iterations.update({'var_name': 'Iterations', 'var_values': range(f.size), 'short_form': 'iter'})
	fom = copy.deepcopy(TEMPLATE_F_VECTOR)
	fom.update({'var_name': 'FoM', 'var_values': trace})
	plot_data = {'r':[iterations],
				'f': [fom],
				'title': 'Figure of Merit - Trace'}
 
	bp = BasicPlot(plot_data)
	bp.append_line_data(plot_colors=['orange'])
	bp.assign_title()
	bp.assign_axis_labels()
	fig, ax = bp.fig, bp.ax
	for i in range(0,numEpochs+1):
		plt.vlines(i*numIter, 0,upperRange, **vline_style)
	bp.fig, bp.ax = fig, ax
	bp.export_plot_config(plot_directory_location,'','fom_trace')
 
#* Evaluation after Optimization

# -- Spectrum Plot Functions (Per Job)

def plot_sorting_transmission_spectrum(plot_data, job_idx, sweep_parameters, job_names, plot_folder, include_overall=False):
	'''For the given job index, plots the sorting efficiency (transmission) spectrum corresponding to that job.'''
	
	# plot_colors = ['blue', 'green', 'red', 'pink']
	# plot_labels = ['Blue', 'Green (x-pol.)', 'Red', 'Green (y-pol.)']
	plot_colors = ['blue', 'green', 'red']	
	plot_labels = ['Blue', 'Green', 'Red']

	if include_overall:
		plot_colors.append('gray')
		plot_labels.append('Trans.')
	else:
		plot_data['f'].pop(-1)		# remove overall transmission from f-vectors

	sp = SpectrumPlot(plot_data, sweep_parameters)
	
	sp.append_line_data(plot_colors, plot_labels)
	
	sp.assign_title('Spectrum in Each Quadrant', job_idx)
	sp.assign_axis_labels('Wavelength (um)', 'Sorting Transmission Efficiency')
	
	sp.export_plot_config(plot_folder, 'sorting_trans_efficiency_spectra', utility.isolate_filename(job_names[job_idx]).replace('.fsp', ''))
	return sp.fig, sp.ax

def plot_device_rta_spectrum(plot_data, job_idx, sweep_parameters, job_names, plot_folder, sum_sides=False):
	'''For the given job index, plots the device RTA and all power components in and out, corresponding to that job.
	The normalization factor is the power through the device input aperture.'''

	plot_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
	plot_labels = None			#! Use the variable names stored in plot_data to label
	# plot_labels = ['Device Reflection', 'Side Scattering E', 'Side Scattering N', 'Side Scattering W', 'Side Scattering S',
	# 			   'Focal Region Power', 'Oblique Scattering Power', 'Absorbed Power']
	# plot_labels = ['Device Reflection', 'Focal Region Transmission', 'Nearest Neighbours Transmission', 'Side Scattering Transmission']

	if sum_sides:
		for var_name_ in ['Side Scattering E', 'Side Scattering N', 'Side Scattering W', 'Side Scattering S']:
			for line_data in plot_data['f']:
				if line_data['var_name'] == var_name_:
					plot_data['f'].remove(line_data)

	sp = SpectrumPlot(plot_data, sweep_parameters)
 
	sp.append_line_data(plot_colors, plot_labels)
	
	sp.assign_title('Device RTA', job_idx)
	sp.assign_axis_labels('Wavelength (um)', 'Normalized Power')
	
	sp.export_plot_config(plot_folder, 'device_rta_spectra', utility.isolate_filename(job_names[job_idx]).replace('.fsp', ''))
	return sp.fig, sp.ax

def plot_device_rta_pareto(plot_data, job_idx, sweep_parameters, job_names, plot_folder, sum_sides=False):
	'''For the given job index, plots the device RTA and all power components in and out, corresponding to that job.
	The normalization factor is the power through the device input aperture.'''

	plot_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
	plot_labels = None			#! Use the variable names stored in plot_data to label
	# plot_labels = ['Device Reflection', 'Side Scattering E', 'Side Scattering N', 'Side Scattering W', 'Side Scattering S',
	# 			   'Focal Region Power', 'Oblique Scattering Power', 'Absorbed Power']
	# plot_labels = ['Device Reflection', 'Focal Region Transmission', 'Nearest Neighbours Transmission', 'Side Scattering Transmission']

	if sum_sides:
		remove_f_data = ['Side Scattering E', 'Side Scattering N', 'Side Scattering W', 'Side Scattering S']
	else:
		remove_f_data = ['Side Scatter']
	for var_name_ in remove_f_data:
		for line_data in plot_data['f']:
			if line_data['var_name'] == var_name_:
				plot_data['f'].remove(line_data)

	# We are going to condense everything into a giant matrix, wih shape num_f_vectors x 3 i.e. {B,G,R}
	
	category_names = []
	pareto_matrix = np.zeros( (len(plot_data['f'][0]['var_values']), len(plot_data['f'])) )
	for line_idx, line_data in enumerate(plot_data['f']):
		category_names.append(line_data['var_name'])
		pareto_matrix[:, line_idx] = line_data['var_values'].reshape(-1)

	plot_colors = ['blue','green', 'red']
	for band_idx, spectral_band in enumerate(plot_colors):

		##  For each spectral band, we sort the category names and values according to order:
		# category_values, category_names_sorted = (list(t) for t in zip(*sorted(zip(pareto_matrix[0,:], category_names))))
		# category_names_sorted.reverse()
		# category_values.reverse()
		# category_values = np.array(category_values)*100

		# See: https://stackoverflow.com/a/53578962
		df = pd.DataFrame({spectral_band: 100*pareto_matrix[band_idx,:]})
		df.index = category_names
		df = df.sort_values(by=spectral_band,ascending=False)
		df["cumpercentage"] = df[spectral_band].cumsum()/df[spectral_band].sum()*100

		fig, ax = plt.subplots()
		ax.bar(df.index, df[spectral_band], color=spectral_band)
		ax.yaxis.set_major_formatter(PercentFormatter())
		ax.set_ylabel('Normalized Power', color=spectral_band)
		ax2 = ax.twinx()
		ax2.plot(df.index, df["cumpercentage"], color="C1", marker="D", ms=7)
		ax2.yaxis.set_major_formatter(PercentFormatter())
		ax2.set_ylabel('Cumulative Power', color="C1")

		ax.tick_params(axis="y", colors="black")
		ax2.tick_params(axis="y", colors="black")
			
		ax = apply_common_plot_style(ax, {})
		ax2 = apply_common_plot_style(ax2, {})
		# plt.setp(ax.get_xticklabels(), rotation=0, horizontalalignment='right')
		plt.setp(ax.get_xticklabels(), fontsize=11, rotation=0, horizontalalignment='center')
		plt.title('Device RTA')
		plt.tight_layout()

		SAVE_LOCATION = os.path.abspath(os.path.join(plot_folder, 'device_rta_spectra'))
		if not os.path.isdir(SAVE_LOCATION):
			os.makedirs(SAVE_LOCATION)
		filename = utility.isolate_filename(job_names[job_idx]).replace('.fsp', '') + f'_pareto_deviceRTA_{spectral_band}'

		plt.savefig(SAVE_LOCATION + f'/{filename}.png', bbox_inches='tight')

		export_msg_string = (SAVE_LOCATION + f'/{filename}'.replace("_", " ")).title()
		logging.info('Exported: ' + export_msg_string)
		plt.close()


# -- Sweep Plot Functions (Overall)

def plot_sorting_transmission_sweep_1d(plot_data, slice_coords, plot_stdDev = True):
	'''Produces a 1D sorting (transmission) efficiency plot using given coordinates to slice the N-D array of data contained in plot_data.
	X-axis i.e. sweep variable is given by the index of the entry of type 'slice' in the slice_coords.'''
	
	sp = SweepPlot(plot_data, slice_coords)
	
	plot_colors = ['blue', 'green', 'red', 'gray']
	plot_labels = ['Blue', 'Green (x-pol.)', 'Red', 'Green (y-pol.)']
	sp.append_line_data(slice_coords, plot_stdDev, plot_colors, plot_labels)
	
	sp.assign_title('Sorting Transmission Efficiency - Parameter Range:')
	sp.assign_axis_labels(slice_coords, 'Sorting Transmission Efficiency')
	
	sp.export_plot_config('sorting_trans_eff_sweep_', slice_coords)
	
	return sp.fig, sp.ax

 
if __name__ == "__main__":
	plot_directory_location = 'plots'
	cutoff_1d_sweep_offset = [0, 0]

	#* This is where we perturb the data points as needed
	# TODO: modify data code

	#* Call plotting function and export 
	plot_basic_1d(TEMPLATE_PLOT_DATA, 'evaluation/plots', 'test', 'test1',
			  			 title='Variation of Circle Diameter with Radius')
 
print(3)