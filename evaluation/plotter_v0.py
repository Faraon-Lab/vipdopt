import os
import sys
import copy

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
							   AutoMinorLocator)
from mpl_toolkits.axes_grid1 import make_axes_locatable

#
#* Plot Types
#

# plots = sweep_settings['plots']

template_r_vector = {
					'var_name': 'material index',
					'var_values': [1.8, 2.4, 2.7, 3.0],
					'short_form': 'n_all',
					'peakInd': 0,
					'formatStr': '%.3f',
					'iterating': True
				   }
sweep_parameters = {template_r_vector['short_form']: template_r_vector}

template_f_vector = {'var_name': 'device_rta_0', 
					'var_values': [0.04491777, 0.055, 0.066, 0.077], 
					'var_stdevs': [0, 0, 0, 0], 
					'statistics': 'mean'
					}

template_plot_data = {'r':copy.deepcopy(sweep_parameters),
					  'f': [copy.deepcopy(template_f_vector)],
					  'title': 'template'
					}


def create_parameter_filename_string(idx):
	''' Input: tuple coordinate in the N-D array. Cross-matches against the sweep_parameters dictionary.
	Output: unique identifier string for naming the corresponding job'''
	
	output_string = ''
	
	for t_idx, p_idx in enumerate(idx):
		try:
			variable = list(sweep_parameters.values())[t_idx]
			variable_name = variable['short_form']
			if isinstance(p_idx, slice):
				output_string += variable_name + '_swp_'
			else:
				variable_value = variable['var_values'][p_idx]
				variable_format = variable['formatStr']
				output_string += variable_name + '_' + variable_format%(variable_value) + '_'
		except Exception as err:
			pass
		
	return output_string[:-1]

#* ----------------------------------------------------------------------------------------------------------------------------------------


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

def apply_common_plot_style(ax=None, plt_kwargs={}, showLegend=True):
	'''Sets universal axis properties for all plots. Function is called in each plot generating function.'''
	
	if ax is None:
		ax = plt.gca()
	
	# Creates legend    
	if showLegend:
		#ax.legend(prop={'size': 10})
		ax.legend(prop={'size': 10}, loc='center left', bbox_to_anchor=(1,.5))
	
	# Figure size
	fig_width = 8.0
	fig_height = fig_width*4/5
	l = ax.figure.subplotpars.left
	r = ax.figure.subplotpars.right
	t = ax.figure.subplotpars.top
	b = ax.figure.subplotpars.bottom
	figw = float(fig_width)/(r-l)
	figh = float(fig_height)/(t-b)
	ax.figure.set_size_inches(figw, figh, forward=True)
	
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
		self.r_vectors = plot_data['r']
		self.f_vectors = plot_data['f']
		
		self.fig, self.ax = plt.subplots()
		self.plot_config = {'title': None,
					'x_axis': {'label':'', 'limits':[]},
					'y_axis': {'label':'', 'limits':[]},
					'lines': []
					}
	
	def append_line_data(self, plot_colors=None, plot_labels=None):
		'''Appends all line data stored in the f_vectors to the plot_config. Here we can assign colors and legend labels all at once.'''
		
		for plot_idx in range(0, len(self.f_vectors)):
			line_data = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
						'y_axis': {'values': None, 'factor': 1, 'offset': 0},
						'cutoff': None,
						'color': None,
						'alpha': 1.0,
						'legend': None,
						'marker_style': marker_style
					}
			
			line_data['x_axis']['values'] = self.r_vectors[list(self.r_vectors.keys())[0]]['var_values']
			line_data['y_axis']['values'] = self.f_vectors[plot_idx]['var_values']
			line_data['cutoff'] = slice(0, len(line_data['x_axis']['values']))
			# line_data['cutoff'] = slice(8,-8)
			
			if plot_colors is not None:
				line_data['color'] = plot_colors[plot_idx]
			if plot_labels is not None:
				line_data['legend'] = plot_labels[plot_idx]
			
			self.plot_config['lines'].append(line_data)
	
	def alter_line_property(self, key, new_value, axis_str=None):
		'''Alters a property throughout all of the line data.
		If new_value is an array it will change all of the values index by index; otherwise it will blanket change everything.
		The axis_str argument is to cover multiple-level dictionaries.'''
		
		for line_idx, line_data in enumerate(self.plot_config['lines']):
			if axis_str is None:
				if not isinstance(new_value, list):
					line_data[key] = new_value
				else:
					line_data[key] = new_value[line_idx]
			else:
				if not isinstance(new_value, list):
					line_data[axis_str][key] = new_value
				else:
					line_data[axis_str][key] = new_value[line_idx]
	
	def assign_title(self, title_string):
		'''Replaces title of plot.'''
			
		self.plot_config['title'] = title_string
	
	def assign_axis_labels(self, x_label_string, y_label_string):
		'''Replaces axis labels of plot.'''
		
		self.plot_config['x_axis']['label'] = x_label_string
		self.plot_config['y_axis']['label'] = y_label_string
		
	def export_plot_config(self, plot_directory_location, plot_subfolder_name, close_plot=True):
		'''Creates plot using the plot config, and then exports.'''
		self.fig, self.ax = enter_plot_data_1d(self.plot_config, self.fig, self.ax)
		
		if not os.path.isdir(plot_directory_location + '/' + plot_subfolder_name):
			os.makedirs(plot_directory_location + '/' + plot_subfolder_name)
		plt.savefig(plot_directory_location + '/' + plot_subfolder_name\
			+ '/'  + '.png',
			bbox_inches='tight')
		
		export_msg_string = (plot_subfolder_name.replace("_", " ")).title()
		print('Exported: ' + export_msg_string)
		sys.stdout.flush()
		
		if close_plot:
			plt.close()

class SpectrumPlot(BasicPlot):
	def __init__(self, plot_data):
		'''Initializes the plot_config variable of this class object and also the Plot object.'''
		super(SpectrumPlot, self).__init__(plot_data)
	
	def append_line_data(self, plot_colors=None, plot_labels=None):
		super(SpectrumPlot, self).append_line_data(plot_colors, plot_labels)
		self.alter_line_property('factor', 1e6, axis_str='x_axis')
  
	def assign_title(self, title_string, job_idx):
		'''Replaces title of plot.
		Overwrites method in BasicPlot.'''
		
		for sweep_param_value in list(sweep_parameters.values()):
			current_value = sweep_param_value['var_values'][job_idx[2][0]]
			optimized_value = sweep_param_value['var_values'][sweep_param_value['peakInd']]
			title_string += '\n' + sweep_param_value['var_name'] + f': Current Value {current_value}, Optimized for {optimized_value}'
			
		self.plot_config['title'] = title_string

	def export_plot_config(self, plot_subfolder_name, job_idx, close_plot=True):
		'''Creates plot using the plot config, and then exports.
  		Overwrites the method in BasicPlot.'''
    
		self.fig, self.ax = enter_plot_data_1d(self.plot_config, self.fig, self.ax)
		
		if not os.path.isdir(plot_directory_location + '/' + plot_subfolder_name):
			os.makedirs(plot_directory_location + '/' + plot_subfolder_name)
		plt.savefig(plot_directory_location + '/' + plot_subfolder_name\
			+ '/' + isolate_filename(job_names[job_idx]).replace('.fsp', '') + '.png',
			bbox_inches='tight')
		
		export_msg_string = (plot_subfolder_name.replace("_", " ")).title()
		print('Exported: ' + export_msg_string)
		sys.stdout.flush()
		
		if close_plot:
			plt.close()
		
class SweepPlot(BasicPlot):
	def __init__(self, plot_data, slice_coords):
		'''Initializes the plot_config variable of this class object and also the Plot object.'''
		super(SweepPlot, self).__init__(plot_data)
		
		# Whichever entry is a slice means that variable in r_vectors is the x-axis of this plot
		r_vector_value_idx = [index for (index , item) in enumerate(slice_coords) if isinstance(item, slice)][0]
		self.r_vectors = list(plot_data['r'].values())[r_vector_value_idx]
		#f_vectors = plot_data['f']              # This is a list of N-D arrays, containing the function values for each parameter combination
	
	
	def append_line_data(self, slice_coords, plot_stdDev, plot_colors=None, plot_labels=None, normalize_against_max=False):
		'''Appends all line data stored in the f_vectors to the plot_config. Here we can assign colors and legend labels all at once.
		Overwriting the method in BasePlot.'''
		
		for plot_idx in range(0, len(self.f_vectors)-0):
			line_data = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
						'y_axis': {'values': None, 'factor': 1, 'offset': 0},
						'cutoff': None,
						'color': None,
						'alpha': 1.0,
						'legend': None,
						'marker_style': marker_style
					}
		
			y_plot_data = self.f_vectors[plot_idx]['var_values']
			y_plot_data = np.reshape(y_plot_data, (len(y_plot_data), 1))
			y_plot_data = y_plot_data[tuple(slice_coords)]
			
			normalization_factor = np.max(y_plot_data) if normalize_against_max else 1
			line_data['y_axis']['values'] = y_plot_data / normalization_factor
			line_data['x_axis']['values'] = self.r_vectors['var_values']
			# line_data['cutoff'] = slice(0, len(line_data['x_axis']['values']))
			line_data['cutoff'] = slice(0 + cutoff_1d_sweep_offset[0], len(line_data['x_axis']['values']) + cutoff_1d_sweep_offset[1])
			
			line_data['color'] = plot_colors[plot_idx]
			line_data['legend'] = plot_labels[plot_idx]
			
			self.plot_config['lines'].append(line_data)
			
			
			if plot_stdDev and self.f_vectors[plot_idx]['statistics'] in ['mean']:
				line_data_2 = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
						'y_axis': {'values': None, 'factor': 1, 'offset': 0},
						'cutoff': None,
						'color': None,
						'alpha': 0.3,
						'legend': '_nolegend_',
						'marker_style': dict(linestyle='-', linewidth=1.0)
					}
			
				line_data_2['x_axis']['values'] = self.r_vectors['var_values']
				line_data_2['y_axis']['values'] = (y_plot_data + self.f_vectors[plot_idx]['var_stdevs']) / normalization_factor
				# line_data_2['cutoff'] = slice(0, len(line_data_2['x_axis']['values']))
				line_data_2['cutoff'] = slice(0 + cutoff_1d_sweep_offset[0], len(line_data_2['x_axis']['values']) + cutoff_1d_sweep_offset[1])
				
				colors_so_far = plt.rcParams["axes.prop_cycle"].by_key()["color"]
				line_data_2['color'] = colors_so_far[-1]
				
				self.plot_config['lines'].append(line_data_2)
				
				line_data_3 = copy.deepcopy(line_data_2)
				line_data_3['y_axis']['values'] = (y_plot_data - self.f_vectors[plot_idx]['var_stdevs']) / normalization_factor
				self.plot_config['lines'].append(line_data_3)
		
	def assign_title(self, title_string):
		'''Replaces title of plot. 
		Overwriting the method in BasePlot.'''
		
		for sweep_param_value in list(sweep_parameters.values()):
			optimized_value = sweep_param_value['var_values'][sweep_param_value['peakInd']]
			title_string += '\n' + sweep_param_value['var_name'] + f': Optimized at {optimized_value}'
		self.plot_config['title'] = title_string
	
	def assign_axis_labels(self, slice_coords, y_label_string):
		'''Replaces axis labels of plot.
		Overwriting the method in BasePlot.'''
		
		sweep_variable_idx = [index for (index, item) in enumerate(slice_coords) if type(item) is slice][0]
		self.plot_config['x_axis']['label'] = list(sweep_parameters.values())[sweep_variable_idx]['var_name']
		self.plot_config['y_axis']['label'] = y_label_string
	
	def export_plot_config(self, file_type_name, slice_coords, plot_subfolder_name='', close_plot=True):
		'''Creates plot using the plot config, and then exports.
		Overwriting the method in BasePlot.'''
		
		self.fig, self.ax = enter_plot_data_1d(self.plot_config, self.fig, self.ax)
		
		if not os.path.isdir(plot_directory_location + '/' + plot_subfolder_name):
			os.makedirs(plot_directory_location + '/' + plot_subfolder_name)
		plt.savefig(plot_directory_location + '/' + plot_subfolder_name\
			+ '/' + file_type_name + create_parameter_filename_string(slice_coords) + '.png',
			bbox_inches='tight')
		
		export_msg_string = (file_type_name.replace("_", " ")).title()
		print('Exported: ' + export_msg_string + create_parameter_filename_string(slice_coords))
		sys.stdout.flush()
		
		if close_plot:
			plt.close()

def plot_basic_sweep_1d(plot_data, title_txt=None, 
                        xlabel_txt=None, ylabel_txt=None, plot_colors=None, plot_labels=None,
                        plot_directory_location='plots', filename_txt=None):
	'''Produces a 1D sweep plot using given coordinates to slice the N-D array of data contained in plot_data.
	X-axis i.e. sweep variable is given by the index of the entry of type 'slice' in the slice_coords.'''
	
	if xlabel_txt is None:
		xlabel_txt = plot_data['r'][list(plot_data['r'].keys())[0]]
	if ylabel_txt is None:
		ylabel_txt = plot_data['f'][0]['var_name']
	if title_txt is None:
		title_txt = ''
	if filename_txt is None:
		filename_txt=''
 
	sp = BasicPlot(plot_data)
 
	sp.append_line_data(plot_colors, plot_labels)
	
	sp.assign_title(title_txt)
	sp.assign_axis_labels(xlabel_txt, ylabel_txt)
	
	sp.export_plot_config(plot_directory_location, filename_txt)
	
	return sp.fig, sp.ax

def plot_sorting_transmission_spectrum(plot_data, job_idx):
	'''For the given job index, plots the sorting efficiency (transmission) spectrum corresponding to that job.'''
	
	sp = SpectrumPlot(plot_data)
	
	plot_colors = ['blue', 'green', 'red', 'gray']
	plot_labels = ['Blue', 'Green (x-pol.)', 'Red', 'Green (y-pol.)']
	sp.append_line_data(plot_colors, plot_labels)
	
	sp.assign_title('Spectrum in Each Quadrant', job_idx)
	sp.assign_axis_labels('Wavelength (um)', 'Sorting Transmission Efficiency')
	
	sp.export_plot_config('sorting_trans_efficiency_spectra', job_idx)
	return sp.fig, sp.ax

def plot_sorting_transmission_sweep_1d(plot_data, slice_coords, plot_stdDev = True):
	'''Produces a 1D sorting (transmission) efficiency plot using given coordinates to slice the N-D array of data contained in plot_data.
	X-axis i.e. sweep variable is given by the index of the entry of type 'slice' in the slice_coords.'''
	
	sp = SweepPlot(plot_data, slice_coords)
	
	plot_colors = ['blue', 'green', 'red', 'gray']
	# plot_labels = ['Blue', 'Green (x-pol.)', 'Red', 'Green (y-pol.)']
	plot_labels = ['Blue', 'Green', 'Red']
	sp.append_line_data(slice_coords, plot_stdDev, plot_colors, plot_labels)
	
	sp.assign_title('Sorting Transmission Efficiency - Parameter Range:')
	sp.assign_axis_labels(slice_coords, 'Sorting Transmission Efficiency')
	
	sp.export_plot_config('sorting_trans_eff_sweep_', slice_coords)
	
	return sp.fig, sp.ax

#* The structure of the dictionaries we are passing into these functions are as follows:
# https://i.imgur.com/J1OctoM.png
# https://tree.nathanfriend.io/?s=(%27opNons!(%27fancy!true~fullPath!fqse~trailingSlash!true~rootDot!fqse)~U(%27U%27Os_dataCjob1CQCjob%202CQCsweep_OsC*OMJrJ*LMFmateriq%20indexE0B1B2XLKn_qlWpeakInd80-formatStrK%25.3fWiteraNng%22%3A*TrueJfY0FReflecNonE449B556B6G5B02B04Hmean%22Y1FAbsorpNonE549B656B3G12B09B01Hpeak%22JQJNtleZ%22device_rta_sweep%22%27)~version!%271%27)*%20%20-J**%22KZB%2C%200.C%5Cn*EWz_vqueVF-zMKG17Xz_stdevV0HXstaNsNcsKJC**K8%22Lshort_formM_nameNtiOplotQ*...Usource!Vs8%5B0.W%22-X%5D-YJ*line_Z%3A%20qalzvar%01zqZYXWVUQONMLKJHGFECB8-*
# Broadly, the plot_data has keys 'r', 'f', and 'title'.
# The values of 'r' and 'f' are a dictionary and array of dictionaries respectively, that contain plot metadata as well as values

if __name__ == "__main__":
	plot_directory_location = 'plots'
	cutoff_1d_sweep_offset = [0, 0]

	#* This is where we perturb the data points as needed
	# TODO: modify data code

	#* Call plotting function and export
	plot_sorting_transmission_spectrum(template_plot_data, job_idx)
	plot_sorting_transmission_sweep_1d(template_plot_data, [slice(None), 0])